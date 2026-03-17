from __future__ import annotations

import math
from typing import TYPE_CHECKING

from pytorch_scheduler.base.scheduler import BaseScheduler

if TYPE_CHECKING:
    from torch.optim import Optimizer


class CosineAnnealingWarmupRestarts(BaseScheduler):
    """Cosine Annealing with Warm Restarts and optional linear warmup per cycle.

    Extends the SGDR schedule (Loshchilov & Hutter, 2017) with:
      - A linear warmup phase at the start of each cycle (min_lr → max_lr).
      - Per-cycle max_lr decay controlled by `gamma`.
      - Cycle length growth controlled by `cycle_mult`.

    This scheduler manages its own LR range (max_lr / min_lr) independently of
    the optimizer's param group initial LRs.

    Args:
        optimizer:         Wrapped optimizer.
        first_cycle_steps: Number of steps in the first cycle.
        warmup_steps:      Steps for linear warmup within each cycle (default 0).
        max_lr:            Peak learning rate for the first cycle (default 0.1).
        min_lr:            Minimum / trough learning rate (default 0.001).
        cycle_mult:        Multiplicative factor for cycle length after each restart
                           (default 1.0 → equal-length cycles).
        gamma:             Multiplicative decay for max_lr after each restart
                           (default 1.0 → no decay).
        last_epoch:        Index of the last step (-1 = before first step).

    Reference:
        Paper: "SGDR: Stochastic Gradient Descent with Warm Restarts" (ICLR 2017)
        URL: https://arxiv.org/abs/1608.03983
    """

    paper_title = "SGDR: Stochastic Gradient Descent with Warm Restarts"
    paper_url = "https://arxiv.org/abs/1608.03983"
    paper_year = 2017
    needs_total_steps = False

    def __init__(
        self,
        optimizer: Optimizer,
        first_cycle_steps: int,
        warmup_steps: int = 0,
        max_lr: float = 0.1,
        min_lr: float = 0.001,
        cycle_mult: float = 1.0,
        gamma: float = 1.0,
        last_epoch: int = -1,
    ) -> None:
        if first_cycle_steps <= 0:
            raise ValueError(f"first_cycle_steps must be positive, got {first_cycle_steps}")
        if warmup_steps < 0:
            raise ValueError(f"warmup_steps must be non-negative, got {warmup_steps}")
        if warmup_steps >= first_cycle_steps:
            raise ValueError(
                f"warmup_steps ({warmup_steps}) must be less than first_cycle_steps ({first_cycle_steps})"
            )
        if max_lr < 0:
            raise ValueError(f"max_lr must be non-negative, got {max_lr}")
        if min_lr < 0:
            raise ValueError(f"min_lr must be non-negative, got {min_lr}")
        if max_lr < min_lr:
            raise ValueError(f"max_lr ({max_lr}) must be >= min_lr ({min_lr})")
        if cycle_mult <= 0:
            raise ValueError(f"cycle_mult must be positive, got {cycle_mult}")
        if gamma <= 0:
            raise ValueError(f"gamma must be positive, got {gamma}")

        self.first_cycle_steps = first_cycle_steps
        self.warmup_steps = warmup_steps
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.cycle_mult = cycle_mult
        self.gamma = gamma

        # Tracking state (resolved before super().__init__ calls step())
        self.cur_cycle_steps = first_cycle_steps
        self.cycle = 0
        self.step_in_cycle = last_epoch  # will be reset inside super().__init__

        super().__init__(optimizer, last_epoch=last_epoch)
        # Override any LR set by super().__init__ to start at min_lr
        self._init_lr()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _init_lr(self) -> None:
        """Force all param groups to start at min_lr."""
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = self.min_lr

    def _compute_cycle_and_step(self, epoch: int) -> tuple[int, int, int, float]:
        """Return (cycle_idx, step_in_cycle, cur_cycle_steps, cur_max_lr)."""
        if epoch < 0:
            return 0, -1, self.first_cycle_steps, self.max_lr

        # Walk through cycles to find which one `epoch` falls into.
        cycle = 0
        cycle_start = 0
        cycle_len = self.first_cycle_steps

        while cycle_start + cycle_len <= epoch:
            cycle_start += cycle_len
            cycle += 1
            if self.cycle_mult == 1.0:
                cycle_len = self.first_cycle_steps
            else:
                cycle_len = int(self.first_cycle_steps * (self.cycle_mult**cycle))

        step_in_cycle = epoch - cycle_start
        cur_max_lr = self.max_lr * (self.gamma**cycle)
        return cycle, step_in_cycle, cycle_len, cur_max_lr

    def _lr_for_step(self, step_in_cycle: int, cycle_len: int, cur_max_lr: float) -> float:
        """Compute the LR for a given position within a cycle."""
        if step_in_cycle <= 0:
            return self.min_lr

        if step_in_cycle < self.warmup_steps:
            # Linear warmup: min_lr → cur_max_lr
            progress = step_in_cycle / self.warmup_steps
            return self.min_lr + (cur_max_lr - self.min_lr) * progress

        # Cosine annealing: cur_max_lr → min_lr
        cosine_steps = cycle_len - self.warmup_steps
        if cosine_steps <= 0:
            return cur_max_lr
        progress = (step_in_cycle - self.warmup_steps) / cosine_steps
        cosine_factor = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.min_lr + (cur_max_lr - self.min_lr) * cosine_factor

    # ------------------------------------------------------------------
    # LRScheduler protocol
    # ------------------------------------------------------------------

    def _lr_at(self, step: int, base_lrs: list[float]) -> list[float]:
        _, step_in_cycle, cycle_len, cur_max_lr = self._compute_cycle_and_step(step)
        lr = self._lr_for_step(step_in_cycle, cycle_len, cur_max_lr)
        return [lr for _ in base_lrs]

    def step(self, epoch: int | None = None) -> None:  # type: ignore[override]
        """Advance the scheduler by one step (or to a given epoch index)."""
        super().step(epoch)
        # Keep public state attributes in sync so callers can inspect them.
        e = self.last_epoch
        self.cycle, self.step_in_cycle, self.cur_cycle_steps, _ = self._compute_cycle_and_step(e)
