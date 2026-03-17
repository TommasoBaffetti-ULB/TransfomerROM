from __future__ import annotations

from typing import TYPE_CHECKING

from pytorch_scheduler.base.scheduler import BaseScheduler

if TYPE_CHECKING:
    from torch.optim import Optimizer


class LinearDecayScheduler(BaseScheduler):
    """Linear Decay to Zero (D2Z) schedule.

    If warmup_steps > 0:
        Phase 1 (0 ≤ step < warmup_steps):  linear ramp from 0 → base_lr
        Phase 2 (warmup_steps ≤ step ≤ total_steps): linear decay from base_lr → min_lr

    If warmup_steps == 0:
        Single phase: linear decay from base_lr → min_lr over total_steps.

    Reference:
        Paper: "Scaling Laws and Compute-Optimal Training Beyond Fixed Training Durations"
               Hägele et al., 2024
        URL: https://arxiv.org/abs/2405.18392
    """

    paper_title = "Scaling Laws and Compute-Optimal Training Beyond Fixed Training Durations"
    paper_url = "https://arxiv.org/abs/2405.18392"
    paper_year = 2024
    needs_total_steps = True

    def __init__(
        self,
        optimizer: Optimizer,
        total_steps: int,
        warmup_steps: int = 0,
        min_lr: float = 0.0,
        last_epoch: int = -1,
    ) -> None:
        if warmup_steps > 0:
            self._validate_steps(total_steps, warmup_steps)
        else:
            if total_steps <= 0:
                raise ValueError(f"total_steps must be positive, got {total_steps}")
        self._validate_positive(min_lr, "min_lr")

        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch=last_epoch)

    def _lr_at(self, step: int, base_lrs: list[float]) -> list[float]:
        if step <= 0:
            if self.warmup_steps > 0:
                # Warmup starts from 0
                return [0.0 for _ in base_lrs]
            else:
                return list(base_lrs)

        if step >= self.total_steps:
            return [self.min_lr for _ in base_lrs]

        if step < self.warmup_steps:
            # Linear warmup: 0 → base_lr
            progress = step / self.warmup_steps
            return [base_lr * progress for base_lr in base_lrs]

        # Linear decay: base_lr → min_lr
        decay_steps = self.total_steps - self.warmup_steps
        progress = (step - self.warmup_steps) / decay_steps
        return [base_lr + (self.min_lr - base_lr) * progress for base_lr in base_lrs]
