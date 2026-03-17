from __future__ import annotations

import math
from typing import TYPE_CHECKING

from pytorch_scheduler.base.scheduler import BaseScheduler

if TYPE_CHECKING:
    from torch.optim import Optimizer


class WarmupHoldCosineScheduler(BaseScheduler):
    """Three-phase schedule: warmup → hold at peak LR → cosine decay.

    Commonly used in large-scale LLM pretraining (MiniCPM, LLaMA-style).
    Maximizes time at peak LR before a final cosine cooldown.

    Formula:
        Warmup (0 ≤ t < warmup_steps):
            lr = base_lr * t / warmup_steps
        Hold (warmup_steps ≤ t < warmup_steps + hold_steps):
            lr = base_lr
        Cosine decay (warmup_steps + hold_steps ≤ t ≤ total_steps):
            lr = min_lr + 0.5 * (base_lr - min_lr) * (1 + cos(π * progress))
            where progress = (t - warmup_steps - hold_steps) / (total_steps - warmup_steps - hold_steps)

    Args:
        optimizer:    Wrapped optimizer.
        total_steps:  Total number of training steps.
        warmup_steps: Steps for linear warmup (default 0).
        hold_steps:   Steps to hold at peak LR (default 0).
        min_lr:       Minimum LR at end of cosine decay (default 0.0).
        last_epoch:   Index of the last step (-1 = before first step).
    """

    paper_title = ""
    paper_url = ""
    paper_year = 0
    needs_total_steps = True

    def __init__(
        self,
        optimizer: Optimizer,
        total_steps: int,
        warmup_steps: int = 0,
        hold_steps: int = 0,
        min_lr: float = 0.0,
        last_epoch: int = -1,
    ) -> None:
        if total_steps <= 0:
            raise ValueError(f"total_steps must be positive, got {total_steps}")
        if warmup_steps < 0:
            raise ValueError(f"warmup_steps must be non-negative, got {warmup_steps}")
        if hold_steps < 0:
            raise ValueError(f"hold_steps must be non-negative, got {hold_steps}")
        if warmup_steps + hold_steps >= total_steps:
            raise ValueError(
                f"warmup_steps ({warmup_steps}) + hold_steps ({hold_steps}) "
                f"must be < total_steps ({total_steps}) to allow at least one decay step"
            )
        self._validate_positive(min_lr, "min_lr")

        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        self.hold_steps = hold_steps
        self.min_lr = min_lr

        self._hold_end = warmup_steps + hold_steps
        self._decay_steps = total_steps - self._hold_end  # ≥ 1 by constraint

        super().__init__(optimizer, last_epoch=last_epoch)

    def _lr_at(self, step: int, base_lrs: list[float]) -> list[float]:
        if step <= 0:
            if self.warmup_steps > 0:
                return [0.0 for _ in base_lrs]
            return list(base_lrs)

        if step >= self.total_steps:
            return [self.min_lr for _ in base_lrs]

        if step < self.warmup_steps:
            # Linear warmup: 0 → base_lr
            progress = step / self.warmup_steps
            return [base_lr * progress for base_lr in base_lrs]

        if step < self._hold_end:
            # Hold phase: constant at base_lr
            return list(base_lrs)

        # Cosine decay: base_lr → min_lr
        progress = (step - self._hold_end) / self._decay_steps
        # Clamp to [0, 1] for numerical safety
        progress = max(0.0, min(1.0, progress))
        cosine_factor = 0.5 * (1.0 + math.cos(math.pi * progress))
        return [self.min_lr + (base_lr - self.min_lr) * cosine_factor for base_lr in base_lrs]
