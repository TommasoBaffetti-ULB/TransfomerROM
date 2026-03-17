from __future__ import annotations

import math
from typing import TYPE_CHECKING

from pytorch_scheduler.base.scheduler import BaseScheduler

if TYPE_CHECKING:
    from torch.optim import Optimizer


class FlatCosineScheduler(BaseScheduler):
    """Flat-then-Cosine learning rate schedule.

    Holds the learning rate constant at base_lr for a flat phase, then
    applies cosine annealing from base_lr down to min_lr for the remainder
    of training.

    Phases:
        flat_end = floor(total_steps * flat_fraction)

        Phase 1 — Flat   (0 ≤ step ≤ flat_end):
            lr = base_lr
        Phase 2 — Cosine (flat_end < step ≤ total_steps):
            progress = (step - flat_end) / (total_steps - flat_end)
            lr = min_lr + 0.5 * (base_lr - min_lr) * (1 + cos(π * progress))

    Args:
        optimizer:      Wrapped optimizer.
        total_steps:    Total number of training steps.
        flat_fraction:  Fraction of steps to hold LR constant (default 0.7).
        min_lr:         Minimum learning rate at the end of cosine decay
                        (default 0.0).
        last_epoch:     Index of the last step (-1 = before first step).
    """

    paper_title = "Flat + Cosine Annealing (common practice)"
    paper_url = ""
    paper_year = 0
    needs_total_steps = True

    def __init__(
        self,
        optimizer: Optimizer,
        total_steps: int,
        flat_fraction: float = 0.7,
        min_lr: float = 0.0,
        last_epoch: int = -1,
    ) -> None:
        if total_steps <= 0:
            raise ValueError(f"total_steps must be positive, got {total_steps}")
        if not (0.0 <= flat_fraction < 1.0):
            raise ValueError(f"flat_fraction must be in [0, 1), got {flat_fraction}")
        self._validate_positive(min_lr, "min_lr")

        self.total_steps = total_steps
        self.flat_fraction = flat_fraction
        self.min_lr = min_lr
        self._flat_end = math.floor(total_steps * flat_fraction)

        super().__init__(optimizer, last_epoch=last_epoch)

    def _lr_at(self, step: int, base_lrs: list[float]) -> list[float]:
        if step >= self.total_steps:
            return [self.min_lr for _ in base_lrs]

        if step <= self._flat_end:
            # Phase 1: constant at base_lr
            return list(base_lrs)

        # Phase 2: cosine annealing from flat_end to total_steps
        cosine_steps = self.total_steps - self._flat_end
        progress = (step - self._flat_end) / cosine_steps
        # Clamp for numerical safety
        progress = max(0.0, min(1.0, progress))
        cosine_factor = 0.5 * (1.0 + math.cos(math.pi * progress))
        return [self.min_lr + (base_lr - self.min_lr) * cosine_factor for base_lr in base_lrs]
