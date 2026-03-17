from __future__ import annotations

import math
from typing import TYPE_CHECKING

from pytorch_scheduler.base.scheduler import BaseScheduler

if TYPE_CHECKING:
    from torch.optim import Optimizer


class TanhDecayScheduler(BaseScheduler):
    """Tanh-based learning rate decay schedule.

    Applies a hyperbolic tangent decay curve. The `steepness` parameter
    controls how sharp the transition from base_lr to min_lr is:
      - Higher steepness → more abrupt, step-function-like transition
      - Lower steepness  → gentler, more gradual decay

    Formula:
        t  = step / total_steps       (progress ∈ [0, 1])
        lr = min_lr + 0.5 * (base_lr - min_lr) * (1 - tanh(steepness * (2*t - 1)))

    Reference:
        Paper: "Online Learning Rate Adaptation with Hypergradient Descent"
               Baydin et al., 2018
        URL: https://arxiv.org/abs/1703.04782
    """

    paper_title = "Online Learning Rate Adaptation with Hypergradient Descent"
    paper_url = "https://arxiv.org/abs/1703.04782"
    paper_year = 2018
    needs_total_steps = True

    def __init__(
        self,
        optimizer: Optimizer,
        total_steps: int,
        steepness: float = 3.0,
        min_lr: float = 0.0,
        last_epoch: int = -1,
    ) -> None:
        if total_steps <= 0:
            raise ValueError(f"total_steps must be positive, got {total_steps}")
        if steepness <= 0:
            raise ValueError(f"steepness must be positive, got {steepness}")
        self._validate_positive(min_lr, "min_lr")

        self.total_steps = total_steps
        self.steepness = steepness
        self.min_lr = min_lr

        super().__init__(optimizer, last_epoch=last_epoch)

    def _lr_at(self, step: int, base_lrs: list[float]) -> list[float]:
        if step <= 0:
            # At t=0: tanh(steepness * (0 - 1)) = tanh(-steepness)
            # For high steepness this approaches -1, giving factor ≈ 1 → base_lr
            # Use formula directly for consistency
            t = 0.0
            factor = 0.5 * (1.0 - math.tanh(self.steepness * (2.0 * t - 1.0)))
            return [self.min_lr + (base_lr - self.min_lr) * factor for base_lr in base_lrs]

        if step >= self.total_steps:
            return [self.min_lr for _ in base_lrs]

        t = step / self.total_steps
        factor = 0.5 * (1.0 - math.tanh(self.steepness * (2.0 * t - 1.0)))
        return [self.min_lr + (base_lr - self.min_lr) * factor for base_lr in base_lrs]
