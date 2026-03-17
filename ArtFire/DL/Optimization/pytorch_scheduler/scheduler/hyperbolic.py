from __future__ import annotations

import math
from typing import TYPE_CHECKING

from pytorch_scheduler.base.scheduler import BaseScheduler

if TYPE_CHECKING:
    from torch.optim import Optimizer


class HyperbolicLRScheduler(BaseScheduler):
    """Hyperbolic Learning Rate Scheduler.

    Decays the learning rate along a hyperbolic curve, providing
    epoch-insensitive behavior: initial LR changes remain consistent
    regardless of total training length.

    During warmup (step <= warmup_steps): linear ramp from min_lr to base_lr.
    After warmup: hyperbolic decay from base_lr toward min_lr.

    Formula (after warmup):
        lr = base_lr + (base_lr - min_lr) * (f(x) - f(0))
        where f(x) = sqrt((N - x) / U * (2 - (N + x) / U))
              x = step - warmup_steps
              N = total_steps - warmup_steps
              U = upper_bound

    Args:
        optimizer:     Wrapped optimizer.
        total_steps:   Total number of training steps.
        upper_bound:   Upper bound parameter controlling decay curvature.
                       Must be >= total_steps.
        min_lr:        Minimum (infimum) learning rate (default 1e-6).
        warmup_steps:  Linear warmup steps from min_lr to base_lr (default 0).
        last_epoch:    Index of last step.

    Reference:
        Paper: "HyperbolicLR: Epoch Insensitive Learning Rate Scheduler" (2024)
        URL: https://arxiv.org/abs/2407.15200
    """

    paper_title = "HyperbolicLR: Epoch Insensitive Learning Rate Scheduler"
    paper_url = "https://arxiv.org/abs/2407.15200"
    paper_year = 2024
    needs_total_steps = True

    def __init__(
        self,
        optimizer: Optimizer,
        total_steps: int,
        upper_bound: int,
        min_lr: float = 1e-6,
        warmup_steps: int = 0,
        last_epoch: int = -1,
    ) -> None:
        if total_steps <= 0:
            raise ValueError(f"total_steps must be positive, got {total_steps}")
        if warmup_steps < 0:
            raise ValueError(f"warmup_steps must be non-negative, got {warmup_steps}")
        if warmup_steps >= total_steps:
            raise ValueError(f"warmup_steps ({warmup_steps}) must be less than total_steps ({total_steps})")
        if upper_bound < total_steps:
            raise ValueError(f"upper_bound ({upper_bound}) must be >= total_steps ({total_steps})")
        if min_lr < 0:
            raise ValueError(f"min_lr must be non-negative, got {min_lr}")

        self.total_steps = total_steps
        self.upper_bound = upper_bound
        self.min_lr = min_lr
        self.warmup_steps = warmup_steps
        self.decay_steps = total_steps - warmup_steps

        super().__init__(optimizer, last_epoch=last_epoch)

    def _hyperbolic_factor(self, x_decay: float) -> float:
        """Compute f(x) = sqrt((N - x) / U * (2 - (N + x) / U))."""
        n = self.decay_steps
        u = self.upper_bound
        return math.sqrt((n - x_decay) / u * (2.0 - (n + x_decay) / u))

    def _lr_at(self, step: int, base_lrs: list[float]) -> list[float]:
        if step <= 0:
            if self.warmup_steps > 0:
                return [self.min_lr for _ in base_lrs]
            return list(base_lrs)

        if self.warmup_steps > 0 and step <= self.warmup_steps:
            progress = step / self.warmup_steps
            return [self.min_lr + (base_lr - self.min_lr) * progress for base_lr in base_lrs]

        x_decay = min(step - self.warmup_steps, self.decay_steps)
        f_x = self._hyperbolic_factor(x_decay)
        f_0 = self._hyperbolic_factor(0)

        return [base_lr + (base_lr - self.min_lr) * (f_x - f_0) for base_lr in base_lrs]


class ExpHyperbolicLRScheduler(BaseScheduler):
    """Exponential Hyperbolic Learning Rate Scheduler.

    Variant of HyperbolicLR that applies the hyperbolic decay in log-space,
    producing an exponential decay curve.

    During warmup (step <= warmup_steps): linear ramp from min_lr to base_lr.
    After warmup: exponential hyperbolic decay from base_lr toward min_lr.

    Formula (after warmup):
        lr = base_lr * (base_lr / min_lr) ^ (f(x) - f(0))
        where f(x) = sqrt((N - x) / U * (2 - (N + x) / U))
              x = step - warmup_steps
              N = total_steps - warmup_steps
              U = upper_bound

    Args:
        optimizer:     Wrapped optimizer.
        total_steps:   Total number of training steps.
        upper_bound:   Upper bound parameter controlling decay curvature.
                       Must be >= total_steps.
        min_lr:        Minimum (infimum) learning rate (default 1e-6).
                       Must be positive (used as divisor).
        warmup_steps:  Linear warmup steps from min_lr to base_lr (default 0).
        last_epoch:    Index of last step.

    Reference:
        Paper: "HyperbolicLR: Epoch Insensitive Learning Rate Scheduler" (2024)
        URL: https://arxiv.org/abs/2407.15200
    """

    paper_title = "HyperbolicLR: Epoch Insensitive Learning Rate Scheduler"
    paper_url = "https://arxiv.org/abs/2407.15200"
    paper_year = 2024
    needs_total_steps = True

    def __init__(
        self,
        optimizer: Optimizer,
        total_steps: int,
        upper_bound: int,
        min_lr: float = 1e-6,
        warmup_steps: int = 0,
        last_epoch: int = -1,
    ) -> None:
        if total_steps <= 0:
            raise ValueError(f"total_steps must be positive, got {total_steps}")
        if warmup_steps < 0:
            raise ValueError(f"warmup_steps must be non-negative, got {warmup_steps}")
        if warmup_steps >= total_steps:
            raise ValueError(f"warmup_steps ({warmup_steps}) must be less than total_steps ({total_steps})")
        if upper_bound < total_steps:
            raise ValueError(f"upper_bound ({upper_bound}) must be >= total_steps ({total_steps})")
        if min_lr <= 0:
            raise ValueError(f"min_lr must be positive, got {min_lr}")

        self.total_steps = total_steps
        self.upper_bound = upper_bound
        self.min_lr = min_lr
        self.warmup_steps = warmup_steps
        self.decay_steps = total_steps - warmup_steps

        super().__init__(optimizer, last_epoch=last_epoch)

    def _hyperbolic_factor(self, x_decay: float) -> float:
        """Compute f(x) = sqrt((N - x) / U * (2 - (N + x) / U))."""
        n = self.decay_steps
        u = self.upper_bound
        return math.sqrt((n - x_decay) / u * (2.0 - (n + x_decay) / u))

    def _lr_at(self, step: int, base_lrs: list[float]) -> list[float]:
        if step <= 0:
            if self.warmup_steps > 0:
                return [self.min_lr for _ in base_lrs]
            return list(base_lrs)

        if self.warmup_steps > 0 and step <= self.warmup_steps:
            progress = step / self.warmup_steps
            return [self.min_lr + (base_lr - self.min_lr) * progress for base_lr in base_lrs]

        x_decay = min(step - self.warmup_steps, self.decay_steps)
        f_x = self._hyperbolic_factor(x_decay)
        f_0 = self._hyperbolic_factor(0)
        exponent = f_x - f_0

        return [base_lr * (base_lr / self.min_lr) ** exponent for base_lr in base_lrs]
