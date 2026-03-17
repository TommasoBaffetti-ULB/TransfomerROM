from __future__ import annotations

from typing import TYPE_CHECKING

from pytorch_scheduler.base.scheduler import BaseScheduler

if TYPE_CHECKING:
    from torch.optim import Optimizer


class TrapezoidalScheduler(BaseScheduler):
    """Trapezoidal (warmup → constant → linear decay) learning rate schedule.

    Three phases:
        Phase 1 (0 ≤ step < warmup_steps):
            Linear ramp from 0 → base_lr.
        Phase 2 (warmup_steps ≤ step < total_steps - decay_steps):
            Constant at base_lr.
        Phase 3 (total_steps - decay_steps ≤ step ≤ total_steps):
            Linear decay from base_lr → min_lr.

    Constraint: warmup_steps + decay_steps <= total_steps
    """

    paper_title = "Trapezoidal Learning Rate Schedule"
    paper_url = ""
    paper_year = 0
    needs_total_steps = True

    def __init__(
        self,
        optimizer: Optimizer,
        total_steps: int,
        warmup_steps: int,
        decay_steps: int,
        min_lr: float = 0.0,
        last_epoch: int = -1,
    ) -> None:
        if total_steps <= 0:
            raise ValueError(f"total_steps must be positive, got {total_steps}")
        if warmup_steps < 0:
            raise ValueError(f"warmup_steps must be non-negative, got {warmup_steps}")
        if decay_steps < 0:
            raise ValueError(f"decay_steps must be non-negative, got {decay_steps}")
        if warmup_steps + decay_steps > total_steps:
            raise ValueError(
                f"warmup_steps ({warmup_steps}) + decay_steps ({decay_steps}) must be <= total_steps ({total_steps})"
            )
        self._validate_positive(min_lr, "min_lr")

        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        self.decay_steps = decay_steps
        self.min_lr = min_lr
        # Step at which decay phase begins
        self._decay_start = total_steps - decay_steps
        super().__init__(optimizer, last_epoch=last_epoch)

    def _lr_at(self, step: int, base_lrs: list[float]) -> list[float]:
        if step <= 0:
            # Start of warmup: return 0 if we have a warmup, else base_lr
            if self.warmup_steps > 0:
                return [0.0 for _ in base_lrs]
            return list(base_lrs)

        if step >= self.total_steps:
            return [self.min_lr for _ in base_lrs]

        if step < self.warmup_steps:
            # Phase 1: linear warmup
            progress = step / self.warmup_steps
            return [base_lr * progress for base_lr in base_lrs]

        if step < self._decay_start:
            # Phase 2: constant
            return list(base_lrs)

        # Phase 3: linear decay
        if self.decay_steps == 0:
            return [self.min_lr for _ in base_lrs]

        progress = (step - self._decay_start) / self.decay_steps
        return [base_lr + (self.min_lr - base_lr) * progress for base_lr in base_lrs]
