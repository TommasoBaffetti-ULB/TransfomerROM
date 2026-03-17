from __future__ import annotations

from typing import TYPE_CHECKING

from pytorch_scheduler.base.scheduler import BaseScheduler

if TYPE_CHECKING:
    from torch.optim import Optimizer


class PolynomialScheduler(BaseScheduler):
    """Polynomial learning rate decay with optional warmup and cycling.

    During warmup (step < warmup_steps): linear ramp from 0 → base_lr.
    After warmup: polynomial decay from base_lr → min_lr.

    Formula (after warmup):
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        lr = (base_lr - min_lr) * (1 - progress) ^ power + min_lr

    When `cycle=True`, the polynomial restarts after total_steps so
    training can continue beyond the original schedule duration. Cycling
    uses the step modulo total_steps as the effective step.

    Args:
        optimizer:    Wrapped optimizer.
        total_steps:  Total steps (end of polynomial decay, or period when
                      cycling).
        power:        Exponent for the polynomial decay (default 1.0 → linear).
        min_lr:       Minimum learning rate at the end of decay (default 0.0).
        warmup_steps: Steps for linear warmup at the start (default 0).
        cycle:        Whether to restart the schedule after total_steps
                      (default False).
        last_epoch:   Index of the last step (-1 = before first step).
    """

    paper_title = "Polynomial Learning Rate Decay"
    paper_url = ""
    paper_year = 0
    needs_total_steps = True

    def __init__(
        self,
        optimizer: Optimizer,
        total_steps: int,
        power: float = 1.0,
        min_lr: float = 0.0,
        warmup_steps: int = 0,
        cycle: bool = False,
        last_epoch: int = -1,
    ) -> None:
        if total_steps <= 0:
            raise ValueError(f"total_steps must be positive, got {total_steps}")
        if power <= 0:
            raise ValueError(f"power must be positive, got {power}")
        self._validate_positive(min_lr, "min_lr")
        if warmup_steps < 0:
            raise ValueError(f"warmup_steps must be non-negative, got {warmup_steps}")
        if warmup_steps >= total_steps:
            raise ValueError(f"warmup_steps ({warmup_steps}) must be less than total_steps ({total_steps})")

        self.total_steps = total_steps
        self.power = power
        self.min_lr = min_lr
        self.warmup_steps = warmup_steps
        self.cycle = cycle

        # Number of steps used for polynomial decay
        self._decay_steps = total_steps - warmup_steps

        super().__init__(optimizer, last_epoch=last_epoch)

    def _polynomial_lr(self, base_lr: float, progress: float) -> float:
        """Compute decayed LR at progress ∈ [0, 1]."""
        return (base_lr - self.min_lr) * ((1.0 - progress) ** self.power) + self.min_lr

    def _lr_at(self, step: int, base_lrs: list[float]) -> list[float]:
        if step <= 0:
            if self.warmup_steps > 0:
                return [0.0 for _ in base_lrs]
            return list(base_lrs)

        # Handle cycling: wrap step back into [0, total_steps)
        if self.cycle and step >= self.total_steps:
            step = step % self.total_steps
            # If modulo lands exactly at 0, treat as start of a new cycle
            if step == 0:
                if self.warmup_steps > 0:
                    return [0.0 for _ in base_lrs]
                return list(base_lrs)

        # Without cycling, clamp at the end value
        if not self.cycle and step >= self.total_steps:
            return [self.min_lr for _ in base_lrs]

        # Warmup phase
        if step < self.warmup_steps:
            progress = step / self.warmup_steps
            return [base_lr * progress for base_lr in base_lrs]

        # Polynomial decay phase
        decay_progress = (step - self.warmup_steps) / self._decay_steps
        decay_progress = max(0.0, min(1.0, decay_progress))
        return [self._polynomial_lr(base_lr, decay_progress) for base_lr in base_lrs]
