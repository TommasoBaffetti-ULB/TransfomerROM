from __future__ import annotations

from typing import TYPE_CHECKING

from pytorch_scheduler.base.scheduler import BaseScheduler

if TYPE_CHECKING:
    from torch.optim import Optimizer


class PowerDecayScheduler(BaseScheduler):
    """Power-law learning rate decay schedule (scaling laws).

    Inspired by the empirical scaling laws of Kaplan et al. (2020), which
    suggest that the optimal LR follows a power-law decay during training.

    Phases:
        Warmup  (0 ≤ step < warmup_steps):
            Linear ramp from 0 → base_lr.
            If warmup_steps = 0, training starts directly at base_lr.
        Decay   (step ≥ warmup_steps):
            lr = max(base_lr * (step / warmup_steps) ^ (-alpha), min_lr)

    Special case: if warmup_steps = 0, decay starts from step 1 using
        lr = max(base_lr * (step + 1) ^ (-alpha), min_lr)
    to avoid division by zero at step 0.

    Args:
        optimizer:    Wrapped optimizer.
        total_steps:  Total number of training steps (used only for
                      validation and boundary handling).
        warmup_steps: Steps for linear warmup (default 0).
        alpha:        Power-law exponent; 0.5 = inverse square root,
                      1.0 = inverse linear (default 0.5).
        min_lr:       Minimum LR floor; the LR is clamped to this value
                      (default 0.0).
        last_epoch:   Index of the last step (-1 = before first step).

    Reference:
        Paper: "Scaling Laws for Neural Language Models"
               Kaplan et al., 2020
        URL: https://arxiv.org/abs/2001.08361
    """

    paper_title = "Scaling Laws for Neural Language Models"
    paper_url = "https://arxiv.org/abs/2001.08361"
    paper_year = 2020
    needs_total_steps = False

    def __init__(
        self,
        optimizer: Optimizer,
        total_steps: int,
        warmup_steps: int = 0,
        alpha: float = 0.5,
        min_lr: float = 0.0,
        last_epoch: int = -1,
    ) -> None:
        if total_steps <= 0:
            raise ValueError(f"total_steps must be positive, got {total_steps}")
        if alpha <= 0:
            raise ValueError(f"alpha must be positive, got {alpha}")
        self._validate_positive(min_lr, "min_lr")
        if warmup_steps < 0:
            raise ValueError(f"warmup_steps must be non-negative, got {warmup_steps}")
        if warmup_steps >= total_steps:
            raise ValueError(f"warmup_steps ({warmup_steps}) must be less than total_steps ({total_steps})")

        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        self.alpha = alpha
        self.min_lr = min_lr

        super().__init__(optimizer, last_epoch=last_epoch)

    def _power_lr(self, base_lr: float, step: int) -> float:
        """Compute the power-law LR for a given step (after warmup)."""
        if self.warmup_steps > 0:
            # Reference point: at step == warmup_steps, factor = 1.0 → base_lr
            factor = (step / self.warmup_steps) ** (-self.alpha)
        else:
            # No warmup: use step+1 to keep step=0 from causing divide-by-zero
            # Normalise so that at step=0 the factor equals 1.0 → base_lr
            factor = 1.0 / (step + 1) ** self.alpha

        lr = base_lr * factor
        return max(lr, self.min_lr)

    def _lr_at(self, step: int, base_lrs: list[float]) -> list[float]:
        # Step 0 (before any update)
        if step <= 0:
            if self.warmup_steps > 0:
                return [0.0 for _ in base_lrs]
            # No warmup: power law from step 0 onward (returns base_lr at step=0)
            return list(base_lrs)

        # Warmup phase
        if self.warmup_steps > 0 and step < self.warmup_steps:
            progress = step / self.warmup_steps
            return [base_lr * progress for base_lr in base_lrs]

        # Power-law decay phase
        return [self._power_lr(base_lr, step) for base_lr in base_lrs]
