from __future__ import annotations

import math
from typing import TYPE_CHECKING, Literal

from pytorch_scheduler.base.scheduler import BaseScheduler

if TYPE_CHECKING:
    from torch.optim import Optimizer
    from torch.optim.lr_scheduler import LRScheduler


class WarmupScheduler(BaseScheduler):
    """Composable warmup wrapper for any LR scheduler.

    During warmup phase (steps 0 to warmup_steps-1), LR increases from min_lr to max_lr.
    After warmup, delegates to the wrapped base_scheduler.

    Args:
        optimizer: Wrapped optimizer.
        base_scheduler: The scheduler to use after warmup completes.
        min_lr: Starting LR at step 0.
        warmup_steps: Number of warmup steps.
        warmup_type: Type of warmup curve ('linear', 'cosine', 'exponential').
        last_epoch: The index of last epoch.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        base_scheduler: LRScheduler,
        min_lr: float,
        warmup_steps: int,
        warmup_type: Literal["linear", "cosine", "exponential"] = "linear",
        last_epoch: int = -1,
    ) -> None:
        self.base_scheduler = base_scheduler
        self.warmup_steps = warmup_steps
        self.warmup_type = warmup_type
        self.min_lr = min_lr

        super().__init__(optimizer, last_epoch=last_epoch)

    def _warmup_factor(self, step: int) -> float:
        """Returns a scaling factor in [0, 1] representing warmup progress."""
        if self.warmup_steps == 0 or step >= self.warmup_steps:
            return 1.0

        progress = step / self.warmup_steps

        if self.warmup_type == "linear":
            return progress
        elif self.warmup_type == "cosine":
            return 0.5 * (1.0 - math.cos(math.pi * progress))
        elif self.warmup_type == "exponential":
            return progress**2
        else:
            raise ValueError(f"Unknown warmup type: {self.warmup_type}")

    def _lr_at(self, step: int, base_lrs: list[float]) -> list[float]:
        if step < self.warmup_steps:
            # Interpolate each param group's LR from min_lr → max_lr independently.
            # base_lrs carries the per-group max target; min_lr is the floor.
            factor = self._warmup_factor(step)
            return [
                self.min_lr + (max_lr - self.min_lr) * factor for max_lr in base_lrs
            ]

        # Warmup complete — hand off to the wrapped scheduler.
        adjusted_step = step - self.warmup_steps
        if isinstance(self.base_scheduler, BaseScheduler):
            return self.base_scheduler._lr_at(adjusted_step, base_lrs)

        # Fallback for stock PyTorch schedulers: temporarily set last_epoch.
        self.base_scheduler.last_epoch = adjusted_step
        return [float(lr) for lr in self.base_scheduler.get_lr()]

    # ------------------------------------------------------------------ #
    #  Serialisation                                                       #
    # ------------------------------------------------------------------ #

    def state_dict(self) -> dict:  # type: ignore[override]
        state = super().state_dict()
        state["base_scheduler"] = self.base_scheduler.state_dict()
        return state

    def load_state_dict(self, state_dict: dict) -> None:  # type: ignore[override]
        base_state = state_dict.pop("base_scheduler")
        super().load_state_dict(state_dict)
        self.base_scheduler.load_state_dict(base_state)
