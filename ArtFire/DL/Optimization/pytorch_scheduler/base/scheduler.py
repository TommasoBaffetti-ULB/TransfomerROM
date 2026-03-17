from __future__ import annotations

import abc
from typing import TYPE_CHECKING

from torch.optim.lr_scheduler import LRScheduler

if TYPE_CHECKING:
    from torch.optim import Optimizer


class BaseScheduler(LRScheduler, abc.ABC):
    """Base class for all custom schedulers in pytorch_scheduler.

    Subclasses must implement `_lr_at()`.
    """

    paper_title: str = ""
    paper_url: str = ""
    paper_year: int = 0

    # Step semantics metadata
    step_unit: str = "step"  # 'step' | 'epoch' | 'metric'
    needs_total_steps: bool = False
    needs_metric: bool = False

    def __init__(self, optimizer: Optimizer, last_epoch: int = -1) -> None:
        super().__init__(optimizer, last_epoch=last_epoch)

    @abc.abstractmethod
    def _lr_at(self, step: int, base_lrs: list[float]) -> list[float]:
        """Compute learning rates for the given step (pure, stateless).

        This is the core mathematical logic of the scheduler, separated from
        state management. Subclasses must implement this method.

        Args:
            step: The current step number.
            base_lrs: The base learning rates for each parameter group.

        Returns:
            A list of learning rates, one per parameter group.
        """
        ...

    def get_lr(self) -> list[float]:
        return self._lr_at(self.last_epoch, [float(lr) for lr in self.base_lrs])

    def validate_config(self, total_training_steps: int | None = None) -> None:
        """Validate scheduler configuration and warn about potential misuse.

        Args:
            total_training_steps: If provided, used to cross-check total_steps.
        """
        import warnings

        if total_training_steps is not None:
            steps_to_check = total_training_steps
        else:
            steps_to_check = getattr(self, "total_steps", None)
        if self.needs_total_steps and steps_to_check is not None and steps_to_check < 100:
            warnings.warn(
                f"{self.__class__.__name__} has total_steps={steps_to_check} which seems low. "
                f"If you're passing epochs instead of steps, use "
                f"total_steps = epochs * steps_per_epoch instead.",
                stacklevel=2,
            )

    def _validate_positive(self, value: float, name: str) -> None:
        if value < 0:
            raise ValueError(f"{name} must be non-negative, got {value}")

    def _validate_steps(self, total_steps: int, warmup_steps: int = 0) -> None:
        if total_steps <= 0:
            raise ValueError(f"total_steps must be positive, got {total_steps}")
        if warmup_steps < 0:
            raise ValueError(f"warmup_steps must be non-negative, got {warmup_steps}")
        if warmup_steps >= total_steps:
            raise ValueError(f"warmup_steps ({warmup_steps}) must be less than total_steps ({total_steps})")
