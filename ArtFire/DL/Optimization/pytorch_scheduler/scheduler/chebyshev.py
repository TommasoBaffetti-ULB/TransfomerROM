from __future__ import annotations

import math
from typing import TYPE_CHECKING

from pytorch_scheduler.base.scheduler import BaseScheduler

if TYPE_CHECKING:
    from torch.optim import Optimizer


class ChebyshevScheduler(BaseScheduler):
    """Chebyshev polynomial learning rate schedule.

    Rather than monotonically decaying the LR, this schedule follows the
    non-monotonic pattern given by Chebyshev polynomial nodes, which are
    theoretically optimal for polynomial interpolation. The oscillating
    pattern can improve convergence by preventing the optimiser from getting
    stuck in local minima.

    For step j ∈ [0, total_steps), the Chebyshev node is:
        x_j = cos(π * (2*j + 1) / (2 * total_steps))

    The learning rate is then mapped from x_j ∈ [-1, 1] to [min_lr, base_lr]:
        lr = min_lr + 0.5 * (base_lr - min_lr) * (1 + x_j)

    After total_steps, the final node value is reused.

    Args:
        optimizer:   Wrapped optimizer.
        total_steps: Total number of training steps (number of Chebyshev nodes).
        min_lr:      Minimum learning rate (default 0.0).
        last_epoch:  Index of the last step (-1 = before first step).
    """

    paper_title = "Chebyshev Learning Rate Schedule"
    paper_url = ""
    paper_year = 0
    needs_total_steps = True

    def __init__(
        self,
        optimizer: Optimizer,
        total_steps: int,
        min_lr: float = 0.0,
        last_epoch: int = -1,
    ) -> None:
        if total_steps <= 0:
            raise ValueError(f"total_steps must be positive, got {total_steps}")
        self._validate_positive(min_lr, "min_lr")

        self.total_steps = total_steps
        self.min_lr = min_lr

        super().__init__(optimizer, last_epoch=last_epoch)

    def _chebyshev_node(self, j: int) -> float:
        """Compute the j-th Chebyshev node in [-1, 1]."""
        return math.cos(math.pi * (2 * j + 1) / (2 * self.total_steps))

    def _lr_at(self, step: int, base_lrs: list[float]) -> list[float]:
        # Clamp step to valid node range [0, total_steps - 1]
        j = max(0, min(step, self.total_steps - 1))

        x_j = self._chebyshev_node(j)
        # Map from [-1, 1] to [min_lr, base_lr]
        scale = 0.5 * (1.0 + x_j)
        return [self.min_lr + (base_lr - self.min_lr) * scale for base_lr in base_lrs]
