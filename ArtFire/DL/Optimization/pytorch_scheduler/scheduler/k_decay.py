from __future__ import annotations

import math
from typing import TYPE_CHECKING

from pytorch_scheduler.base.scheduler import BaseScheduler

if TYPE_CHECKING:
    from torch.optim import Optimizer


class KDecayScheduler(BaseScheduler):
    """k-decay learning rate schedule.

    Applies a k-decay modifier to cosine annealing. The exponent k controls
    the sharpness of the decay curve:
      - k = 1  : standard cosine annealing
      - k > 1  : decay is delayed (LR stays high longer before dropping)
      - k < 1  : decay is accelerated (LR drops quickly early on)

    Formula:
        lr = min_lr + 0.5 * (base_lr - min_lr) * (1 + cos(π * (step / total_steps)^k))

    Reference:
        Paper: "k-decay: A New Method for Learning Rate Schedule"
               Zhang & Li, 2020
        URL: https://arxiv.org/abs/2004.05909
    """

    paper_title = "k-decay: A New Method for Learning Rate Schedule"
    paper_url = "https://arxiv.org/abs/2004.05909"
    paper_year = 2020
    needs_total_steps = True

    def __init__(
        self,
        optimizer: Optimizer,
        total_steps: int,
        k: float = 1.0,
        min_lr: float = 0.0,
        last_epoch: int = -1,
    ) -> None:
        if total_steps <= 0:
            raise ValueError(f"total_steps must be positive, got {total_steps}")
        if k <= 0:
            raise ValueError(f"k must be positive, got {k}")
        self._validate_positive(min_lr, "min_lr")

        self.total_steps = total_steps
        self.k = k
        self.min_lr = min_lr

        super().__init__(optimizer, last_epoch=last_epoch)

    def _lr_at(self, step: int, base_lrs: list[float]) -> list[float]:
        if step <= 0:
            return list(base_lrs)

        if step >= self.total_steps:
            return [self.min_lr for _ in base_lrs]

        # t = (step / total_steps)^k, then apply cosine
        t = (step / self.total_steps) ** self.k
        cosine_factor = 0.5 * (1.0 + math.cos(math.pi * t))
        return [self.min_lr + (base_lr - self.min_lr) * cosine_factor for base_lr in base_lrs]
