from __future__ import annotations

import math
from typing import TYPE_CHECKING

from pytorch_scheduler.base.scheduler import BaseScheduler

if TYPE_CHECKING:
    from torch.optim import Optimizer


class CosineWithWarmupScheduler(BaseScheduler):
    """Cosine annealing with linear warmup (no restarts).

    The most common LR schedule in modern LLM pretraining and ViT training.
    Linear warmup from 0 to base_lr, then cosine decay to min_lr.

    Formula:
        Warmup (0 ≤ t < warmup_steps):
            lr = base_lr * t / warmup_steps
        Cosine decay (warmup_steps ≤ t ≤ total_steps):
            lr = min_lr + 0.5 * (base_lr - min_lr) * (1 + cos(π * (t - warmup_steps) / (total_steps - warmup_steps)))

    Reference:
        This schedule is the de facto standard in modern transformer training.
        Used in GPT-3, LLaMA, ViT, and most large-scale training recipes.
    """

    paper_title = ""
    paper_url = ""
    paper_year = 0
    needs_total_steps = True

    def __init__(
        self,
        optimizer: Optimizer,
        total_steps: int,
        warmup_steps: int = 0,
        min_lr: float = 0.0,
        last_epoch: int = -1,
    ) -> None:
        if total_steps <= 0:
            raise ValueError(f"total_steps must be positive, got {total_steps}")
        if warmup_steps < 0:
            raise ValueError(f"warmup_steps must be non-negative, got {warmup_steps}")
        if warmup_steps >= total_steps:
            raise ValueError(f"warmup_steps ({warmup_steps}) must be less than total_steps ({total_steps})")
        self._validate_positive(min_lr, "min_lr")

        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        self.min_lr = min_lr

        super().__init__(optimizer, last_epoch=last_epoch)

    def _lr_at(self, step: int, base_lrs: list[float]) -> list[float]:
        if step <= 0:
            if self.warmup_steps > 0:
                return [0.0 for _ in base_lrs]
            return list(base_lrs)

        if step >= self.total_steps:
            return [self.min_lr for _ in base_lrs]

        if step < self.warmup_steps:
            # Linear warmup: 0 → base_lr
            progress = step / self.warmup_steps
            return [base_lr * progress for base_lr in base_lrs]

        # Cosine decay: base_lr → min_lr
        cosine_steps = self.total_steps - self.warmup_steps
        progress = (step - self.warmup_steps) / cosine_steps
        # Clamp for numerical safety
        progress = max(0.0, min(1.0, progress))
        cosine_factor = 0.5 * (1.0 + math.cos(math.pi * progress))
        return [self.min_lr + (base_lr - self.min_lr) * cosine_factor for base_lr in base_lrs]
