from __future__ import annotations

import math
from typing import TYPE_CHECKING

from pytorch_scheduler.base.scheduler import BaseScheduler

if TYPE_CHECKING:
    from torch.optim import Optimizer


class SlantedTriangularScheduler(BaseScheduler):
    """Slanted Triangular learning rate schedule (ULMFiT).

    Short linear warmup followed by a longer linear decay. Proposed by Howard
    & Ruder for discriminative fine-tuning of language models.

    Phases:
        cut_point = floor(total_steps * cut_frac)

        Warmup  (0 ≤ step < cut_point):
            p = step / cut_point
        Decay   (cut_point ≤ step ≤ total_steps):
            p = 1 - (step - cut_point) / (total_steps - cut_point)

        lr = base_lr * (1 + p * (ratio - 1)) / ratio

    Args:
        optimizer:   Wrapped optimizer.
        total_steps: Total number of training steps.
        cut_frac:    Fraction of steps used for warmup (default 0.1).
        ratio:       Controls the steepness of the decay; the LR rises from
                     base_lr/ratio to base_lr during warmup (default 32.0).
        last_epoch:  Index of the last step (-1 = before first step).

    Reference:
        Paper: "Universal Language Model Fine-tuning for Text Classification"
               Howard & Ruder, 2018
        URL: https://arxiv.org/abs/1801.06146
    """

    paper_title = "Universal Language Model Fine-tuning for Text Classification"
    paper_url = "https://arxiv.org/abs/1801.06146"
    paper_year = 2018
    needs_total_steps = True

    def __init__(
        self,
        optimizer: Optimizer,
        total_steps: int,
        cut_frac: float = 0.1,
        ratio: float = 32.0,
        last_epoch: int = -1,
    ) -> None:
        if total_steps <= 0:
            raise ValueError(f"total_steps must be positive, got {total_steps}")
        if not (0.0 < cut_frac < 1.0):
            raise ValueError(f"cut_frac must be in (0, 1), got {cut_frac}")
        if ratio <= 0:
            raise ValueError(f"ratio must be positive, got {ratio}")

        self.total_steps = total_steps
        self.cut_frac = cut_frac
        self.ratio = ratio
        self._cut_point = math.floor(total_steps * cut_frac)

        super().__init__(optimizer, last_epoch=last_epoch)

    def _compute_p(self, step: int) -> float:
        """Compute the triangular progress value p for a given step."""
        cut_point = self._cut_point

        if step <= 0:
            return 0.0

        if step < cut_point:
            # Warmup phase
            return step / cut_point

        # Decay phase
        decay_steps = self.total_steps - cut_point
        if decay_steps <= 0:
            # Degenerate: entire schedule is warmup, stay at peak
            return 1.0
        p = 1.0 - (step - cut_point) / decay_steps
        return max(0.0, p)

    def _lr_at(self, step: int, base_lrs: list[float]) -> list[float]:
        # End of schedule: p at exactly total_steps
        p = self._compute_p(self.total_steps) if step >= self.total_steps else self._compute_p(step)

        factor = (1.0 + p * (self.ratio - 1.0)) / self.ratio
        # Clamp to [0, 1] for numerical safety
        factor = max(0.0, min(1.0, factor))
        return [base_lr * factor for base_lr in base_lrs]
