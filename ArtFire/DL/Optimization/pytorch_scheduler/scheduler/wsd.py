from __future__ import annotations

import math
from typing import TYPE_CHECKING, Literal

from pytorch_scheduler.base.scheduler import BaseScheduler

if TYPE_CHECKING:
    from torch.optim import Optimizer


class WSDScheduler(BaseScheduler):
    """Warmup-Stable-Decay (WSD) learning rate schedule.

    Three phases:
        Phase 1 — Warmup  (0 ≤ step < warmup_steps):
            Linear ramp from 0 → base_lr.
        Phase 2 — Stable  (warmup_steps ≤ step < warmup_steps + stable_steps):
            Constant at base_lr.
        Phase 3 — Decay   (warmup_steps + stable_steps ≤ step ≤ total_steps):
            Smooth decay from base_lr → min_lr using one of:
              'cosine': half-cosine  min_lr + 0.5*(base_lr-min_lr)*(1+cos(π·p))
              'linear': linear       base_lr + (min_lr-base_lr)*p
              'sqrt':   square-root  min_lr + (base_lr-min_lr)*(1-√p)
            where p = progress within the decay phase ∈ [0, 1].

    Constraint: warmup_steps + stable_steps < total_steps
                (at least one decay step is required)

    Reference:
        Paper: "MiniCPM: Scaling Large Language Models with Scalable Strategies"
               Hu et al., 2024
        URL: https://arxiv.org/abs/2404.06395
    """

    paper_title = "MiniCPM: Scaling Large Language Models with Scalable Strategies"
    paper_url = "https://arxiv.org/abs/2404.06395"
    paper_year = 2024
    needs_total_steps = True

    def __init__(
        self,
        optimizer: Optimizer,
        total_steps: int,
        warmup_steps: int,
        stable_steps: int,
        min_lr: float = 0.0,
        decay_type: Literal["cosine", "linear", "sqrt"] = "cosine",
        last_epoch: int = -1,
    ) -> None:
        if total_steps <= 0:
            raise ValueError(f"total_steps must be positive, got {total_steps}")
        if warmup_steps < 0:
            raise ValueError(f"warmup_steps must be non-negative, got {warmup_steps}")
        if stable_steps < 0:
            raise ValueError(f"stable_steps must be non-negative, got {stable_steps}")
        if warmup_steps + stable_steps >= total_steps:
            raise ValueError(
                f"warmup_steps ({warmup_steps}) + stable_steps ({stable_steps}) "
                f"must be < total_steps ({total_steps}) to allow at least one decay step"
            )
        self._validate_positive(min_lr, "min_lr")

        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        self.stable_steps = stable_steps
        self.min_lr = min_lr
        self.decay_type = decay_type

        self._stable_end = warmup_steps + stable_steps
        self._decay_steps = total_steps - self._stable_end  # ≥ 1 by constraint

        super().__init__(optimizer, last_epoch=last_epoch)

    # ------------------------------------------------------------------
    # Decay helpers
    # ------------------------------------------------------------------

    def _decay_lr(self, base_lr: float, progress: float) -> float:
        """Interpolate from base_lr → min_lr according to decay_type."""
        if self.decay_type == "cosine":
            cosine_factor = 0.5 * (1.0 + math.cos(math.pi * progress))
            return self.min_lr + (base_lr - self.min_lr) * cosine_factor
        elif self.decay_type == "linear":
            return base_lr + (self.min_lr - base_lr) * progress
        else:  # 'sqrt'
            return self.min_lr + (base_lr - self.min_lr) * (1.0 - math.sqrt(progress))

    # ------------------------------------------------------------------
    # LRScheduler protocol
    # ------------------------------------------------------------------

    def _lr_at(self, step: int, base_lrs: list[float]) -> list[float]:
        if step <= 0:
            if self.warmup_steps > 0:
                return [0.0 for _ in base_lrs]
            return list(base_lrs)

        if step >= self.total_steps:
            return [self.min_lr for _ in base_lrs]

        if step < self.warmup_steps:
            # Phase 1: linear warmup
            progress = step / self.warmup_steps
            return [base_lr * progress for base_lr in base_lrs]

        if step < self._stable_end:
            # Phase 2: stable
            return list(base_lrs)

        # Phase 3: decay
        progress = (step - self._stable_end) / self._decay_steps
        # Clamp to [0, 1] for numerical safety
        progress = max(0.0, min(1.0, progress))
        return [self._decay_lr(base_lr, progress) for base_lr in base_lrs]
