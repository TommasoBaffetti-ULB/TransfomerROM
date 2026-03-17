from __future__ import annotations

import math
from typing import TYPE_CHECKING

from pytorch_scheduler.base.scheduler import BaseScheduler

if TYPE_CHECKING:
    from torch.optim import Optimizer


class InverseSqrtScheduler(BaseScheduler):
    """Inverse Square Root learning rate schedule from "Attention is All You Need".

    During warmup (step < warmup_steps): linear warmup from 0 to base_lr.
    After warmup: lr = base_lr * sqrt(warmup_steps / step)

    Unified formula (Vaswani et al.):
        lr = base_lr * min(step^{-0.5}, step * warmup_steps^{-1.5})

    Note:
        This scheduler has warmup built into its mathematical formula
        (the ``min(step^{-0.5}, step * warmup_steps^{-1.5})`` term).
        Do **not** wrap it with ``WarmupScheduler`` — doing so would
        apply warmup twice and distort the intended learning-rate curve.

    Reference:
        Paper: "Attention is All You Need" (NeurIPS 2017)
        URL: https://arxiv.org/abs/1706.03762
    """

    paper_title = "Attention is All You Need"
    paper_url = "https://arxiv.org/abs/1706.03762"
    paper_year = 2017
    needs_total_steps = False

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int,
        last_epoch: int = -1,
    ) -> None:
        if warmup_steps <= 0:
            raise ValueError(f"warmup_steps must be positive, got {warmup_steps}")
        self.warmup_steps = warmup_steps
        super().__init__(optimizer, last_epoch=last_epoch)

    def _lr_at(self, step: int, base_lrs: list[float]) -> list[float]:
        # At step 0, avoid division by zero: return 0 (warmup not yet started)
        if step <= 0:
            return [0.0 for _ in base_lrs]

        # min(step^{-0.5}, step * warmup_steps^{-1.5})
        # = min(1/sqrt(step), step / warmup_steps^{1.5})
        inv_sqrt_step = step**-0.5
        linear_warmup = step * (self.warmup_steps**-1.5)
        factor = min(inv_sqrt_step, linear_warmup)

        # Normalise so that at step == warmup_steps both branches equal
        # warmup_steps^{-0.5}, and base_lr corresponds to that peak value.
        # The factor already encodes the full scale; multiply by warmup_steps^{0.5}
        # so that the peak at warmup_steps equals base_lr.
        normalised_factor = factor * math.sqrt(self.warmup_steps)

        return [base_lr * normalised_factor for base_lr in base_lrs]
