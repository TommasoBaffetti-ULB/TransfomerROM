from __future__ import annotations

from typing import TYPE_CHECKING

from pytorch_scheduler.base.scheduler import BaseScheduler

if TYPE_CHECKING:
    from torch.optim import Optimizer


class RexScheduler(BaseScheduler):
    """REX: Revisiting Budgeted Training with an Improved Schedule.

    Formula: lr = base_lr * (1 - t) / (1 - t/2)
    where t = step / total_steps

    Reference:
        Paper: "Revisiting Budgeted Training with an Improved Schedule" (MLSys 2022)
        URL: https://arxiv.org/abs/2107.04197
    """

    paper_title = "Revisiting Budgeted Training with an Improved Schedule"
    paper_url = "https://arxiv.org/abs/2107.04197"
    paper_year = 2022
    needs_total_steps = True

    def __init__(
        self,
        optimizer: Optimizer,
        total_steps: int,
        last_epoch: int = -1,
    ) -> None:
        if total_steps <= 0:
            raise ValueError(f"total_steps must be positive, got {total_steps}")
        self.total_steps = total_steps
        super().__init__(optimizer, last_epoch=last_epoch)

    def _lr_at(self, step: int, base_lrs: list[float]) -> list[float]:
        # At step 0, return base_lrs unchanged (t=0 → factor=1)
        if step <= 0:
            return list(base_lrs)

        # After total_steps, t=1 → numerator=0, so lr=0
        if step >= self.total_steps:
            return [0.0 for _ in base_lrs]

        t = step / self.total_steps
        # Denominator is always > 0 for t in [0, 1): (1 - t/2) >= 0.5
        factor = (1.0 - t) / (1.0 - t / 2.0)
        return [base_lr * factor for base_lr in base_lrs]
