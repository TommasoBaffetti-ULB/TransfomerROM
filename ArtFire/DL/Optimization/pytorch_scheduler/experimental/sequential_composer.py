from __future__ import annotations

from typing import TYPE_CHECKING

from torch.optim.lr_scheduler import LRScheduler

from pytorch_scheduler.base.scheduler import BaseScheduler

if TYPE_CHECKING:
    from torch.optim import Optimizer


class SequentialComposer(LRScheduler):
    """Chain multiple schedulers sequentially at specified milestones.

    Each scheduler is responsible for a segment of the training run.
    Unlike PyTorch's SequentialLR, this avoids the step(0) deprecation
    issues and correctly handles state.

    Args:
        optimizer: Wrapped optimizer.
        schedulers: List of LRScheduler instances (must share the same optimizer).
        milestones: List of step indices where transitions occur.
                    len(milestones) == len(schedulers) - 1
                    E.g., milestones=[100, 500] with 3 schedulers means:
                      scheduler[0] for steps 0-99
                      scheduler[1] for steps 100-499
                      scheduler[2] for steps 500+
        last_epoch: Index of last step.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        schedulers: list[LRScheduler],
        milestones: list[int],
        last_epoch: int = -1,
    ) -> None:
        if len(milestones) != len(schedulers) - 1:
            raise ValueError(
                f"Expected {len(schedulers) - 1} milestones for {len(schedulers)} schedulers, got {len(milestones)}"
            )
        if milestones != sorted(milestones):
            raise ValueError(f"milestones must be sorted, got {milestones}")
        if any(m < 0 for m in milestones):
            raise ValueError("milestones must be non-negative")

        self.schedulers = schedulers
        self.milestones = milestones

        super().__init__(optimizer, last_epoch=last_epoch)

    def _get_active_scheduler_and_offset(self, step: int) -> tuple[LRScheduler, int]:
        offset = 0
        for i, milestone in enumerate(self.milestones):
            if step < milestone:
                return self.schedulers[i], offset
            offset = milestone
        # Past all milestones -> last scheduler
        return self.schedulers[-1], offset

    def get_lr(self) -> list[float]:
        step = self.last_epoch
        scheduler, offset = self._get_active_scheduler_and_offset(step)
        adjusted_step = step - offset
        if isinstance(scheduler, BaseScheduler):
            return scheduler._lr_at(adjusted_step, [float(lr) for lr in self.base_lrs])
        # Fallback for non-BaseScheduler instances
        scheduler.last_epoch = adjusted_step
        return [float(lr) for lr in scheduler.get_lr()]

    def state_dict(self) -> dict:  # type: ignore[override]
        state = super().state_dict()
        state["schedulers"] = [s.state_dict() for s in self.schedulers]
        return state

    def load_state_dict(self, state_dict: dict) -> None:  # type: ignore[override]
        scheduler_states = state_dict.pop("schedulers")
        super().load_state_dict(state_dict)
        for scheduler, s_state in zip(self.schedulers, scheduler_states, strict=True):
            scheduler.load_state_dict(s_state)
