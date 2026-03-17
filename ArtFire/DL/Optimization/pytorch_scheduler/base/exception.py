from __future__ import annotations


class NegativeLRError(ValueError):
    """Raised when learning rate is negative."""

    def __init__(self, lr: float) -> None:
        super().__init__(f"Learning rate must be non-negative, got {lr}")


class NegativeStepError(ValueError):
    """Raised when step count is negative."""

    def __init__(self, step: int) -> None:
        super().__init__(f"Step must be non-negative, got {step}")
