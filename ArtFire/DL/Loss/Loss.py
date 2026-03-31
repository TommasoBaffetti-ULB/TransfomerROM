from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict

import torch
import torch.nn as nn


class Loss(nn.Module, ABC):
    """
    Abstract base class for all losses in ArtFire.

    Every concrete loss must return a dictionary containing at least:
        {
            "loss": scalar tensor
        }

    Additional named terms can also be returned, e.g.:
        {
            "loss": ...,
            "reconstruction_loss": ...,
            "aux_loss": ...
        }
    """

    def __init__(self, name: str) -> None:
        super().__init__()
        self.name = name

    @abstractmethod
    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute the loss.

        Args:
            prediction: Model prediction.
            target: Ground-truth target.
            **kwargs: Optional extra arguments for specialized losses.

        Returns:
            Dictionary of loss terms.
        """
        pass

    @staticmethod
    def _validate_shapes(
        prediction: torch.Tensor,
        target: torch.Tensor,
    ) -> None:
        if prediction.shape != target.shape:
            raise ValueError(
                f"Prediction and target must have the same shape, "
                f"got {prediction.shape} and {target.shape}."
            )
