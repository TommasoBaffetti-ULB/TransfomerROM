from __future__ import annotations

from typing import Dict, Literal

import torch
import torch.nn.functional as F

from ArtFire.DL.Loss.Loss import Loss


class TransformerLoss(Loss):
    def __init__(self, type:Literal["MSE","MAE","Huber","SmoothL1"]="MSE", lambda_t1:float=0.7, lambda_t2:float=0.3, reduction: Literal["mean", "sum"] = "mean") -> None:
        super().__init__(name="transformer loss")
        if type == "MSE":
            self.loss=MSEReconstructionLoss(reduction=reduction)
        elif type=="MAE":
            self.loss=MAEReconstructionLoss(reduction=reduction)
        elif type=="Huber":
            self.loss=HuberReconstructionLoss(reduction=reduction)
        elif type=="SmoothL1":
            self.loss=SmoothL1ReconstructionLoss(reduction=reduction)
        else:
            raise ValueError(f"Unknown loss type: {type}")

        self.lambda_t1 = lambda_t1
        self.lambda_t2 = lambda_t2


    def forward(self, prediction: torch.Tensor, target: torch.Tensor, **kwargs):
        loss_t, loss_m=self.loss(prediction=prediction, target=target, **kwargs)
        return loss_t, self.lambda_t2 * loss_m + self.lambda_t1 * loss_t[0]



class MSEReconstructionLoss(Loss):
    """
    Mean Squared Error reconstruction loss for a sequence of images.

    Expected tensor shape:
        [B, T,NT, DT]
    but any matching shape is accepted.
    """

    def __init__(self, reduction: Literal["mean", "sum"] = "mean") -> None:
        super().__init__(name="mse_reconstruction")
        self.reduction = reduction

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        **kwargs,
    ):
        self._validate_shapes(prediction, target)

        loss = (prediction - target) ** 2

        loss_m = loss.mean()

        return (
            loss.mean(dim=(0, 2, 3)),
            loss_m,
        )  # the first one keeps the temporal dimension


class MAEReconstructionLoss(Loss):
    """
    L1 / MAE reconstruction loss for a sequence of images.
    """

    def __init__(self, reduction: Literal["mean", "sum"] = "mean") -> None:
        super().__init__(name="mae_reconstruction")
        self.reduction = reduction

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self._validate_shapes(prediction, target)

        loss = torch.abs(prediction - target)

        loss_m = loss.mean()

        return loss.mean(dim=(0, 2, 3)), loss_m


class HuberReconstructionLoss(Loss):
    """
    Huber / Smooth L1 reconstruction loss for a sequence of images.
    """

    def __init__(
        self,
        delta: float = 1.0,
        reduction: Literal["mean", "sum"] = "mean",
    ) -> None:
        super().__init__(name="huber_reconstruction")
        self.delta = delta
        self.reduction = reduction

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self._validate_shapes(prediction, target)

        loss = F.huber_loss(
            prediction,
            target,
            delta=self.delta,
            reduction="none",
        )

        loss_m = loss.mean()

        return loss.mean(dim=(0, 2, 3)), loss_m


class SmoothL1ReconstructionLoss(Loss):
    """
    Smooth L1 reconstruction loss for a sequence of images.
    """

    def __init__(
        self,
        beta: float = 1.0,
        reduction: Literal["mean", "sum"] = "mean",
    ) -> None:
        super().__init__(name="smooth_l1_reconstruction")
        self.beta = beta
        self.reduction = reduction

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self._validate_shapes(prediction, target)

        loss = F.smooth_l1_loss(
            prediction,
            target,
            beta=self.beta,
            reduction="none",
        )

        loss_m = loss.mean()

        return loss.mean(dim=(0, 2, 3)), loss_m
