from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class _DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)


class _ResBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
        )
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        return self.act(x + self.block(x))


class UNetForecaster(nn.Module):
    """
    Full-space U-Net baseline forecaster.

    Input:
        x_t: [B, C, H, W] or [C, H, W]

    Output:
        preds: [B, horizon, C, H, W] or [horizon, C, H, W]
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int | None = None,
        base_channels: int = 32,
        use_residual: bool = True,
    ) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels or in_channels
        self.use_residual = use_residual

        c1, c2, c3, c4 = (
            base_channels,
            base_channels * 2,
            base_channels * 4,
            base_channels * 8,
        )

        self.enc1 = _DoubleConv(self.in_channels, c1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.enc2 = _DoubleConv(c1, c2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.enc3 = _DoubleConv(c2, c3)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = _DoubleConv(c3, c4)

        self.up3 = nn.ConvTranspose2d(c4, c3, kernel_size=2, stride=2)
        self.dec3 = _DoubleConv(c3 + c3, c3)
        self.up2 = nn.ConvTranspose2d(c3, c2, kernel_size=2, stride=2)
        self.dec2 = _DoubleConv(c2 + c2, c2)
        self.up1 = nn.ConvTranspose2d(c2, c1, kernel_size=2, stride=2)
        self.dec1 = _DoubleConv(c1 + c1, c1)

        self.head = nn.Conv2d(c1, self.out_channels, kernel_size=1)

    def _check_shape(self, x: Tensor) -> None:
        if x.ndim != 4:
            raise ValueError(
                f"Expected x with shape [B, C, H, W], got {tuple(x.shape)}"
            )
        if x.shape[1] != self.in_channels:
            raise ValueError(f"Expected {self.in_channels} channels, got {x.shape[1]}")

    @staticmethod
    def _match_spatial(x: Tensor, ref: Tensor) -> Tensor:
        if x.shape[-2:] == ref.shape[-2:]:
            return x
        return nn.functional.interpolate(
            x, size=ref.shape[-2:], mode="bilinear", align_corners=False
        )

    def _forward_one_step(self, x_t: Tensor) -> Tensor:
        self._check_shape(x_t)

        s1 = self.enc1(x_t)
        s2 = self.enc2(self.pool1(s1))
        s3 = self.enc3(self.pool2(s2))

        x = self.bottleneck(self.pool3(s3))

        x = self._match_spatial(self.up3(x), s3)
        x = self.dec3(torch.cat([x, s3], dim=1))

        x = self._match_spatial(self.up2(x), s2)
        x = self.dec2(torch.cat([x, s2], dim=1))

        x = self._match_spatial(self.up1(x), s1)
        x = self.dec1(torch.cat([x, s1], dim=1))

        delta = self.head(x)
        if self.use_residual and self.out_channels == self.in_channels:
            return x_t + delta
        return delta

    def forward(self, x_t: Tensor, horizon: int) -> Tensor:
        if horizon <= 0:
            raise ValueError(f"horizon must be > 0, got {horizon}")

        squeeze_batch = False
        if x_t.ndim == 3:
            x_t = x_t.unsqueeze(0)
            squeeze_batch = True

        self._check_shape(x_t)

        preds = []
        x = x_t
        for _ in range(horizon):
            x = self._forward_one_step(x)
            preds.append(x)

        out = torch.stack(preds, dim=1)
        if squeeze_batch:
            out = out.squeeze(0)
        return out


class ResNetForecaster(nn.Module):
    """
    Full-space ResNet baseline forecaster.

    Input:
        x_t: [B, C, H, W] or [C, H, W]

    Output:
        preds: [B, horizon, C, H, W] or [horizon, C, H, W]
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int | None = None,
        hidden_channels: int = 64,
        n_blocks: int = 6,
        use_residual: bool = True,
    ) -> None:
        super().__init__()
        if n_blocks < 1:
            raise ValueError(f"n_blocks must be >= 1, got {n_blocks}")

        self.in_channels = in_channels
        self.out_channels = out_channels or in_channels
        self.use_residual = use_residual

        self.stem = nn.Sequential(
            nn.Conv2d(self.in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
        )
        self.blocks = nn.ModuleList([_ResBlock(hidden_channels) for _ in range(n_blocks)])
        self.head = nn.Conv2d(hidden_channels, self.out_channels, kernel_size=1)

    def _check_shape(self, x: Tensor) -> None:
        if x.ndim != 4:
            raise ValueError(
                f"Expected x with shape [B, C, H, W], got {tuple(x.shape)}"
            )
        if x.shape[1] != self.in_channels:
            raise ValueError(f"Expected {self.in_channels} channels, got {x.shape[1]}")

    def _forward_one_step(self, x_t: Tensor) -> Tensor:
        self._check_shape(x_t)

        x = self.stem(x_t)
        for block in self.blocks:
            x = block(x)

        delta = self.head(x)
        if self.use_residual and self.out_channels == self.in_channels:
            return x_t + delta
        return delta

    def forward(self, x_t: Tensor, horizon: int) -> Tensor:
        if horizon <= 0:
            raise ValueError(f"horizon must be > 0, got {horizon}")

        squeeze_batch = False
        if x_t.ndim == 3:
            x_t = x_t.unsqueeze(0)
            squeeze_batch = True

        self._check_shape(x_t)

        preds = []
        x = x_t
        for _ in range(horizon):
            x = self._forward_one_step(x)
            preds.append(x)

        out = torch.stack(preds, dim=1)
        if squeeze_batch:
            out = out.squeeze(0)
        return out
