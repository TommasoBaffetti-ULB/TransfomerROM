from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
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


class SpectralConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, modes1: int, modes2: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2

        scale = 1.0 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat)
        )
        self.weights2 = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat)
        )

    @staticmethod
    def compl_mul2d(input_ft: Tensor, weights: Tensor) -> Tensor:
        return torch.einsum("bixy,ioxy->boxy", input_ft, weights)

    def forward(self, x: Tensor) -> Tensor:
        bsz = x.shape[0]
        x_ft = torch.fft.rfft2(x)

        out_ft = torch.zeros(
            bsz,
            self.out_channels,
            x.size(-2),
            x.size(-1) // 2 + 1,
            dtype=torch.cfloat,
            device=x.device,
        )
        m1 = min(self.modes1, x_ft.size(-2))
        m2 = min(self.modes2, x_ft.size(-1))

        out_ft[:, :, :m1, :m2] = self.compl_mul2d(
            x_ft[:, :, :m1, :m2], self.weights1[:, :, :m1, :m2]
        )
        out_ft[:, :, -m1:, :m2] = self.compl_mul2d(
            x_ft[:, :, -m1:, :m2], self.weights2[:, :, :m1, :m2]
        )

        return torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))


class SpectralConv3d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, modes1: int, modes2: int, modes3: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2
        self.modes3 = modes3

        scale = 1.0 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, modes1, modes2, modes3, dtype=torch.cfloat)
        )
        self.weights2 = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, modes1, modes2, modes3, dtype=torch.cfloat)
        )
        self.weights3 = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, modes1, modes2, modes3, dtype=torch.cfloat)
        )
        self.weights4 = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, modes1, modes2, modes3, dtype=torch.cfloat)
        )

    @staticmethod
    def compl_mul3d(input_ft: Tensor, weights: Tensor) -> Tensor:
        return torch.einsum("bixyz,ioxyz->boxyz", input_ft, weights)

    def forward(self, x: Tensor) -> Tensor:
        bsz = x.shape[0]
        x_ft = torch.fft.rfftn(x, dim=[-3, -2, -1])

        out_ft = torch.zeros(
            bsz,
            self.out_channels,
            x.size(-3),
            x.size(-2),
            x.size(-1) // 2 + 1,
            dtype=torch.cfloat,
            device=x.device,
        )
        m1 = min(self.modes1, x_ft.size(-3))
        m2 = min(self.modes2, x_ft.size(-2))
        m3 = min(self.modes3, x_ft.size(-1))

        out_ft[:, :, :m1, :m2, :m3] = self.compl_mul3d(
            x_ft[:, :, :m1, :m2, :m3], self.weights1[:, :, :m1, :m2, :m3]
        )
        out_ft[:, :, -m1:, :m2, :m3] = self.compl_mul3d(
            x_ft[:, :, -m1:, :m2, :m3], self.weights2[:, :, :m1, :m2, :m3]
        )
        out_ft[:, :, :m1, -m2:, :m3] = self.compl_mul3d(
            x_ft[:, :, :m1, -m2:, :m3], self.weights3[:, :, :m1, :m2, :m3]
        )
        out_ft[:, :, -m1:, -m2:, :m3] = self.compl_mul3d(
            x_ft[:, :, -m1:, -m2:, :m3], self.weights4[:, :, :m1, :m2, :m3]
        )

        return torch.fft.irfftn(out_ft, s=(x.size(-3), x.size(-2), x.size(-1)))


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
        return F.interpolate(x, size=ref.shape[-2:], mode="bilinear", align_corners=False)

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


class FNO2DForecaster(nn.Module):
    """
    FNO2D baseline forecaster with autoregressive rollout.

    Input:
        x_t: [B, C, H, W] or [C, H, W]
    Output:
        preds: [B, horizon, C, H, W] or [horizon, C, H, W]
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int | None = None,
        modes1: int = 12,
        modes2: int = 12,
        width: int = 32,
        n_layers: int = 4,
        use_residual: bool = True,
    ) -> None:
        super().__init__()
        if n_layers < 1:
            raise ValueError(f"n_layers must be >= 1, got {n_layers}")

        self.in_channels = in_channels
        self.out_channels = out_channels or in_channels
        self.use_residual = use_residual

        self.p = nn.Conv2d(in_channels + 2, width, kernel_size=1)
        self.spectral_layers = nn.ModuleList(
            [SpectralConv2d(width, width, modes1, modes2) for _ in range(n_layers)]
        )
        self.pointwise_layers = nn.ModuleList(
            [nn.Conv2d(width, width, kernel_size=1) for _ in range(n_layers)]
        )
        self.q = nn.Sequential(
            nn.Conv2d(width, width * 2, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(width * 2, self.out_channels, kernel_size=1),
        )

    def _check_shape(self, x: Tensor) -> None:
        if x.ndim != 4:
            raise ValueError(
                f"Expected x with shape [B, C, H, W], got {tuple(x.shape)}"
            )
        if x.shape[1] != self.in_channels:
            raise ValueError(f"Expected {self.in_channels} channels, got {x.shape[1]}")

    @staticmethod
    def _grid_2d(x: Tensor) -> Tensor:
        bsz, _, h, w = x.shape
        gy = torch.linspace(0, 1, steps=h, device=x.device, dtype=x.dtype).view(1, 1, h, 1)
        gx = torch.linspace(0, 1, steps=w, device=x.device, dtype=x.dtype).view(1, 1, 1, w)
        gy = gy.expand(bsz, 1, h, w)
        gx = gx.expand(bsz, 1, h, w)
        return torch.cat([gy, gx], dim=1)

    def _forward_one_step(self, x_t: Tensor) -> Tensor:
        self._check_shape(x_t)

        x = torch.cat([x_t, self._grid_2d(x_t)], dim=1)
        x = self.p(x)

        for spec, pw in zip(self.spectral_layers, self.pointwise_layers):
            x = F.gelu(spec(x) + pw(x))

        delta = self.q(x)
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


class FNO3DForecaster(nn.Module):
    """
    FNO3D baseline forecaster.

    Builds a space-time volume repeating x_t over the requested horizon,
    then predicts the full sequence in one pass.

    Input:
        x_t: [B, C, H, W] or [C, H, W]
    Output:
        preds: [B, horizon, C, H, W] or [horizon, C, H, W]
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int | None = None,
        modes1: int = 8,
        modes2: int = 8,
        modes3: int = 8,
        width: int = 20,
        n_layers: int = 4,
    ) -> None:
        super().__init__()
        if n_layers < 1:
            raise ValueError(f"n_layers must be >= 1, got {n_layers}")

        self.in_channels = in_channels
        self.out_channels = out_channels or in_channels

        self.p = nn.Conv3d(in_channels + 3, width, kernel_size=1)
        self.spectral_layers = nn.ModuleList(
            [SpectralConv3d(width, width, modes1, modes2, modes3) for _ in range(n_layers)]
        )
        self.pointwise_layers = nn.ModuleList(
            [nn.Conv3d(width, width, kernel_size=1) for _ in range(n_layers)]
        )
        self.q = nn.Sequential(
            nn.Conv3d(width, width * 2, kernel_size=1),
            nn.GELU(),
            nn.Conv3d(width * 2, self.out_channels, kernel_size=1),
        )

    def _check_shape(self, x: Tensor) -> None:
        if x.ndim != 4:
            raise ValueError(
                f"Expected x with shape [B, C, H, W], got {tuple(x.shape)}"
            )
        if x.shape[1] != self.in_channels:
            raise ValueError(f"Expected {self.in_channels} channels, got {x.shape[1]}")

    @staticmethod
    def _grid_3d(batch: int, h: int, w: int, t: int, device: torch.device, dtype: torch.dtype) -> Tensor:
        gy = torch.linspace(0, 1, steps=h, device=device, dtype=dtype).view(1, 1, h, 1, 1)
        gx = torch.linspace(0, 1, steps=w, device=device, dtype=dtype).view(1, 1, 1, w, 1)
        gt = torch.linspace(0, 1, steps=t, device=device, dtype=dtype).view(1, 1, 1, 1, t)

        gy = gy.expand(batch, 1, h, w, t)
        gx = gx.expand(batch, 1, h, w, t)
        gt = gt.expand(batch, 1, h, w, t)
        return torch.cat([gy, gx, gt], dim=1)

    def forward(self, x_t: Tensor, horizon: int) -> Tensor:
        if horizon <= 0:
            raise ValueError(f"horizon must be > 0, got {horizon}")

        squeeze_batch = False
        if x_t.ndim == 3:
            x_t = x_t.unsqueeze(0)
            squeeze_batch = True

        self._check_shape(x_t)

        bsz, channels, h, w = x_t.shape

        x = x_t.unsqueeze(-1).repeat(1, 1, 1, 1, horizon)  # [B,C,H,W,T]
        grid = self._grid_3d(bsz, h, w, horizon, x_t.device, x_t.dtype)
        x = torch.cat([x, grid], dim=1)

        x = self.p(x)
        for spec, pw in zip(self.spectral_layers, self.pointwise_layers):
            x = F.gelu(spec(x) + pw(x))

        x = self.q(x)  # [B,C,H,W,T]
        x = x.permute(0, 4, 1, 2, 3).contiguous()  # [B,T,C,H,W]

        if squeeze_batch:
            x = x.squeeze(0)
        return x


# Backward-compatible aliases for common misspelling.
FNO2DForcaster = FNO2DForecaster
FNO3DForcaster = FNO3DForecaster
