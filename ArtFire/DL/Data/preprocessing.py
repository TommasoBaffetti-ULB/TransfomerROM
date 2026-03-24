from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch


@dataclass
class FieldScaler:
    """Per-feature standardization for simulation tensors [T, F, Z, X]."""

    mean: torch.Tensor
    std: torch.Tensor
    eps: float = 1e-6

    @classmethod
    def fit(cls, data: torch.Tensor, eps: float = 1e-6) -> "FieldScaler":
        if data.ndim != 4:
            raise ValueError(f"Expected [T,F,Z,X], got shape {tuple(data.shape)}")
        mean = data.mean(dim=(0, 2, 3), keepdim=True)
        std = data.std(dim=(0, 2, 3), keepdim=True).clamp_min(eps)
        return cls(mean=mean, std=std, eps=eps)

    def transform(self, data: torch.Tensor) -> torch.Tensor:
        return (data - self.mean) / self.std

    def inverse_transform(self, data: torch.Tensor) -> torch.Tensor:
        return data * self.std + self.mean


def split_time_series(
    data: torch.Tensor,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Temporal split preserving causality."""
    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio must be in (0, 1)")
    if not 0 <= val_ratio < 1:
        raise ValueError("val_ratio must be in [0, 1)")
    if train_ratio + val_ratio >= 1:
        raise ValueError("train_ratio + val_ratio must be < 1")

    nt = data.shape[0]
    train_end = int(nt * train_ratio)
    val_end = train_end + int(nt * val_ratio)

    train = data[:train_end]
    val = data[train_end:val_end]
    test = data[val_end:]
    return train, val, test
