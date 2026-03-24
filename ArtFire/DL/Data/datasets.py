from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch.utils.data import Dataset


class SnapshotWindowDataset(Dataset):
    """Returns consecutive windows [Nw, F, Z, X] for CAE or joint training."""

    def __init__(self, data: torch.Tensor, window: int = 50) -> None:
        if data.ndim != 4:
            raise ValueError("data must be [T,F,Z,X]")
        self.data = data
        self.window = window

    def __len__(self) -> int:
        return self.data.shape[0] - self.window

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.data[idx : idx + self.window]


class LatentSequenceDataset(Dataset):
    """Produces latent sequence pairs for next-step prediction.

    x: [Nw-1, D], y: [Nw-1, D] where y[t] = z[t+1]
    """

    def __init__(
        self,
        latents: torch.Tensor,
        window: int = 50,
        autoregressive_context: int = 1,
    ) -> None:
        if latents.ndim != 2:
            raise ValueError("latents must be [T,D]")
        self.latents = latents
        self.window = window
        self.autoregressive_context = autoregressive_context

    def __len__(self) -> int:
        return self.latents.shape[0] - self.window

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        seq = self.latents[idx : idx + self.window]
        x = seq[:-1]
        y = seq[1:]

        # Padding at sequence start supports later AR modes (e.g. [z_t, z_t-1, z_t-2]).
        if self.autoregressive_context > 1:
            pad = x[:1].repeat(self.autoregressive_context - 1, 1)
            x = torch.cat([pad, x], dim=0)
        return x, y


@dataclass
class TrainingModes:
    """Switchboard for future experiments.

    For now they are placeholders wired into pipeline control flow.
    """

    joint_cae_transformer: bool = False
    with_boundary_conditions: bool = False
    recursive_training: bool = False
    autoregressive: bool = False
    physics_informed_loss: bool = False

    # Future behavior limits.
    recursive_steps: int = 3
