from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
from torch import Tensor

from ArtFire.Data.BaseDataset import BaseDataset


class CAEDataset(BaseDataset):
    """
    Dataset for CAE training on data stored in a .npy file of shape [T, C, H, W].

    Each sample returns:
        {
            "x": [C, H, W],
            "y": [C, H, W]
        }

    where x == y, since the CAE reconstructs the same input frame.
    """

    def __init__(
        self,
        npy_path: str | Path,
        split: Tuple[float, float, float],  # (train, val, test)
        mode: str,  # "train" | "val" | "test"
        normalize: bool = True,
        stats: Optional[Tuple[Tensor, Tensor]] = None,
    ) -> None:
        super().__init__(horizon=1, split=split)

        if mode not in {"train", "val", "test"}:
            raise ValueError("mode must be one of: 'train', 'val', 'test'")

        if len(split) != 3:
            raise ValueError("split must be a tuple (train, val, test)")

        if not np.isclose(sum(split), 1.0):
            raise ValueError(f"split must sum to 1, got {split}")

        self.npy_path = Path(npy_path)
        if not self.npy_path.exists():
            raise FileNotFoundError(f"{self.npy_path} not found")

        self.mode = mode
        self.normalize = normalize

        data_np = np.load(self.npy_path, mmap_mode="r")  # [T, C, H, W]
        if data_np.ndim != 4:
            raise ValueError(f"Expected [T, C, H, W], got {data_np.shape}")

        self.data = torch.from_numpy(
            np.asarray(data_np, dtype=np.float32)
        )  # CPU tensor
        self.T, self.C, self.H, self.W = self.data.shape

        train_frac, val_frac, test_frac = split
        train_end = int(self.T * train_frac)
        val_end = int(self.T * (train_frac + val_frac))

        self.splits = {
            "train": (0, train_end),
            "val": (train_end, val_end),
            "test": (val_end, self.T),
        }

        self.start, self.end = self.splits[self.mode]

        if self.end <= self.start:
            raise ValueError(f"Empty split '{self.mode}'")

        if self.normalize:
            if stats is None:
                train_data = self.data[:train_end]  # [T_train, C, H, W]
                self.mean = train_data.mean(dim=(0, 2, 3))  # [C]
                self.std = train_data.std(dim=(0, 2, 3))  # [C]
                self.std = torch.clamp(self.std, min=1e-6)
            else:
                self.mean, self.std = stats
                self.std = torch.clamp(self.std, min=1e-6)
        else:
            self.mean = None
            self.std = None

    def __len__(self) -> int:
        return self.end - self.start

    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Index {idx} out of range for split '{self.mode}'")

        t = self.start + idx
        x = self.data[t]  # [C, H, W]

        if self.normalize:
            x = self._normalize_frame(x)

        return {
            "x": x,
            "y": x.clone(),
        }

    def _normalize_frame(self, x: Tensor) -> Tensor:
        return (x - self.mean[:, None, None]) / self.std[:, None, None]

    def denormalize_frame(self, x: Tensor) -> Tensor:
        if not self.normalize:
            return x
        return x * self.std[:, None, None] + self.mean[:, None, None]

    def get_stats(self) -> Optional[Tuple[Tensor, Tensor]]:
        if not self.normalize:
            return None
        return self.mean, self.std
