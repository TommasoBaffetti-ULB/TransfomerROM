from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional, Tuple

import numpy as np
import torch
from torch import Tensor

from ArtFire.Data.BaseDataset import BaseDataset


class SimulatedDataset(BaseDataset):
    """
    Dataset for [T, C, H, W] data stored in a .npy file.

    Each sample returns:
        x_t   : [C, H, W]
        y_seq : [horizon, C, H, W]
        bc_seq: [horizon, C, H, W]
    """

    def __init__(
        self,
        npy_path: str | Path,
        split: Tuple[float, float, float],  # (train, val, test)
        mode: str,  # "train" | "val" | "test"
        horizon: int,
        normalize: bool = True,
        stats: Optional[Tuple[Tensor, Tensor]] = None,
        bc_function: Optional[Callable[[int, int, int, int], np.ndarray]] = None,
    ) -> None:
        super().__init__(horizon=horizon, split=split)

        if mode not in {"train", "val", "test"}:
            raise ValueError("mode must be one of: 'train', 'val', 'test'")

        self.npy_path = Path(npy_path)
        if not self.npy_path.exists():
            raise FileNotFoundError(f"{self.npy_path} not found")

        self.mode = mode
        self.normalize = normalize
        self.bc_function = bc_function or self.default_bc

        # Load numpy and convert once to torch
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

        self.max_start = self.end - self.horizon - 1
        if self.max_start < self.start:
            raise ValueError(
                f"Split '{self.mode}' too small for horizon={self.horizon}. "
                f"Range: [{self.start}, {self.end})"
            )

        # Compute normalization only on train split
        self.stats = stats
        if self.normalize:
            if self.stats is None:
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
        return self.max_start - self.start + 1

    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Index {idx} out of range for split '{self.mode}'")

        t0 = self.start + idx

        x_t = self.data[t0]  # [C, H, W]
        y_seq = self.data[t0 + 1 : t0 + 1 + self.horizon]  # [horizon, C, H, W]

        bc_seq = self._build_bc_sequence(t0)  # [horizon, C, H, W]

        if self.normalize:
            x_t = self._normalize_frame(x_t)
            y_seq = self._normalize_sequence(y_seq)
            bc_seq = self._normalize_sequence(bc_seq)

        return {
            "x_t": x_t,
            "y_seq": y_seq,
            "bc_seq": bc_seq,
        }

    def _normalize_frame(self, x: Tensor) -> Tensor:
        return (x - self.mean[:, None, None]) / self.std[:, None, None]

    def _normalize_sequence(self, x: Tensor) -> Tensor:
        return (x - self.mean[None, :, None, None]) / self.std[None, :, None, None]

    def get_stats(self) -> Optional[Tuple[Tensor, Tensor]]:
        if not self.normalize:
            return None
        return self.mean, self.std

    def _build_bc_sequence(self, t0: int) -> Tensor:
        bc_list = []
        for t in range(t0 + 1, t0 + 1 + self.horizon):
            bc = self.bc_function(t, self.C, self.H, self.W)
            bc_list.append(bc)
        return torch.stack(bc_list, dim=0)

    @staticmethod
    def default_bc(t: int, C: int, H: int, W: int) -> Tensor:
        bc = torch.zeros(C, H, W)

        ch = torch.arange(C, dtype=torch.float32)

        vals1 = torch.sin(0.02 * t + ch)
        vals2 = torch.cos(0.02 * t + ch)

        bc[:, 0, :] = vals1[:, None]
        bc[:, -1, :] = vals2[:, None]
        bc[:, :, 0] = vals1[:, None]
        bc[:, :, -1] = vals2[:, None]

        return bc
