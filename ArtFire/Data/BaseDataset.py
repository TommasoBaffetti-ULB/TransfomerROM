from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable, Optional, Tuple

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset


class BaseDataset(Dataset):
    def __init__(self, horizon: int,split: Tuple[float,float,float] ) -> None:
        super().__init__()
        if horizon < 1:
            raise ValueError("horizon must be >= 1")
        self.horizon = horizon
        if len(split)!=3 or np.sum(split)!=1:
            raise ValueError("split must 3 floats summing to 1")
        self.split = split




# ======================================================================
# SPLIT DATASET (this is what DataLoader will use)
# ======================================================================


class SimulatedDataSplit(Dataset):
    def __init__(self, parent: SimulatedData, start: int, end: int) -> None:
        self.parent = parent
        self.start = start
        self.end = end
        self.horizon = parent.horizon

        self.max_start = self.end - self.horizon - 1
        if self.max_start < self.start:
            raise ValueError("Split too small for given horizon")

    def __len__(self) -> int:
        return self.max_start - self.start + 1

    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        t0 = self.start + idx

        data = self.parent.data

        x_t = np.asarray(data[t0], dtype=np.float32)
        y_seq = np.asarray(
            data[t0 + 1 : t0 + 1 + self.horizon], dtype=np.float32
        )

        # build BC sequence
        bc_list = [
            self.parent.build_bc(t)
            for t in range(t0 + 1, t0 + 1 + self.horizon)
        ]
        bc_seq = np.stack(bc_list, axis=0)

        # normalize
        x_t = self.parent.normalize_frame(x_t)
        y_seq = self.parent.normalize_sequence(y_seq)
        bc_seq = self.parent.normalize_sequence(bc_seq)

        return {
            "x_t": torch.from_numpy(x_t),
            "y_seq": torch.from_numpy(y_seq),
            "bc_seq": torch.from_numpy(bc_seq),
        }