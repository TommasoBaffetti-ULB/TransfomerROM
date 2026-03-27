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


