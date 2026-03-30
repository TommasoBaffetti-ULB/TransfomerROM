from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

from ArtFire.Data.BaseDataset import BaseDataset
from ArtFire.Data.CAEDataset import CAEDataset


class ForecastDataset(BaseDataset):
    """
    Dataset for latent forecasting.

    Pipeline:
        1) Encode ALL data with encoder -> z_t
        2) Apply split on latent sequence
        3) (Optional) normalize latents using TRAIN stats

    Returns:
        {
            "z_t":   [D, H_z, W_z],
            "z_seq": [horizon, D, H_z, W_z]
        }
    """

    def __init__(
        self,
        cae_dataset: DataLoader,          # should be FULL dataset (e.g. mode="train" but full coverage)
        encoder: torch.nn.Module,
        split: Tuple[float, float, float],
        mode: str,                        # "train" | "val" | "test"
        horizon: int,
        normalize: bool = True,
        stats: Optional[Tuple[Tensor, Tensor]] = None,
        device: str | torch.device = "cuda",
    ) -> None:
        super().__init__(horizon=horizon, split=split)

        if mode not in {"train", "val", "test"}:
            raise ValueError("mode must be one of: 'train', 'val', 'test'")

        if len(split) != 3 or not torch.isclose(torch.tensor(sum(split)), torch.tensor(1.0)):
            raise ValueError(f"split must sum to 1, got {split}")

        self.mode = mode
        self.normalize = normalize
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        # --------------------------------------------------
        # 1) ENCODE FULL DATASET
        # --------------------------------------------------
        self.encoder = encoder.to(self.device)
        self.encoder.eval()

        self.latents = self._encode_full_dataset(
            cae_dataset,
        )  # [T, D, H_z, W_z]

        self.T = self.latents.shape[0]

        # --------------------------------------------------
        # 2) APPLY NEW SPLIT ON LATENTS
        # --------------------------------------------------
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
            raise ValueError(f"Split '{self.mode}' too small for horizon={horizon}")

        # --------------------------------------------------
        # 3) LATENT NORMALIZATION (TRAIN ONLY)
        # --------------------------------------------------
        if self.normalize:
            if stats is None:
                train_latents = self.latents[:train_end]  # [T_train, Tokens, dim_tokens ]

                self.mean = train_latents.mean(dim=(0, 1))  # [D]
                self.std = train_latents.std(dim=(0, 1))    # [D]
                self.std = torch.clamp(self.std, min=1e-6)
            else:
                self.mean, self.std = stats
                self.std = torch.clamp(self.std, min=1e-6)
        else:
            self.mean, self.std = None, None

    # --------------------------------------------------

    def __len__(self) -> int:
        return self.max_start - self.start + 1

    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Index {idx} out of range")

        t0 = self.start + idx

        z_t = self.latents[t0]
        z_seq = self.latents[t0 + 1 : t0 + 1 + self.horizon]

        if self.normalize:
            z_t = self._normalize_frame(z_t)
            z_seq = self._normalize_sequence(z_seq)

        return {
            "z_t": z_t,
            "z_seq": z_seq,
        }

    # --------------------------------------------------

    def _normalize_frame(self, z: Tensor) -> Tensor:
        return (z - self.mean[None, :]) / self.std[None, :]

    def _normalize_sequence(self, z: Tensor) -> Tensor:
        return (z - self.mean[None, None, :]) / self.std[None, None, :]

    def get_stats(self) -> Optional[Tuple[Tensor, Tensor]]:
        if not self.normalize:
            return None
        return self.mean, self.std

    # --------------------------------------------------

    @torch.no_grad()
    def _encode_full_dataset(
        self,
        cae_dataset: DataLoader
    ) -> Tensor:


        latents = []

        for batch in tqdm(cae_dataset, desc="Encoding full dataset"):
            x = batch["x"].to(self.device)
            z = self.encoder(x)  # [B, D, H_z, W_z]
            latents.append(z.cpu())

        return torch.cat(latents, dim=0)