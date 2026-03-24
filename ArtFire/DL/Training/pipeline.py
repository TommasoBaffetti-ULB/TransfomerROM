from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

from ArtFire.DL.Data.datasets import LatentSequenceDataset, SnapshotWindowDataset, TrainingModes
from ArtFire.DL.Models.CAE import ConvAutoEncoder
from ArtFire.DL.Models.TransformerPredictor import LatentTransformerPredictor
from ArtFire.DL.Optimization.optimizers import build_optimizer


@dataclass
class PipelineConfig:
    device: str = "cpu"
    window: int = 50
    batch_size: int = 8
    cae_lr: float = 1e-3
    tr_lr: float = 1e-4


class ROMTrainingPipeline:
    """Modular pipeline for CAE + latent Transformer experiments."""

    def __init__(
        self,
        cae: ConvAutoEncoder,
        transformer: LatentTransformerPredictor,
        modes: TrainingModes,
        config: PipelineConfig,
    ) -> None:
        self.cae = cae.to(config.device)
        self.transformer = transformer.to(config.device)
        self.modes = modes
        self.config = config

        self.cae_optimizer = build_optimizer(self.cae.parameters(), {"name": "adamw", "lr": config.cae_lr})
        joint_params = list(self.transformer.parameters())
        if modes.joint_cae_transformer:
            joint_params += list(self.cae.parameters())
        self.transformer_optimizer = build_optimizer(joint_params, {"name": "adamw", "lr": config.tr_lr})

    def _physics_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Placeholder hook for future PINN losses.
        return F.mse_loss(pred, target)

    def train_cae_epoch(self, data: torch.Tensor) -> float:
        dataset = SnapshotWindowDataset(data, window=self.config.window)
        loader = DataLoader(dataset, batch_size=self.config.batch_size, shuffle=True)

        self.cae.train()
        total = 0.0
        n = 0
        for windows in loader:
            windows = windows.to(self.config.device)
            flat = windows.reshape(-1, *windows.shape[2:])
            recon, _ = self.cae(flat)
            loss = F.mse_loss(recon, flat)

            self.cae_optimizer.zero_grad()
            loss.backward()
            self.cae_optimizer.step()

            total += loss.item()
            n += 1
        return total / max(n, 1)

    @torch.no_grad()
    def encode_series(self, data: torch.Tensor) -> torch.Tensor:
        self.cae.eval()
        data = data.to(self.config.device)
        latents = self.cae.encode(data)
        return latents.detach().cpu()

    def train_transformer_epoch(self, latents: torch.Tensor) -> float:
        ar_context = 3 if self.modes.autoregressive else 1
        dataset = LatentSequenceDataset(
            latents=latents,
            window=self.config.window,
            autoregressive_context=ar_context,
        )
        loader = DataLoader(dataset, batch_size=self.config.batch_size, shuffle=True)

        self.transformer.train()
        if self.modes.joint_cae_transformer:
            self.cae.train()

        total = 0.0
        n = 0
        for x, y in loader:
            x = x.to(self.config.device)
            y = y.to(self.config.device)

            pred = self.transformer(x)
            pred = pred[:, -y.shape[1] :, :]

            if self.modes.recursive_training:
                recursive_steps = min(self.modes.recursive_steps, 3)
                rollout = [pred]
                last = pred
                for _ in range(recursive_steps - 1):
                    last = self.transformer(last)
                    rollout.append(last)
                pred = rollout[0]

            loss = F.mse_loss(pred, y)
            if self.modes.physics_informed_loss:
                loss = loss + 0.1 * self._physics_loss(pred, y)

            self.transformer_optimizer.zero_grad()
            loss.backward()
            self.transformer_optimizer.step()

            total += loss.item()
            n += 1
        return total / max(n, 1)

    def fit(self, train_data: torch.Tensor, epochs_cae: int = 5, epochs_transformer: int = 5) -> Dict[str, float]:
        history: Dict[str, float] = {}

        if not self.modes.joint_cae_transformer:
            for _ in range(epochs_cae):
                history["cae_loss"] = self.train_cae_epoch(train_data)

        latents = self.encode_series(train_data)
        for _ in range(epochs_transformer):
            history["transformer_loss"] = self.train_transformer_epoch(latents)

        return history
