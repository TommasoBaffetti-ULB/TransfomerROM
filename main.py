from __future__ import annotations

import torch

from ArtFire.DL.Data.preprocessing import FieldScaler, split_time_series
from ArtFire.DL.Data.datasets import TrainingModes
from ArtFire.DL.Models import ConvAutoEncoder, LatentTransformerPredictor
from ArtFire.DL.Training import PipelineConfig, ROMTrainingPipeline


def build_pipeline(
    nf: int = 13,
    nz: int = 256,
    nx: int = 64,
    latent_dim: int = 256,
    window: int = 50,
) -> ROMTrainingPipeline:
    cae = ConvAutoEncoder(
        in_channels=nf,
        channel_progression=(32, 64, 128, 256),
        latent_dim=latent_dim,
        input_hw=(nz, nx),
    )
    transformer = LatentTransformerPredictor(latent_dim=latent_dim)

    modes = TrainingModes(
        joint_cae_transformer=False,
        with_boundary_conditions=False,
        recursive_training=False,
        autoregressive=False,
        physics_informed_loss=False,
    )
    config = PipelineConfig(window=window)
    return ROMTrainingPipeline(cae=cae, transformer=transformer, modes=modes, config=config)


def run_example() -> None:
    # Mock tensor with expected geometry: [Nt, Nf, Nz, Nx] = [2001, 13, 256, 64]
    data = torch.randn(2001, 13, 256, 64)

    train, val, test = split_time_series(data)
    scaler = FieldScaler.fit(train)
    train = scaler.transform(train)
    _ = scaler.transform(val), scaler.transform(test)

    pipeline = build_pipeline()
    history = pipeline.fit(train_data=train, epochs_cae=1, epochs_transformer=1)
    print("Training completed", history)


if __name__ == "__main__":
    run_example()
