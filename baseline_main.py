from __future__ import annotations

import gc
import os
from pathlib import Path

import torch
import torch.nn as nn
from pytorch_scheduler.scheduler.polynomial import PolynomialScheduler
from torch.utils.data import DataLoader

from ArtFire.Data.SimulatedDataset import SimulatedDataset
from ArtFire.DL.Models.Baselines import (
    FNO2DForecaster,
    FNO3DForecaster,
    ResNetForecaster,
    UNetForecaster,
)
from ArtFire.DL.Optimization.optimizers import build_optimizer, build_parameter_groups
from ArtFire.DL.Optimization.warmup import WarmupScheduler
from ArtFire.DL.Training.BaselineTrainer import BaselineTrainer
from ArtFire.utils.config import load_data_config, load_model_config, load_train_config
from ArtFire.utils.save import save_json
from ArtFire.utils.seed import seed_worker, set_seed


SEED = 42
set_seed(SEED)
g = torch.Generator()
g.manual_seed(SEED)


def build_baseline(model_name: str, model_cfg: dict, in_channels: int) -> torch.nn.Module:
    if model_name == "unet":
        cfg = model_cfg["UNet"]
        return UNetForecaster(
            in_channels=in_channels,
            out_channels=cfg.get("out_channels", in_channels),
            base_channels=cfg.get("base_channels", 32),
            use_residual=cfg.get("use_residual", True),
        )

    if model_name == "resnet":
        cfg = model_cfg["ResNet"]
        return ResNetForecaster(
            in_channels=in_channels,
            out_channels=cfg.get("out_channels", in_channels),
            hidden_channels=cfg.get("hidden_channels", 64),
            n_blocks=cfg.get("n_blocks", 6),
            use_residual=cfg.get("use_residual", True),
        )

    if model_name == "fno2d":
        cfg = model_cfg["FNO2D"]
        return FNO2DForecaster(
            in_channels=in_channels,
            out_channels=cfg.get("out_channels", in_channels),
            modes1=cfg.get("modes1", 12),
            modes2=cfg.get("modes2", 12),
            width=cfg.get("width", 32),
            n_layers=cfg.get("n_layers", 4),
            use_residual=cfg.get("use_residual", True),
        )

    if model_name == "fno3d":
        cfg = model_cfg["FNO3D"]
        return FNO3DForecaster(
            in_channels=in_channels,
            out_channels=cfg.get("out_channels", in_channels),
            modes1=cfg.get("modes1", 8),
            modes2=cfg.get("modes2", 8),
            modes3=cfg.get("modes3", 8),
            width=cfg.get("width", 20),
            n_layers=cfg.get("n_layers", 4),
        )

    raise ValueError("Unsupported baseline model. Use one of: unet, resnet, fno2d, fno3d")


def main(verbose: bool = False) -> None:
    gc.collect()
    torch.cuda.empty_cache()

    data_config = load_data_config()
    model_config = load_model_config()
    train_config = load_train_config()

    baseline_data_cfg = data_config["Baseline"]
    baseline_model_cfg = model_config["Baseline"]
    baseline_train_cfg = train_config["Baseline"]

    model_name = os.getenv("BASELINE_MODEL", baseline_model_cfg.get("model_name", "unet")).lower()

    train_dataset = SimulatedDataset(
        npy_path=baseline_data_cfg["data_path"],
        split=tuple(baseline_data_cfg["split"]),
        mode="train",
        horizon=baseline_data_cfg["horizon"],
        normalize=baseline_data_cfg["normalize"],
    )

    val_dataset = SimulatedDataset(
        npy_path=baseline_data_cfg["data_path"],
        split=tuple(baseline_data_cfg["split"]),
        mode="val",
        horizon=baseline_data_cfg["horizon"],
        normalize=baseline_data_cfg["normalize"],
        stats=train_dataset.get_stats(),
    )

    test_dataset = SimulatedDataset(
        npy_path=baseline_data_cfg["data_path"],
        split=tuple(baseline_data_cfg["split"]),
        mode="test",
        horizon=baseline_data_cfg["horizon"],
        normalize=baseline_data_cfg["normalize"],
        stats=train_dataset.get_stats(),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=baseline_data_cfg["train_batch_size"],
        shuffle=True,
        num_workers=baseline_data_cfg["loader"]["loaders_num_workers"],
        pin_memory=baseline_data_cfg["loader"]["pin_memory"],
        persistent_workers=baseline_data_cfg["loader"]["persistent_workers"],
        worker_init_fn=seed_worker,
        generator=g,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=baseline_data_cfg["val_batch_size"],
        shuffle=True,
        num_workers=baseline_data_cfg["loader"]["loaders_num_workers"],
        pin_memory=baseline_data_cfg["loader"]["pin_memory"],
        persistent_workers=baseline_data_cfg["loader"]["persistent_workers"],
        worker_init_fn=seed_worker,
        generator=g,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=baseline_data_cfg["test_batch_size"],
        shuffle=True,
        num_workers=baseline_data_cfg["loader"]["loaders_num_workers"],
        pin_memory=baseline_data_cfg["loader"]["pin_memory"],
        persistent_workers=baseline_data_cfg["loader"]["persistent_workers"],
        worker_init_fn=seed_worker,
        generator=g,
    )

    in_channels = train_dataset[0]["x_t"].shape[0]
    model = build_baseline(model_name, baseline_model_cfg, in_channels).to(
        baseline_train_cfg["device"]
    )

    if verbose:
        print(model)

    parameter_groups = build_parameter_groups(
        model,
        lr=baseline_train_cfg["lr"],
        weight_decay=baseline_train_cfg["weight_decay"],
    )
    optimizer = build_optimizer(
        params=parameter_groups[1],
        config={
            "name": baseline_train_cfg["optimizer"],
            "use_lookahead": baseline_train_cfg["use_lookahead"],
        },
    )

    scheduler = PolynomialScheduler(
        optimizer=optimizer,
        total_steps=baseline_train_cfg["n_epochs"],
        power=baseline_train_cfg["scheduler"]["power"],
        min_lr=baseline_train_cfg["scheduler"]["min_lr"],
    )

    warmup_scheduler = WarmupScheduler(
        optimizer,
        scheduler,
        min_lr=baseline_train_cfg["scheduler"]["min_lr_warmup"],
        warmup_steps=baseline_train_cfg["scheduler"]["warmup_steps"],
        warmup_type=baseline_train_cfg["scheduler"]["warmup_type"],
    )

    save_root = Path(baseline_train_cfg["saving_folder"]) / model_name
    save_root.mkdir(parents=True, exist_ok=True)

    trainer = BaselineTrainer(
        model=model,
        loaders=[train_loader, val_loader, test_loader],
        optimizer=optimizer,
        criterion=nn.MSELoss(),
        scheduler=warmup_scheduler,
        device=baseline_train_cfg["device"],
        gradient_clip=baseline_train_cfg["gradient_clip"],
        best_model_path=save_root / "best_baseline.pt",
    )

    train_results = trainer.learn(num_epochs=baseline_train_cfg["n_epochs"])
    test_results = trainer.test()

    save_json(train_results, save_root / "train_results.json")
    save_json(test_results, save_root / "test_result.json")

    print(f"Completed baseline training for model: {model_name}")


if __name__ == "__main__":
    main()
