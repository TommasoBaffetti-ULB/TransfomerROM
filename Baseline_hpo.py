"""
Baseline Hyperparameter Optimization (Optuna).

Tunes one baseline architecture at a time: unet | resnet | fno2d | fno3d.

Examples
--------
python Baseline_hpo.py --baseline-model unet --n-trials 30 --hpo-epochs 10
python Baseline_hpo.py --baseline-model fno2d --n-trials 50 --hpo-epochs 15 --storage sqlite:///hpo.db
"""
from __future__ import annotations

import argparse
import gc
from pathlib import Path
from typing import Optional

import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
import torch
import torch.nn as nn
from pytorch_scheduler.scheduler.polynomial import PolynomialScheduler
from torch.utils.data import DataLoader
import yaml

from ArtFire.Data.SimulatedDataset import SimulatedDataset
from ArtFire.DL.Models.Baselines import (
    FNO2DForecaster,
    FNO3DForecaster,
    ResNetForecaster,
    UNetForecaster,
)
from ArtFire.DL.Optimization.optimizers import build_optimizer, build_parameter_groups
from ArtFire.DL.Optimization.warmup import WarmupScheduler
from ArtFire.utils.config import load_data_config
from ArtFire.utils.seed import seed_worker, set_seed

SEED = 42
set_seed(SEED)
_g = torch.Generator()
_g.manual_seed(SEED)

_N_TRIALS_DEFAULT = 50
_HPO_EPOCHS_DEFAULT = 20
_N_STARTUP_TRIALS = 10
_PRUNER_WARMUP = 5


VALID_MODELS = {"unet", "resnet", "fno2d", "fno3d"}


def _suggest_hparams(trial: optuna.Trial, model_name: str) -> dict:
    hp = {
        "lr": trial.suggest_float("lr", 1e-5, 5e-3, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
        "optimizer": trial.suggest_categorical("optimizer", ["adamw", "adam"]),
        "use_lookahead": trial.suggest_categorical("use_lookahead", [True, False]),
        "gradient_clip": trial.suggest_categorical("gradient_clip", [-1.0, 1.0, 3.0, 5.0]),
        "batch_size": trial.suggest_categorical("batch_size", [8, 16, 32]),
        "scheduler_power": trial.suggest_int("scheduler_power", 1, 3),
        "warmup_steps": trial.suggest_int("warmup_steps", 0, 20),
        "min_lr_ratio": trial.suggest_float("min_lr_ratio", 0.01, 0.5),
    }

    if model_name == "unet":
        hp.update(
            {
                "base_channels": trial.suggest_categorical("base_channels", [16, 32, 48, 64]),
                "use_residual": trial.suggest_categorical("use_residual", [True, False]),
            }
        )
    elif model_name == "resnet":
        hp.update(
            {
                "hidden_channels": trial.suggest_categorical("hidden_channels", [32, 64, 96, 128]),
                "n_blocks": trial.suggest_int("n_blocks", 2, 10),
                "use_residual": trial.suggest_categorical("use_residual", [True, False]),
            }
        )
    elif model_name == "fno2d":
        hp.update(
            {
                "modes1": trial.suggest_categorical("modes1", [8, 12, 16, 20]),
                "modes2": trial.suggest_categorical("modes2", [8, 12, 16, 20]),
                "width": trial.suggest_categorical("width", [16, 24, 32, 48, 64]),
                "n_layers": trial.suggest_int("n_layers", 2, 6),
                "use_residual": trial.suggest_categorical("use_residual", [True, False]),
            }
        )
    elif model_name == "fno3d":
        hp.update(
            {
                "modes1": trial.suggest_categorical("modes1", [4, 6, 8, 10]),
                "modes2": trial.suggest_categorical("modes2", [4, 6, 8, 10]),
                "modes3": trial.suggest_categorical("modes3", [4, 6, 8, 10]),
                "width": trial.suggest_categorical("width", [12, 16, 20, 24, 32]),
                "n_layers": trial.suggest_int("n_layers", 2, 6),
            }
        )

    hp["min_lr"] = hp["lr"] * hp["min_lr_ratio"]
    return hp


def _build_baseline(model_name: str, hp: dict, in_channels: int) -> torch.nn.Module:
    if model_name == "unet":
        return UNetForecaster(
            in_channels=in_channels,
            out_channels=in_channels,
            base_channels=hp["base_channels"],
            use_residual=hp["use_residual"],
        )
    if model_name == "resnet":
        return ResNetForecaster(
            in_channels=in_channels,
            out_channels=in_channels,
            hidden_channels=hp["hidden_channels"],
            n_blocks=hp["n_blocks"],
            use_residual=hp["use_residual"],
        )
    if model_name == "fno2d":
        return FNO2DForecaster(
            in_channels=in_channels,
            out_channels=in_channels,
            modes1=hp["modes1"],
            modes2=hp["modes2"],
            width=hp["width"],
            n_layers=hp["n_layers"],
            use_residual=hp["use_residual"],
        )
    if model_name == "fno3d":
        return FNO3DForecaster(
            in_channels=in_channels,
            out_channels=in_channels,
            modes1=hp["modes1"],
            modes2=hp["modes2"],
            modes3=hp["modes3"],
            width=hp["width"],
            n_layers=hp["n_layers"],
        )
    raise ValueError(f"Unknown baseline model: {model_name}")


def objective(
    trial: optuna.Trial,
    train_dataset: SimulatedDataset,
    val_dataset: SimulatedDataset,
    input_channels: int,
    device: torch.device,
    num_workers: int,
    hpo_epochs: int,
    model_name: str,
) -> float:
    hp = _suggest_hparams(trial, model_name)

    try:
        model = _build_baseline(model_name, hp, input_channels).to(device)
    except Exception as exc:
        print(f"Trial {trial.number} pruned for invalid model config: {exc}")
        raise optuna.exceptions.TrialPruned()

    pin = device.type != "cpu"
    train_loader = DataLoader(
        train_dataset,
        batch_size=hp["batch_size"],
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin,
        persistent_workers=False,
        worker_init_fn=seed_worker,
        generator=_g,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=hp["batch_size"],
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin,
        persistent_workers=False,
    )

    parameter_groups = build_parameter_groups(
        model, lr=hp["lr"], weight_decay=hp["weight_decay"]
    )
    optimizer = build_optimizer(
        params=parameter_groups[1],
        config={"name": hp["optimizer"], "use_lookahead": hp["use_lookahead"]},
    )

    base_scheduler = PolynomialScheduler(
        optimizer=optimizer,
        total_steps=hpo_epochs,
        power=hp["scheduler_power"],
        min_lr=hp["min_lr"],
    )
    scheduler = WarmupScheduler(
        optimizer,
        base_scheduler,
        min_lr=hp["min_lr"],
        warmup_steps=hp["warmup_steps"],
        warmup_type="linear",
    )

    criterion = nn.MSELoss()
    best_val_loss = float("inf")

    try:
        for epoch in range(hpo_epochs):
            model.train()
            for batch in train_loader:
                x_t = batch["x_t"].to(device)
                x_seq = batch["x_seq"].to(device)

                optimizer.zero_grad()
                pred_seq = model(x_t, horizon=x_seq.shape[1])
                loss = criterion(pred_seq, x_seq)
                loss.backward()
                if hp["gradient_clip"] > 0:
                    torch.nn.utils.clip_grad_value_(model.parameters(), hp["gradient_clip"])
                optimizer.step()
            scheduler.step()

            model.eval()
            running = 0.0
            total = 0
            with torch.no_grad():
                for batch in val_loader:
                    x_t = batch["x_t"].to(device)
                    x_seq = batch["x_seq"].to(device)
                    pred_seq = model(x_t, horizon=x_seq.shape[1])
                    loss = criterion(pred_seq, x_seq)
                    running += loss.item() * x_t.size(0)
                    total += x_t.size(0)

            val_loss = running / max(total, 1)
            if val_loss < best_val_loss:
                best_val_loss = val_loss

            trial.report(val_loss, epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

    finally:
        del model, optimizer, scheduler, train_loader, val_loader
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    return best_val_loss


def _save_best_params(study: optuna.Study, model_name: str, output_path: Path) -> None:
    best = dict(study.best_params)
    structured = {
        "baseline_model": model_name,
        "best_value": float(study.best_value),
        "params": best,
        "suggested_config_patch": {
            "model": {"Baseline": {"model_name": model_name}},
            "train": {
                "Baseline": {
                    "lr": float(best["lr"]),
                    "weight_decay": float(best["weight_decay"]),
                    "optimizer": best["optimizer"],
                    "use_lookahead": bool(best["use_lookahead"]),
                    "gradient_clip": float(best["gradient_clip"]),
                    "scheduler": {
                        "power": int(best["scheduler_power"]),
                        "warmup_steps": int(best["warmup_steps"]),
                        "warmup_type": "linear",
                        "min_lr": float(best["lr"] * best["min_lr_ratio"]),
                        "min_lr_warmup": float(best["lr"] * best["min_lr_ratio"] * 0.1),
                    },
                }
            },
            "data": {"Baseline": {"train_batch_size": int(best["batch_size"])}}
        },
    }

    with open(output_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(structured, f, sort_keys=False)


def run_hpo(
    baseline_model: str,
    n_trials: int = _N_TRIALS_DEFAULT,
    hpo_epochs: int = _HPO_EPOCHS_DEFAULT,
    study_name: str = "baseline_hpo",
    storage: Optional[str] = None,
    device: Optional[str] = None,
) -> optuna.Study:
    model_name = baseline_model.lower()
    if model_name not in VALID_MODELS:
        raise ValueError(f"baseline_model must be one of {sorted(VALID_MODELS)}")

    device_str = device or ("cuda" if torch.cuda.is_available() else "cpu")
    device_obj = torch.device(device_str)

    data_cfg = load_data_config()["Baseline"]

    train_dataset = SimulatedDataset(
        npy_path=data_cfg["data_path"],
        split=tuple(data_cfg["split"]),
        mode="train",
        horizon=int(data_cfg["horizon"]),
        normalize=bool(data_cfg["normalize"]),
    )
    val_dataset = SimulatedDataset(
        npy_path=data_cfg["data_path"],
        split=tuple(data_cfg["split"]),
        mode="val",
        horizon=int(data_cfg["horizon"]),
        normalize=bool(data_cfg["normalize"]),
        stats=train_dataset.get_stats(),
    )

    input_channels = train_dataset[0]["x_t"].shape[0]
    num_workers = data_cfg.get("loader", {}).get("loaders_num_workers", 0)

    sampler = TPESampler(
        seed=SEED,
        n_startup_trials=_N_STARTUP_TRIALS,
        multivariate=True,
        warn_independent_sampling=False,
    )
    pruner = MedianPruner(
        n_startup_trials=_N_STARTUP_TRIALS,
        n_warmup_steps=_PRUNER_WARMUP,
        interval_steps=1,
    )

    composed_study_name = f"{study_name}_{model_name}"
    study = optuna.create_study(
        study_name=composed_study_name,
        direction="minimize",
        sampler=sampler,
        pruner=pruner,
        storage=storage,
        load_if_exists=(storage is not None),
    )

    study.optimize(
        lambda trial: objective(
            trial,
            train_dataset,
            val_dataset,
            input_channels,
            device_obj,
            num_workers,
            hpo_epochs,
            model_name,
        ),
        n_trials=n_trials,
        show_progress_bar=True,
        gc_after_trial=True,
    )

    out_path = Path(f"baseline_hpo_best_params_{model_name}.yaml")
    _save_best_params(study, model_name, out_path)
    print(f"Best params saved to: {out_path}")
    print(f"Best value (val MSE): {study.best_value:.6f}")

    return study


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Bayesian hyperparameter optimization for baseline forecasters."
    )
    parser.add_argument(
        "--baseline-model",
        type=str,
        default="unet",
        choices=sorted(VALID_MODELS),
        help="Baseline architecture to optimize.",
    )
    parser.add_argument("--n-trials", type=int, default=_N_TRIALS_DEFAULT)
    parser.add_argument("--hpo-epochs", type=int, default=_HPO_EPOCHS_DEFAULT)
    parser.add_argument("--study-name", type=str, default="baseline_hpo")
    parser.add_argument("--storage", type=str, default=None)
    parser.add_argument("--device", type=str, default=None, choices=["cuda", "cpu"])

    args = parser.parse_args()
    run_hpo(
        baseline_model=args.baseline_model,
        n_trials=args.n_trials,
        hpo_epochs=args.hpo_epochs,
        study_name=args.study_name,
        storage=args.storage,
        device=args.device,
    )
