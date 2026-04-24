"""
CAEst Hyperparameter Optimization using Optuna (Bayesian / TPE).

Searches jointly over architecture and training hyperparameters to minimise
the CAEst validation MSE. Trials are pruned early via MedianPruner so
unpromising configurations are discarded without wasting compute.

Usage
-----
# quick smoke-test (3 trials, 2 epochs each)
python CAEst_hpo.py --n-trials 3 --hpo-epochs 2

# full run
python CAEst_hpo.py --n-trials 50 --hpo-epochs 20

# persistent study (resumable, visualisable with optuna-dashboard)
python CAEst_hpo.py --n-trials 100 --hpo-epochs 25 --storage sqlite:///hpo.db
"""
from __future__ import annotations

import argparse
import gc
import warnings
from pathlib import Path
from typing import Optional

import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pytorch_scheduler.scheduler.polynomial import PolynomialScheduler
import yaml

from ArtFire.DL.Models.CAE import CAEst, ConvEncoderst, ConvDecoderst
from ArtFire.Data.CAEDataset import CAEDataset
from ArtFire.DL.Optimization.optimizers import build_optimizer, build_parameter_groups
from ArtFire.DL.Optimization.warmup import WarmupScheduler
from ArtFire.utils.config import load_data_config
from ArtFire.utils.seed import set_seed, seed_worker

# ---------------------------------------------------------------------------
# Global constants
# ---------------------------------------------------------------------------
SEED = 42
set_seed(SEED)
_g = torch.Generator()
_g.manual_seed(SEED)

_N_TRIALS_DEFAULT = 50
_HPO_EPOCHS_DEFAULT = 20
_N_STARTUP_TRIALS = 10
_PRUNER_WARMUP = 5

# Joint architecture catalogue encoding (hidden_channels, output_channels) pairs.
_ARCH_CONFIGS: dict[str, dict] = {
    "[]→16": {"hidden_channels": [], "output_channels": 16},
    "[]→32": {"hidden_channels": [], "output_channels": 32},
    "[]→64": {"hidden_channels": [], "output_channels": 64},
    "[]→128": {"hidden_channels": [], "output_channels": 128},
    "[16]→32": {"hidden_channels": [16], "output_channels": 32},
    "[16]→64": {"hidden_channels": [16], "output_channels": 64},
    "[16]→128": {"hidden_channels": [16], "output_channels": 128},
    "[32]→64": {"hidden_channels": [32], "output_channels": 64},
    "[32]→128": {"hidden_channels": [32], "output_channels": 128},
    "[64]→128": {"hidden_channels": [64], "output_channels": 128},
    "[16,32]→64": {"hidden_channels": [16, 32], "output_channels": 64},
    "[16,32]→128": {"hidden_channels": [16, 32], "output_channels": 128},
    "[16,64]→128": {"hidden_channels": [16, 64], "output_channels": 128},
    "[32,64]→128": {"hidden_channels": [32, 64], "output_channels": 128},
    "[16,32,64]→128": {"hidden_channels": [16, 32, 64], "output_channels": 128},
}


def _suggest_hparams(trial: optuna.Trial) -> dict:
    """Sample the full set of CAEst hyperparameters from an Optuna trial."""

    arch_key = trial.suggest_categorical("arch", list(_ARCH_CONFIGS.keys()))
    arch_cfg = _ARCH_CONFIGS[arch_key]
    hidden_channels: list[int] = arch_cfg["hidden_channels"]
    output_channels: int = arch_cfg["output_channels"]

    normalization = trial.suggest_categorical(
        "normalization", ["Batch-Norm", "Instance-Norm", "Group-Norm", None]
    )
    activation = trial.suggest_categorical("activation", ["relu", "gelu", "elu", "silu"])

    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True)
    optimizer_name = trial.suggest_categorical("optimizer", ["adamw", "adam"])
    use_lookahead = trial.suggest_categorical("use_lookahead", [True, False])
    gradient_clip = trial.suggest_categorical("gradient_clip", [-1.0, 1.0, 3.0, 5.0])
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
    scheduler_power = trial.suggest_int("scheduler_power", 1, 3)
    warmup_steps = trial.suggest_int("warmup_steps", 0, 20)
    min_lr_ratio = trial.suggest_float("min_lr_ratio", 0.01, 0.5)

    return {
        "output_channels": output_channels,
        "hidden_channels": hidden_channels,
        "normalization": normalization,
        "activation": activation,
        "lr": lr,
        "weight_decay": weight_decay,
        "optimizer": optimizer_name,
        "use_lookahead": use_lookahead,
        "gradient_clip": gradient_clip,
        "batch_size": batch_size,
        "scheduler_power": scheduler_power,
        "warmup_steps": warmup_steps,
        "min_lr": lr * min_lr_ratio,
    }


def _build_model(
    hp: dict,
    input_channels: int,
    sample_x: torch.Tensor,
    device: torch.device,
) -> CAEst:
    """Build a CAEst from the sampled hyperparameters."""
    hidden_channels: list[int] = hp["hidden_channels"]

    sp_conv_config = {
        "input_channels": input_channels,
        "output_channels": hp["output_channels"],
        "hidden_channels": hidden_channels,
        "dim": 2,
        "paddings": 0,
        "strides": 2,
        "normalization": hp["normalization"],
        "activation": hp["activation"],
    }
    t_conv_config = {
        "input_channels": 1,
        "output_channels": hp["output_channels"],
        "hidden_channels": hidden_channels,
        "dim": 2,
        "paddings": 0,
        "strides": 2,
        "normalization": hp["normalization"],
        "activation": hp["activation"],
    }

    encoder = ConvEncoderst(sp_conv_config, t_conv_config).to(device)

    with torch.no_grad():
        x = encoder.sp_conv(sample_x.to(device))
        bsz, h_tokens = x.shape[0], x.shape[2]
        x = x.permute((0, 2, 1, 3)).unsqueeze(2).reshape(bsz * h_tokens, 1, x.shape[1], x.shape[3])
        no_flatten_dim = encoder.t_conv(x).shape[1:]

    sp_output_paddings = [(0, 0)] * len(hidden_channels) + [(1, 1)]
    t_output_paddings = [(0, 0)] * len(hidden_channels) + [(1, 0)]

    sp_tconv_config = {
        "input_channels": hp["output_channels"],
        "output_channels": input_channels,
        "hidden_channels": hidden_channels[::-1],
        "dim": 2,
        "paddings": 0,
        "output_paddings": sp_output_paddings,
        "strides": 2,
        "normalization": hp["normalization"],
        "activation": hp["activation"],
    }
    t_tconv_config = {
        "input_channels": hp["output_channels"],
        "output_channels": 1,
        "hidden_channels": hidden_channels[::-1],
        "dim": 2,
        "paddings": 0,
        "output_paddings": t_output_paddings,
        "strides": 2,
        "normalization": hp["normalization"],
        "activation": hp["activation"],
    }

    decoder = ConvDecoderst(sp_tconv_config, t_tconv_config, no_flatten_dim).to(device)
    return CAEst(encoder, decoder).to(device)


def objective(
    trial: optuna.Trial,
    train_dataset: CAEDataset,
    val_dataset: CAEDataset,
    input_channels: int,
    sample_x: torch.Tensor,
    device: torch.device,
    num_workers: int,
    hpo_epochs: int,
) -> float:
    """Optuna objective: train for hpo_epochs and return the best val MSE."""
    hp = _suggest_hparams(trial)

    try:
        model = _build_model(hp, input_channels, sample_x, device)
    except Exception as exc:
        warnings.warn(f"Trial {trial.number}: invalid architecture — {exc}")
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

    parameter_groups = build_parameter_groups(model, lr=hp["lr"], weight_decay=hp["weight_decay"])
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
    clip = hp["gradient_clip"]

    try:
        for epoch in range(hpo_epochs):
            model.train()
            for batch in train_loader:
                x = batch["x"].to(device)
                y = batch["y"].to(device)
                optimizer.zero_grad()
                loss = criterion(model(x), y)
                loss.backward()
                if clip > 0:
                    torch.nn.utils.clip_grad_value_(model.parameters(), clip)
                optimizer.step()
            scheduler.step()

            model.eval()
            running_loss = total = 0
            with torch.no_grad():
                for batch in val_loader:
                    x = batch["x"].to(device)
                    y = batch["y"].to(device)
                    loss = criterion(model(x), y)
                    running_loss += loss.item() * x.size(0)
                    total += x.size(0)

            val_loss = running_loss / total
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


def _save_best_params(best_params: dict, output_path: Path) -> None:
    """Write best CAEst hyperparameters to a structured YAML file."""
    arch_cfg = _ARCH_CONFIGS[best_params["arch"]]
    hidden_channels = arch_cfg["hidden_channels"]
    output_channels = arch_cfg["output_channels"]
    min_lr = best_params["lr"] * best_params["min_lr_ratio"]

    structured = {
        "model": {
            "CAEst": {
                "spatial_conv_config": {
                    "output_channels": output_channels,
                    "hidden_channels": hidden_channels,
                    "dim": 2,
                    "paddings": 0,
                    "strides": 2,
                    "normalization": best_params["normalization"],
                    "activation": best_params["activation"],
                },
                "temporal_conv_config": {
                    "output_channels": output_channels,
                    "hidden_channels": hidden_channels,
                    "dim": 2,
                    "paddings": 0,
                    "strides": 2,
                    "normalization": best_params["normalization"],
                    "activation": best_params["activation"],
                },
                "spatial_tconv_config": {
                    "input_channels": output_channels,
                    "output_channels": "<set_input_channels>",
                    "hidden_channels": hidden_channels[::-1],
                    "dim": 2,
                    "paddings": 0,
                    "output_paddings": [(0, 0)] * len(hidden_channels) + [(1, 1)],
                    "strides": 2,
                    "normalization": best_params["normalization"],
                    "activation": best_params["activation"],
                },
                "temporal_tconv_config": {
                    "input_channels": output_channels,
                    "output_channels": 1,
                    "hidden_channels": hidden_channels[::-1],
                    "dim": 2,
                    "paddings": 0,
                    "output_paddings": [(0, 0)] * len(hidden_channels) + [(1, 0)],
                    "strides": 2,
                    "normalization": best_params["normalization"],
                    "activation": best_params["activation"],
                },
                "no_flatten_dim": "<compute_from_sample>",
            }
        },
        "training": {
            "CAE": {
                "lr": float(best_params["lr"]),
                "weight_decay": float(best_params["weight_decay"]),
                "optimizer": best_params["optimizer"],
                "use_lookahead": bool(best_params["use_lookahead"]),
                "gradient_clip": float(best_params["gradient_clip"]),
                "train_batch_size": int(best_params["batch_size"]),
                "scheduler": {
                    "power": int(best_params["scheduler_power"]),
                    "warmup_steps": int(best_params["warmup_steps"]),
                    "warmup_type": "linear",
                    "min_lr": float(min_lr),
                    "min_lr_warmup": float(min_lr * 0.1),
                },
            }
        },
    }
    with open(output_path, "w") as f:
        yaml.dump(structured, f, default_flow_style=False, sort_keys=False)


def run_hpo(
    n_trials: int = _N_TRIALS_DEFAULT,
    hpo_epochs: int = _HPO_EPOCHS_DEFAULT,
    study_name: str = "caest_hpo",
    storage: Optional[str] = None,
    device: Optional[str] = None,
) -> optuna.Study:
    """Run Bayesian hyperparameter optimisation for the CAEst."""
    if device is None:
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device_str = device
    device_obj = torch.device(device_str)

    if device_obj.type == "cuda":
        print("CUDA available — running on GPU")

    data_config = load_data_config()
    cae_config = data_config["CAE"]

    print(f"\nLoading datasets from: {cae_config['data_path']}")
    train_dataset = CAEDataset(
        npy_path=cae_config["data_path"],
        split=cae_config["split"],
        mode="train",
        normalize=cae_config["normalize"],
    )
    val_dataset = CAEDataset(
        npy_path=cae_config["data_path"],
        split=cae_config["split"],
        mode="val",
        normalize=cae_config["normalize"],
        stats=train_dataset.get_stats(),
    )

    sample = train_dataset[0]["x"]
    input_channels = sample.shape[0]
    sample_x = sample.unsqueeze(0)

    num_workers = cae_config.get("loader", {}).get("loaders_num_workers", 0)

    print(
        f"Dataset: {len(train_dataset)} train / {len(val_dataset)} val samples, "
        f"C={input_channels}, H={sample.shape[1]}, W={sample.shape[2]}"
    )
    print(f"\nHPO config: {n_trials} trials × {hpo_epochs} epochs on {device_str}\n")

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
    study = optuna.create_study(
        study_name=study_name,
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
            sample_x,
            device_obj,
            num_workers,
            hpo_epochs,
        ),
        n_trials=n_trials,
        show_progress_bar=True,
        gc_after_trial=True,
    )

    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    pruned = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]

    print("\n" + "=" * 60)
    print("HPO Complete")
    print("=" * 60)
    print(f"  Completed trials : {len(completed)}")
    print(f"  Pruned trials    : {len(pruned)}")
    print(f"  Best trial       : #{study.best_trial.number}")
    print(f"  Best val MSE     : {study.best_value:.6f}")
    print("\nBest hyperparameters:")
    for k, v in study.best_params.items():
        print(f"  {k:22s}: {v}")

    output_path = Path("caest_hpo_best_params.yaml")
    _save_best_params(study.best_params, output_path)
    print(f"\nBest params saved to: {output_path}")

    return study


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Bayesian hyperparameter optimisation for the CAEst model."
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=_N_TRIALS_DEFAULT,
        help=f"Number of Optuna trials (default: {_N_TRIALS_DEFAULT})",
    )
    parser.add_argument(
        "--hpo-epochs",
        type=int,
        default=_HPO_EPOCHS_DEFAULT,
        help=f"Training epochs per trial (default: {_HPO_EPOCHS_DEFAULT})",
    )
    parser.add_argument(
        "--study-name",
        type=str,
        default="caest_hpo",
        help="Optuna study name (default: caest_hpo)",
    )
    parser.add_argument(
        "--storage",
        type=str,
        default=None,
        help='Optuna storage URL, e.g. "sqlite:///hpo.db" (default: in-memory)',
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "cuda"],
        default=None,
        help="Device override (default: auto-detect)",
    )

    args = parser.parse_args()
    run_hpo(
        n_trials=args.n_trials,
        hpo_epochs=args.hpo_epochs,
        study_name=args.study_name,
        storage=args.storage,
        device=args.device,
    )
