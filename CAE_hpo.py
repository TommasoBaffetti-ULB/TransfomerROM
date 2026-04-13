"""
CAE Hyperparameter Optimization using Optuna (Bayesian / TPE).

Searches jointly over architecture and training hyperparameters to minimise
the CAE validation MSE.  Trials are pruned early via MedianPruner so
unpromising configurations are discarded without wasting compute.

Usage
-----
# quick smoke-test (3 trials, 2 epochs each)
python CAE_hpo.py --n-trials 3 --hpo-epochs 2

# full run
python CAE_hpo.py --n-trials 50 --hpo-epochs 20

# persistent study (resumable, visualisable with optuna-dashboard)
python CAE_hpo.py --n-trials 100 --hpo-epochs 25 --storage sqlite:///hpo.db
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

from ArtFire.DL.Models.CAE import CAE, ConvEncoder, ConvDecoder
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
_N_STARTUP_TRIALS = 10   # random exploration before TPE kicks in
_PRUNER_WARMUP = 5       # epochs before the pruner is allowed to fire

# Joint architecture catalogue encoding (hidden_channels, output_channels) pairs.
# Every entry satisfies output_channels > hidden_channels[-1] by construction.
# Channels are monotonically non-decreasing throughout the encoder.
# Add or remove entries here to change the search space.
_ARCH_CONFIGS: dict[str, dict] = {
    # 0 hidden layers — any output is valid
    "[]→16":          {"hidden_channels": [],           "output_channels": 16},
    "[]→32":          {"hidden_channels": [],           "output_channels": 32},
    "[]→64":          {"hidden_channels": [],           "output_channels": 64},
    "[]→128":         {"hidden_channels": [],           "output_channels": 128},
    # 1 hidden layer
    "[16]→32":        {"hidden_channels": [16],         "output_channels": 32},
    "[16]→64":        {"hidden_channels": [16],         "output_channels": 64},
    "[16]→128":       {"hidden_channels": [16],         "output_channels": 128},
    "[32]→64":        {"hidden_channels": [32],         "output_channels": 64},
    "[32]→128":       {"hidden_channels": [32],         "output_channels": 128},
    "[64]→128":       {"hidden_channels": [64],         "output_channels": 128},
    # 2 hidden layers
    "[16,32]→64":     {"hidden_channels": [16, 32],     "output_channels": 64},
    "[16,32]→128":    {"hidden_channels": [16, 32],     "output_channels": 128},
    "[16,64]→128":    {"hidden_channels": [16, 64],     "output_channels": 128},
    "[32,64]→128":    {"hidden_channels": [32, 64],     "output_channels": 128},
    # 3 hidden layers
    "[16,32,64]→128": {"hidden_channels": [16, 32, 64], "output_channels": 128},
}


# ---------------------------------------------------------------------------
# Hyperparameter suggestion
# ---------------------------------------------------------------------------

def _suggest_hparams(trial: optuna.Trial) -> dict:
    """Sample the full set of CAE hyperparameters from an Optuna trial."""

    # --- Architecture --------------------------------------------------
    # Single categorical over the joint (hidden, output) catalogue.
    # The constraint output_channels > last(hidden_channels) is guaranteed by
    # construction — no need for rejection sampling or dynamic parameter domains.
    arch_key = trial.suggest_categorical("arch", list(_ARCH_CONFIGS.keys()))
    arch_cfg = _ARCH_CONFIGS[arch_key]
    hidden_channels: list[int] = arch_cfg["hidden_channels"]
    output_channels: int = arch_cfg["output_channels"]

    normalization = trial.suggest_categorical(
        "normalization", ["Batch-Norm", "Instance-Norm", "Group-Norm", None]
    )
    activation = trial.suggest_categorical(
        "activation", ["relu", "gelu", "elu", "silu"]
    )

    # --- Training ------------------------------------------------------
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True)
    optimizer_name = trial.suggest_categorical("optimizer", ["adamw", "adam"])
    use_lookahead = trial.suggest_categorical("use_lookahead", [True, False])
    gradient_clip = trial.suggest_categorical(
        "gradient_clip", [-1.0, 1.0, 3.0, 5.0]
    )
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
    scheduler_power = trial.suggest_int("scheduler_power", 1, 3)
    warmup_steps = trial.suggest_int("warmup_steps", 0, 20)
    # min_lr as a ratio of lr to guarantee min_lr < lr always
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


# ---------------------------------------------------------------------------
# Model construction
# ---------------------------------------------------------------------------

def _build_model(
    hp: dict,
    input_channels: int,
    sample_x: torch.Tensor,
    device: torch.device,
) -> CAE:
    """
    Build a CAE from the sampled hyperparameters.

    output_paddings pattern: [(0,0)] * n_hidden + [(1,1)]
    Verified correct for H=256, W=64 with k=3, stride=2, padding=0:
    all intermediate decoder reconstructions are exact (op=(0,0)); only the
    final reconstruction to 256×64 requires op=(1,1).
    """
    hidden_channels: list[int] = hp["hidden_channels"]

    conv_config = {
        "input_channels": input_channels,
        "output_channels": hp["output_channels"],
        "hidden_channels": hidden_channels,
        "dim": 2,
        "paddings": 0,
        "strides": 2,
        "normalization": hp["normalization"],
        "activation": hp["activation"],
    }

    encoder = ConvEncoder(conv_config).to(device)

    with torch.no_grad():
        no_flatten_dim = encoder.get_no_flatten_dim(sample_x.to(device))

    output_paddings = [(0, 0)] * len(hidden_channels) + [(1, 1)]

    tconv_config = {
        "input_channels": hp["output_channels"],
        "output_channels": input_channels,
        "hidden_channels": hidden_channels[::-1],
        "paddings": 0,
        "output_paddings": output_paddings,
        "strides": 2,
        "normalization": hp["normalization"],
        "activation": hp["activation"],
    }

    decoder = ConvDecoder(tconv_config, no_flatten_dim).to(device)
    return CAE(encoder, decoder).to(device)


# ---------------------------------------------------------------------------
# Optuna objective
# ---------------------------------------------------------------------------

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

    # Build model — prune if architecture is invalid (e.g. spatial dims collapse)
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
    clip = hp["gradient_clip"]

    try:
        for epoch in range(hpo_epochs):
            # --- train ---------------------------------------------------
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

            # --- validate ------------------------------------------------
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

            # report for pruning
            trial.report(val_loss, epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

    finally:
        del model, optimizer, scheduler, train_loader, val_loader
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    return best_val_loss


# ---------------------------------------------------------------------------
# Save results
# ---------------------------------------------------------------------------

def _save_best_params(best_params: dict, output_path: Path) -> None:
    """
    Write the best hyperparameters to a structured YAML file.
    Formatted to be copy-pasteable into model.yaml / train.yaml.

    Note: best_params contains the raw Optuna-tracked parameters
    (hidden_size_0/1/2, n_hidden_layers, min_lr_ratio), so hidden_channels
    and min_lr are reconstructed here from those raw values.
    """
    arch_cfg = _ARCH_CONFIGS[best_params["arch"]]
    hidden_channels = arch_cfg["hidden_channels"]
    output_channels = arch_cfg["output_channels"]
    min_lr = best_params["lr"] * best_params["min_lr_ratio"]

    structured = {
        "model": {
            "CAE": {
                "conv_config": {
                    "output_channels": output_channels,
                    "hidden_channels": hidden_channels,
                    "dim": 2,
                    "paddings": 0,
                    "strides": 2,
                    "normalization": best_params["normalization"],
                    "activation": best_params["activation"],
                }
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


# ---------------------------------------------------------------------------
# Main HPO runner
# ---------------------------------------------------------------------------

def run_hpo(
    n_trials: int = _N_TRIALS_DEFAULT,
    hpo_epochs: int = _HPO_EPOCHS_DEFAULT,
    study_name: str = "cae_hpo",
    storage: Optional[str] = None,
    device: Optional[str] = None,
) -> optuna.Study:
    """
    Run Bayesian hyperparameter optimisation for the CAE.

    Parameters
    ----------
    n_trials:
        Total number of Optuna trials.
    hpo_epochs:
        Training epochs per trial (shorter than full training).
    study_name:
        Name for the Optuna study (used with persistent storage).
    storage:
        Optuna storage URL, e.g. ``"sqlite:///hpo.db"``.
        Use None for in-memory (non-persistent) runs.
    device:
        ``"cuda"`` or ``"cpu"``.  Auto-detected if None.

    Returns
    -------
    optuna.Study
    """
    if device is None:
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device_str = device
    device_obj = torch.device(device_str)

    if device_obj.type == "cuda":
        print("CUDA available — running on GPU")

    # ------------------------------------------------------------------
    # Load data once; datasets are reused across all trials
    # ------------------------------------------------------------------
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

    sample = train_dataset[0]["x"]          # [C, H, W]
    input_channels = sample.shape[0]
    sample_x = sample.unsqueeze(0)          # [1, C, H, W] for get_no_flatten_dim

    num_workers = cae_config.get("loader", {}).get("loaders_num_workers", 0)

    print(
        f"Dataset: {len(train_dataset)} train / {len(val_dataset)} val samples, "
        f"C={input_channels}, H={sample.shape[1]}, W={sample.shape[2]}"
    )
    print(
        f"\nHPO config: {n_trials} trials × {hpo_epochs} epochs on {device_str}\n"
    )

    # ------------------------------------------------------------------
    # Create Optuna study
    # ------------------------------------------------------------------
    # multivariate=True (MOTPE) models inter-parameter correlations, which
    # significantly outperforms independent univariate TPE.
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

    # ------------------------------------------------------------------
    # Report results
    # ------------------------------------------------------------------
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

    output_path = Path("hpo_best_params.yaml")
    _save_best_params(study.best_params, output_path)
    print(f"\nBest params saved to: {output_path}")

    return study


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Bayesian hyperparameter optimisation for the CAE model."
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
        default="cae_hpo",
        help="Optuna study name (default: cae_hpo)",
    )
    parser.add_argument(
        "--storage",
        type=str,
        default=None,
        help=(
            "Optuna storage URL for persistence, e.g. sqlite:///hpo.db. "
            "Enables resuming runs and optuna-dashboard visualisation."
        ),
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cuda", "cpu"],
        help="Device (auto-detected if omitted)",
    )
    args = parser.parse_args()

    run_hpo(
        n_trials=args.n_trials,
        hpo_epochs=args.hpo_epochs,
        study_name=args.study_name,
        storage=args.storage,
        device=args.device,
    )
