"""Opinionated presets for common training scenarios.

Each preset provides a recommended scheduler configuration for a specific
training task, reducing choice paralysis and misconfiguration risk.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torch.optim import Optimizer
    from torch.optim.lr_scheduler import LRScheduler

PRESETS: dict[str, dict] = {
    "llm_pretrain": {
        "scheduler": "wsd",
        "description": "Warmup-Stable-Decay for large language model pretraining",
        "defaults": {
            "warmup_ratio": 0.01,
            "stable_ratio": 0.7,
            "decay_type": "cosine",
        },
        "when_to_use": "Pre-training LLMs from scratch with known compute budget",
        "when_not_to_use": "Fine-tuning or small-scale training",
    },
    "llm_finetune": {
        "scheduler": "cosine_with_warmup",
        "description": "Cosine decay with short warmup for LLM fine-tuning",
        "defaults": {
            "warmup_ratio": 0.03,
            "min_lr_ratio": 0.1,
        },
        "when_to_use": "Fine-tuning pre-trained LLMs on downstream tasks",
        "when_not_to_use": "Pre-training from scratch",
    },
    "vision_finetune": {
        "scheduler": "cosine_with_warmup",
        "description": "Cosine schedule with moderate warmup for vision fine-tuning",
        "defaults": {
            "warmup_ratio": 0.05,
            "min_lr_ratio": 0.01,
        },
        "when_to_use": "Fine-tuning ViTs or CNNs on image classification/detection",
        "when_not_to_use": "Training from scratch on ImageNet-scale data",
    },
    "vision_pretrain": {
        "scheduler": "warmup_hold_cosine",
        "description": "Hold at peak LR then cosine decay for vision pre-training",
        "defaults": {
            "warmup_ratio": 0.05,
            "hold_ratio": 0.3,
            "min_lr_ratio": 0.001,
        },
        "when_to_use": "Pre-training ViTs on large image datasets",
        "when_not_to_use": "Quick fine-tuning experiments",
    },
    "transfer_small_data": {
        "scheduler": "slanted_triangular",
        "description": "ULMFiT-style schedule for transfer learning with limited data",
        "defaults": {
            "cut_frac": 0.1,
        },
        "when_to_use": "Fine-tuning with small datasets (<10K samples)",
        "when_not_to_use": "Large-scale pre-training",
    },
    "budgeted_training": {
        "scheduler": "rex",
        "description": "REX schedule for training with fixed compute budget",
        "defaults": {},
        "when_to_use": "When compute budget is fixed and you want optimal allocation",
        "when_not_to_use": "When you can afford to tune the schedule",
    },
}


def list_presets() -> list[str]:
    """List all available preset names."""
    return list(PRESETS.keys())


def get_preset_info(name: str) -> dict:
    """Get detailed information about a preset."""
    if name not in PRESETS:
        available = ", ".join(PRESETS.keys())
        raise ValueError(f"Unknown preset '{name}'. Available: {available}")
    return PRESETS[name]


def create_from_preset(
    optimizer: Optimizer,
    preset: str,
    total_steps: int,
    **overrides,
) -> LRScheduler:
    """Create a scheduler from a named preset.

    Args:
        optimizer: The optimizer to schedule.
        preset: Preset name (e.g., 'llm_pretrain', 'vision_finetune').
        total_steps: Total number of training steps.
        **overrides: Override any default parameters.

    Returns:
        A configured scheduler instance.

    Example:
        >>> scheduler = create_from_preset(optimizer, 'llm_pretrain', total_steps=100000)
    """
    from pytorch_scheduler.scheduler import create_scheduler

    if preset not in PRESETS:
        available = ", ".join(PRESETS.keys())
        raise ValueError(f"Unknown preset '{preset}'. Available: {available}")

    info = PRESETS[preset]
    scheduler_name = info["scheduler"]
    defaults = dict(info["defaults"])

    # Convert ratio-based defaults to absolute values
    if "warmup_ratio" in defaults:
        defaults["warmup_steps"] = int(total_steps * defaults.pop("warmup_ratio"))
    if "stable_ratio" in defaults:
        defaults["stable_steps"] = int(total_steps * defaults.pop("stable_ratio"))
    if "hold_ratio" in defaults:
        defaults["hold_steps"] = int(total_steps * defaults.pop("hold_ratio"))
    if "min_lr_ratio" in defaults:
        # Get base_lr from optimizer
        base_lr = optimizer.param_groups[0]["lr"]
        defaults["min_lr"] = base_lr * defaults.pop("min_lr_ratio")

    defaults["total_steps"] = total_steps
    defaults.update(overrides)

    return create_scheduler(optimizer, scheduler_name, **defaults)
