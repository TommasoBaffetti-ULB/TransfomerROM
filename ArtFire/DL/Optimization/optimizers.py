from collections import defaultdict

import torch
from torch.optim.optimizer import Optimizer


from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, Iterator, List, Mapping, Optional, Union

"""  How to use it
from Optimization.optimizers import build_optimizer

optimizer = build_optimizer(
    model.parameters(),
    {
        "name": "adamw",
        "lr": 3e-4,
        "weight_decay": 1e-4,
        use_lookahead=True
    },
)
"""

ParamsLike = Union[
    Iterable[torch.nn.Parameter],
    Iterable[Dict[str, Any]],
]

@dataclass
class OptimizerConfig:
    """
    Generic optimizer configuration.

    Notes
    -----
    - `name` is case-insensitive.
    - `kwargs` stores optimizer-specific arguments not explicitly listed here.
    - `lookahead` can wrap the base optimizer if enabled.
    """
    name: str
    lr: float
    weight_decay: float = 0.0

    # Lookahead wrapper
    use_lookahead: bool = False

    # Extra optimizer-specific kwargs
    kwargs: Dict[str, Any] = field(default_factory=dict)

def _normalize_name(name: str) -> str:
    return name.strip().lower().replace("_", "").replace("-", "")

def _validate_params(params: ParamsLike) -> None:
    if params is None:
        raise ValueError("`params` cannot be None.")


def _config_from_dict(config: Mapping[str, Any]) -> OptimizerConfig:
    config = dict(config)

    known_fields = {
        "name",
        "lr",
        "weight_decay",
        "use_lookahead",
        "kwargs",
    }

    extra_kwargs = config.get("kwargs", {}).copy()
    for key in list(config.keys()):
        if key not in known_fields:
            extra_kwargs[key] = config.pop(key)

    config["kwargs"] = extra_kwargs
    return OptimizerConfig(**config)

def _build_sgd(params: ParamsLike, cfg: OptimizerConfig) -> Optimizer:
    return torch.optim.SGD(
        params,
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
        **cfg.kwargs,
    )


def _build_adam(params: ParamsLike, cfg: OptimizerConfig) -> Optimizer:
    return torch.optim.Adam(
        params,
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
        **cfg.kwargs,
    )


def _build_adamw(params: ParamsLike, cfg: OptimizerConfig) -> Optimizer:
    return torch.optim.AdamW(
        params,
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
        **cfg.kwargs,
    )


def _build_rmsprop(params: ParamsLike, cfg: OptimizerConfig) -> Optimizer:
    return torch.optim.RMSprop(
        params,
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
        **cfg.kwargs,
    )


def _build_adagrad(params: ParamsLike, cfg: OptimizerConfig) -> Optimizer:
    return torch.optim.Adagrad(
        params,
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
        **cfg.kwargs,
    )


def build_optimizer(
    params: ParamsLike,
    config: Union[OptimizerConfig, Mapping[str, Any]],
) -> Optimizer:
    """
    Factory for constructing optimizers.

    Parameters
    ----------
    params:
        Model parameters or parameter groups.
    config:
        Either an OptimizerConfig or a dict-like config.

    Returns
    -------
    torch.optim.Optimizer
        Instantiated optimizer, optionally wrapped with Lookahead.
    """
    _validate_params(params)

    if isinstance(config, Mapping):
        config = _config_from_dict(config)
    elif not isinstance(config, OptimizerConfig):
        raise TypeError(
            "`config` must be either an OptimizerConfig or a dict-like object."
        )

    name = _normalize_name(config.name)

    optimizer_builders = {
        "sgd": _build_sgd,
        "adam": _build_adam,
        "adamw": _build_adamw,
        "rmsprop": _build_rmsprop,
        "adagrad": _build_adagrad,
    }

    if name not in optimizer_builders:
        available = ", ".join(sorted(optimizer_builders.keys()))
        raise ValueError(f"Unsupported optimizer '{config.name}'. Available: {available}.")

    optimizer = optimizer_builders[name](params, config)

    if config.use_lookahead:
        optimizer = Lookahead(
            optimizer=optimizer,
        )

    return optimizer


class Lookahead(Optimizer):
    r"""PyTorch implementation of the lookahead wrapper.

    Lookahead Optimizer: https://arxiv.org/abs/1907.08610
    """

    def __init__(self, optimizer, la_steps=5, la_alpha=0.8, pullback_momentum="none"):
        """optimizer: inner optimizer
        la_steps (int): number of lookahead steps
        la_alpha (float): linear interpolation factor. 1.0 recovers the inner optimizer.
        pullback_momentum (str): change to inner optimizer momentum on interpolation update
        """
        self.optimizer = optimizer
        self._la_step = 0  # counter for inner optimizer
        self.la_alpha = la_alpha
        self._total_la_steps = la_steps
        pullback_momentum = pullback_momentum.lower()
        assert pullback_momentum in ["reset", "pullback", "none"]
        self.pullback_momentum = pullback_momentum

        self.state = defaultdict(dict)

        # Cache the current optimizer parameters
        for group in optimizer.param_groups:
            for p in group['params']:
                param_state = self.state[p]
                param_state['cached_params'] = torch.zeros_like(p.data)
                param_state['cached_params'].copy_(p.data)
                if self.pullback_momentum == "pullback":
                    param_state['cached_mom'] = torch.zeros_like(p.data)

    def __getstate__(self):
        return {
            'state': self.state,
            'optimizer': self.optimizer,
            'la_alpha': self.la_alpha,
            '_la_step': self._la_step,
            '_total_la_steps': self._total_la_steps,
            'pullback_momentum': self.pullback_momentum
        }

    def zero_grad(self):
        self.optimizer.zero_grad()

    def get_la_step(self):
        return self._la_step

    def state_dict(self):
        return self.optimizer.state_dict()

    def load_state_dict(self, state_dict):
        self.optimizer.load_state_dict(state_dict)

    def _backup_and_load_cache(self):
        """Useful for performing evaluation on the slow weights (which typically generalize better)
        """
        for group in self.optimizer.param_groups:
            for p in group['params']:
                param_state = self.state[p]
                param_state['backup_params'] = torch.zeros_like(p.data)
                param_state['backup_params'].copy_(p.data)
                p.data.copy_(param_state['cached_params'])

    def _clear_and_load_backup(self):
        for group in self.optimizer.param_groups:
            for p in group['params']:
                param_state = self.state[p]
                p.data.copy_(param_state['backup_params'])
                del param_state['backup_params']

    @property
    def param_groups(self):
        return self.optimizer.param_groups

    def step(self, closure=None):
        """Performs a single Lookahead optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = self.optimizer.step(closure)
        self._la_step += 1

        if self._la_step >= self._total_la_steps:
            self._la_step = 0
            # Lookahead and cache the current optimizer parameters
            for group in self.optimizer.param_groups:
                for p in group['params']:
                    param_state = self.state[p]
                    p.data.mul_(self.la_alpha).add_(param_state['cached_params'], alpha=1.0 - self.la_alpha)  # crucial line
                    param_state['cached_params'].copy_(p.data)
                    if self.pullback_momentum == "pullback":
                        internal_momentum = self.optimizer.state[p]["momentum_buffer"]
                        self.optimizer.state[p]["momentum_buffer"] = internal_momentum.mul_(self.la_alpha).add_(
                            1.0 - self.la_alpha, param_state["cached_mom"])
                        param_state["cached_mom"] = self.optimizer.state[p]["momentum_buffer"]
                    elif self.pullback_momentum == "reset":
                        self.optimizer.state[p]["momentum_buffer"] = torch.zeros_like(p.data)

        return loss

def get_optimizer_name(optimizer: Optimizer) -> str:
    """
    Return a readable optimizer name, including wrappers.
    """
    if isinstance(optimizer, Lookahead):
        return f"Lookahead({optimizer.base_optimizer.__class__.__name__})"
    return optimizer.__class__.__name__


def unwrap_optimizer(optimizer: Optimizer) -> Optimizer:
    """
    Return the base optimizer if wrapped, otherwise return the optimizer itself.
    """
    if isinstance(optimizer, Lookahead):
        return optimizer.base_optimizer
    return optimizer