import random
import numpy as np
import torch

def set_seed(seed: int = 42) -> None:
    """Set seed for all sources of randomness in the ArtFire pipeline.

    Covers: Python built-ins, NumPy, PyTorch (CPU + CUDA), and
    DataLoader worker processes.

    Args:
        seed: The integer seed to use everywhere.
    """
    # Python built-in RNG (used by random.shuffle, etc.)
    random.seed(seed)

    # NumPy RNG (used by dataset transforms, splits, etc.)
    np.random.seed(seed)

    # PyTorch CPU RNG
    torch.manual_seed(seed)

    # PyTorch CUDA RNG — covers all GPUs
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for multi-GPU setups

    # Make cuDNN deterministic (slight perf cost, but fully reproducible)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int) -> None:
    """Per-worker seed function to pass to DataLoader.

    DataLoader spawns fresh worker processes that inherit no seed state.
    This ensures each worker gets a unique but deterministic seed derived
    from the global PyTorch seed set in set_seed().

    Usage:
        DataLoader(..., worker_init_fn=seed_worker, generator=torch.Generator().manual_seed(seed))
    """
    worker_seed = torch.initial_seed() % (2 ** 32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)