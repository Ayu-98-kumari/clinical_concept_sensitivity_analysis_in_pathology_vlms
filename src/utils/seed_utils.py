"""
Reproducibility utilities for setting random seeds.

This module ensures reproducibility across different runs
by setting seeds for all randomness sources.
"""

import random
import numpy as np
import torch
import os


def set_seed(seed: int = 42):
    """
    Set all random seeds for reproducibility.

    Sets seeds for:
        - Python's random module
        - NumPy
        - PyTorch (CPU and CUDA)
        - PyTorch backends (cuDNN)

    Args:
        seed: Random seed value

    Example:
        >>> set_seed(42)
        >>> # Now all random operations will be reproducible
    """
    # Python random module
    random.seed(seed)

    # NumPy
    np.random.seed(seed)

    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU

    # PyTorch backends
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Set environment variable for Python hash seed
    os.environ['PYTHONHASHSEED'] = str(seed)


def get_rng_state():
    """
    Get current random number generator states.

    Returns:
        Dictionary containing RNG states

    Example:
        >>> state = get_rng_state()
        >>> # Do some random operations
        >>> set_rng_state(state)  # Restore state
    """
    return {
        'python': random.getstate(),
        'numpy': np.random.get_state(),
        'torch': torch.get_rng_state(),
        'torch_cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    }


def set_rng_state(state: dict):
    """
    Set random number generator states.

    Args:
        state: Dictionary containing RNG states (from get_rng_state)

    Example:
        >>> state = get_rng_state()
        >>> set_rng_state(state)
    """
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])

    if state['torch_cuda'] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state['torch_cuda'])
