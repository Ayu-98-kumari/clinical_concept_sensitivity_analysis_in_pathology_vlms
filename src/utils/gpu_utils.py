"""
GPU memory management and utilities.

This module provides utilities for managing GPU resources,
monitoring memory usage, and optimizing batch sizes.
"""

import torch
import subprocess
from typing import Optional, List, Dict


def get_available_gpus() -> List[int]:
    """
    Get list of available GPU indices.

    Returns:
        List of GPU indices

    Example:
        >>> gpus = get_available_gpus()
        >>> print(f"Available GPUs: {gpus}")
        Available GPUs: [0, 1, 2, 3]
    """
    if not torch.cuda.is_available():
        return []

    return list(range(torch.cuda.device_count()))


def get_gpu_memory_info(device: Optional[int] = None) -> Dict[str, float]:
    """
    Get GPU memory information in GB.

    Args:
        device: GPU device index (None = current device)

    Returns:
        Dictionary with memory info (allocated, reserved, free, total)

    Example:
        >>> info = get_gpu_memory_info(0)
        >>> print(f"Allocated: {info['allocated']:.2f} GB")
    """
    if not torch.cuda.is_available():
        return {
            'allocated': 0.0,
            'reserved': 0.0,
            'free': 0.0,
            'total': 0.0
        }

    if device is None:
        device = torch.cuda.current_device()

    allocated = torch.cuda.memory_allocated(device) / 1024**3  # GB
    reserved = torch.cuda.memory_reserved(device) / 1024**3  # GB
    total = torch.cuda.get_device_properties(device).total_memory / 1024**3  # GB
    free = total - allocated

    return {
        'allocated': allocated,
        'reserved': reserved,
        'free': free,
        'total': total
    }


def clear_gpu_memory():
    """
    Clear GPU memory cache.

    Useful between experiments to free up memory.

    Example:
        >>> clear_gpu_memory()
    """
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def print_gpu_memory_summary(device: Optional[int] = None):
    """
    Print GPU memory summary.

    Args:
        device: GPU device index (None = current device)

    Example:
        >>> print_gpu_memory_summary(0)
        GPU 0 Memory:
          Allocated: 2.50 GB
          Reserved: 3.00 GB
          Free: 13.50 GB
          Total: 16.00 GB
    """
    info = get_gpu_memory_info(device)

    if device is None:
        device = torch.cuda.current_device() if torch.cuda.is_available() else -1

    print(f"GPU {device} Memory:")
    print(f"  Allocated: {info['allocated']:.2f} GB")
    print(f"  Reserved: {info['reserved']:.2f} GB")
    print(f"  Free: {info['free']:.2f} GB")
    print(f"  Total: {info['total']:.2f} GB")


def estimate_max_batch_size(
    model: torch.nn.Module,
    input_shape: tuple,
    max_memory_fraction: float = 0.9
) -> int:
    """
    Estimate maximum batch size that fits in GPU memory.

    Args:
        model: PyTorch model
        input_shape: Input shape (C, H, W)
        max_memory_fraction: Maximum fraction of GPU memory to use

    Returns:
        Estimated maximum batch size

    Example:
        >>> model = MyModel()
        >>> max_bs = estimate_max_batch_size(model, (3, 224, 224))
        >>> print(f"Recommended batch size: {max_bs}")
    """
    if not torch.cuda.is_available():
        return 1

    device = next(model.parameters()).device
    info = get_gpu_memory_info(device.index if device.type == 'cuda' else None)

    # Available memory
    available_memory = info['free'] * max_memory_fraction * 1024**3  # bytes

    # Estimate memory per sample (rough approximation)
    # Model parameters
    model_memory = sum(p.numel() * p.element_size() for p in model.parameters())

    # Input memory (assume float32)
    input_memory = torch.prod(torch.tensor(input_shape)).item() * 4  # 4 bytes per float32

    # Activation memory (rough estimate: 4x input size for activations)
    activation_memory = input_memory * 4

    # Total memory per sample
    memory_per_sample = input_memory + activation_memory

    # Estimate batch size
    batch_size = int(available_memory / memory_per_sample)

    return max(1, batch_size)


def get_device(device: Optional[str] = None) -> torch.device:
    """
    Get torch device, with automatic GPU detection.

    Args:
        device: Device string ('cuda', 'cuda:0', 'cpu', or None)

    Returns:
        torch.device object

    Example:
        >>> device = get_device()
        >>> print(device)
        cuda:0  # If GPU available
    """
    if device is None:
        if torch.cuda.is_available():
            device = 'cuda'
        elif torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'

    return torch.device(device)


def set_device(device: Optional[str] = None) -> torch.device:
    """
    Set default CUDA device.

    Args:
        device: Device string or index

    Returns:
        torch.device object

    Example:
        >>> device = set_device('cuda:1')
        >>> # Now all CUDA operations will use GPU 1
    """
    device_obj = get_device(device)

    if device_obj.type == 'cuda':
        torch.cuda.set_device(device_obj)

    return device_obj


def enable_mixed_precision() -> bool:
    """
    Check if mixed precision (FP16) is available.

    Returns:
        True if mixed precision is supported

    Example:
        >>> if enable_mixed_precision():
        ...     scaler = torch.cuda.amp.GradScaler()
    """
    return torch.cuda.is_available() and torch.cuda.is_bf16_supported()


def get_gpu_utilization() -> List[Dict]:
    """
    Get GPU utilization information using nvidia-smi.

    Returns:
        List of dictionaries with GPU info

    Example:
        >>> utils = get_gpu_utilization()
        >>> for i, util in enumerate(utils):
        ...     print(f"GPU {i}: {util['utilization']}% utilized")
    """
    if not torch.cuda.is_available():
        return []

    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=index,utilization.gpu,memory.used,memory.total',
             '--format=csv,noheader,nounits'],
            capture_output=True,
            text=True
        )

        gpus = []
        for line in result.stdout.strip().split('\n'):
            if line:
                idx, util, mem_used, mem_total = line.split(',')
                gpus.append({
                    'index': int(idx.strip()),
                    'utilization': int(util.strip()),
                    'memory_used': int(mem_used.strip()),
                    'memory_total': int(mem_total.strip())
                })

        return gpus

    except Exception as e:
        print(f"Warning: Could not get GPU utilization: {e}")
        return []


def select_best_gpu() -> int:
    """
    Select GPU with most free memory.

    Returns:
        Index of best GPU

    Example:
        >>> best_gpu = select_best_gpu()
        >>> device = torch.device(f'cuda:{best_gpu}')
    """
    if not torch.cuda.is_available():
        return -1

    gpus = get_gpu_utilization()
    if not gpus:
        return 0

    # Select GPU with most free memory
    best_gpu = max(gpus, key=lambda x: x['memory_total'] - x['memory_used'])
    return best_gpu['index']


class GPUMemoryTracker:
    """
    Context manager for tracking GPU memory usage.

    Example:
        >>> with GPUMemoryTracker() as tracker:
        ...     model = load_large_model()
        ...     output = model(input)
        >>> print(f"Peak memory: {tracker.peak_memory:.2f} GB")
    """

    def __init__(self, device: Optional[int] = None):
        self.device = device if device is not None else torch.cuda.current_device()
        self.start_memory = 0
        self.peak_memory = 0

    def __enter__(self):
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(self.device)
            self.start_memory = torch.cuda.memory_allocated(self.device) / 1024**3
        return self

    def __exit__(self, *args):
        if torch.cuda.is_available():
            self.peak_memory = torch.cuda.max_memory_allocated(self.device) / 1024**3

    def get_memory_usage(self) -> Dict[str, float]:
        """Get memory usage statistics."""
        return {
            'start': self.start_memory,
            'peak': self.peak_memory,
            'used': self.peak_memory - self.start_memory
        }
