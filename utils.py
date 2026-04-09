"""
Small utilities shared across scripts.
"""

import torch


def get_device() -> torch.device:
    """
    Prefer CUDA if available, otherwise Apple Silicon MPS, otherwise CPU.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

