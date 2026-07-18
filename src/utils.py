"""General reproducibility, device, directory, and checkpoint helpers."""

from __future__ import annotations

from pathlib import Path
import os
import random

import numpy as np


def set_global_seed(seed: int) -> None:
    """Set Python, NumPy, and PyTorch random seeds."""

    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        mps_backend = getattr(torch.backends, "mps", None)
        if mps_backend is not None and mps_backend.is_available():
            torch.mps.manual_seed(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except TypeError:
            torch.use_deterministic_algorithms(True)
    except ModuleNotFoundError:
        pass


def ensure_dir(path: str | Path) -> Path:
    """Create a directory if needed and return it as a Path."""

    path = Path(path).resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_device():
    """Return CUDA, then Apple MPS, then CPU."""

    import torch

    if torch.cuda.is_available():
        return torch.device("cuda")
    mps_backend = getattr(torch.backends, "mps", None)
    if mps_backend is not None and mps_backend.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def save_checkpoint(
    path: str | Path,
    model,
    optimizer=None,
    epoch: int | None = None,
    metrics: dict | None = None,
    extra: dict | None = None,
) -> Path:
    """Save a PyTorch checkpoint."""

    import torch

    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state_dict": model.state_dict(),
        "epoch": epoch,
        "metrics": metrics or {},
        "extra": extra or {},
    }
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    torch.save(payload, path)
    return path


def load_checkpoint(path: str | Path, model, optimizer=None, map_location=None):
    """Load a PyTorch checkpoint into a model and optional optimizer."""

    import torch

    checkpoint = torch.load(Path(path).resolve(), map_location=map_location)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint


def count_trainable_parameters(model) -> int:
    """Count trainable model parameters."""

    return int(sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad))
