"""Runtime helpers for seeds, devices, directories, and JSON serialization."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and Torch RNGs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def resolve_device(device: str | None = None) -> torch.device:
    """Choose an execution device, preferring MPS on Apple Silicon when available."""
    mps_available = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    if device is not None:
        if device in {"metal", "mps"}:
            return torch.device("mps" if mps_available else "cpu")
        if device == "cuda":
            if torch.cuda.is_available():
                return torch.device("cuda")
            if mps_available:
                return torch.device("mps")
            return torch.device("cpu")
        return torch.device(device)
    if mps_available:
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def ensure_parent_dir(path: str | Path) -> None:
    """Create the parent directory for an output path if needed."""
    Path(path).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)


def to_serializable(value: Any) -> Any:
    """Convert tensors, arrays, and paths into JSON-friendly values."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        if value.ndim == 0:
            return value.item()
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): to_serializable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_serializable(item) for item in value]
    return value


def write_json(path: str | Path, payload: Any) -> None:
    """Write a JSON payload with stable formatting."""
    ensure_parent_dir(path)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(to_serializable(payload), handle, indent=2, sort_keys=True)
        handle.write("\n")
