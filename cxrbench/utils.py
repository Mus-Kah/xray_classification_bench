from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(require_cuda: bool = True) -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if require_cuda:
        raise RuntimeError(
            "CUDA is not available. Install a CUDA build of PyTorch, for example:\n"
            "  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124\n"
            "Use --cpu only for lightweight debugging."
        )
    return torch.device("cpu")


def describe_device(device: torch.device) -> str:
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        return f"CUDA {torch.version.cuda} | {props.name} | {props.total_memory / 2**30:.1f} GB"
    return "CPU"


def bootstrap_ci(
    y_true: np.ndarray, y_pred: np.ndarray, n_bootstrap: int = 2000, seed: int = 0
) -> tuple[float, float]:
    rng = np.random.RandomState(seed)
    n_samples = len(y_true)
    indices = rng.randint(0, n_samples, (n_bootstrap, n_samples))
    accuracies = (y_pred[indices] == y_true[indices]).mean(axis=1)
    low, high = np.percentile(accuracies, [2.5, 97.5])
    return float(low), float(high)


def save_json(path: str | Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=float)


def json_safe_records(table: pd.DataFrame) -> list[dict]:
    records = []
    for row in table.to_dict(orient="records"):
        records.append(
            {
                key: None if isinstance(value, float) and np.isnan(value) else value
                for key, value in row.items()
            }
        )
    return records


def short_name(model_name: str) -> str:
    return model_name.split(".")[0]
