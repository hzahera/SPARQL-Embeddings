from __future__ import annotations

import json
import math
import os
import random
import re
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch


def normalize_query(query: str) -> str:
    """Normalization used only for identity/dedup/leakage checks."""
    return re.sub(r"\s+", " ", query.strip())


def set_global_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Deterministic kernels can reduce speed; use deterministic algorithms where available.
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def read_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str | Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, sort_keys=False)


def nested_get(mapping: dict[str, Any], dotted_key: str) -> Any:
    cur: Any = mapping
    for part in dotted_key.split("."):
        if not isinstance(cur, dict) or part not in cur:
            raise KeyError(f"Missing key {dotted_key!r}; failed at {part!r}")
        cur = cur[part]
    return cur


def flatten_dict(d: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            out.update(flatten_dict(v, key))
        else:
            out[key] = v
    return out


def mean_std(values: Iterable[float]) -> tuple[float, float]:
    vals = np.asarray(list(values), dtype=float)
    if vals.size == 0:
        return math.nan, math.nan
    if vals.size == 1:
        return float(vals[0]), 0.0
    return float(vals.mean()), float(vals.std(ddof=1))
