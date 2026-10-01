from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_DIR = PROJECT_ROOT / "configs"


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = deepcopy(v)
    return out


def load_registry(config_dir: str | Path = DEFAULT_CONFIG_DIR) -> dict[str, Any]:
    return load_yaml(Path(config_dir) / "model_registry.yaml")


def load_experiment_config(path: str | Path | None = None) -> dict[str, Any]:
    return load_yaml(path or (DEFAULT_CONFIG_DIR / "experiment.yaml"))


def load_model_config(model_key: str, config_dir: str | Path = DEFAULT_CONFIG_DIR) -> dict[str, Any]:
    config_dir = Path(config_dir)
    registry = load_registry(config_dir)
    models = registry.get("models", {})
    if model_key not in models:
        available = ", ".join(models)
        raise KeyError(f"Unknown model {model_key!r}. Available: {available}")
    rel = models[model_key]["config"]
    cfg = load_yaml(config_dir / rel)
    cfg["registry"] = deepcopy(models[model_key])
    return cfg


def resolve_dataset_paths(exp_cfg: dict[str, Any], dataset: str, base_dir: str | Path | None = None) -> dict[str, Path]:
    if dataset not in exp_cfg["datasets"]:
        raise KeyError(f"Dataset {dataset!r} not defined in experiment config")
    base = Path(base_dir) if base_dir else PROJECT_ROOT
    result: dict[str, Path] = {}
    for split, raw in exp_cfg["datasets"][dataset].items():
        p = Path(raw)
        result[split] = p if p.is_absolute() else (base / p).resolve()
    return result
