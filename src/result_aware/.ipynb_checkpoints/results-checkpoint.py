from __future__ import annotations

from pathlib import Path
from typing import Any

from .utils import mean_std, read_json, write_json


def _numeric_leaf_paths(obj: Any, prefix: str = "") -> dict[str, float]:
    out: dict[str, float] = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            key = f"{prefix}.{k}" if prefix else str(k)
            out.update(_numeric_leaf_paths(v, key))
    elif isinstance(obj, (int, float)) and not isinstance(obj, bool):
        out[prefix] = float(obj)
    return out


def _aggregate_objects(objects: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    paths: dict[str, list[float]] = {}
    for obj in objects:
        for key, value in _numeric_leaf_paths(obj).items():
            paths.setdefault(key, []).append(value)
    result: dict[str, dict[str, float]] = {}
    for key, vals in paths.items():
        mean, std = mean_std(vals)
        result[key] = {"mean": mean, "std": std, "n": len(vals)}
    return result


def aggregate_results(
    results_dir: str | Path,
    model_keys: list[str],
    datasets: list[str],
) -> dict[str, Any]:
    results_dir = Path(results_dir)
    summary: dict[str, Any] = {}
    for dataset in datasets:
        summary[dataset] = {}
        for model in model_keys:
            base = results_dir / "raw" / dataset / model
            entry: dict[str, Any] = {}

            zs_eval = base / "zero_shot" / "evaluation_test.json"
            zs_cache = base / "zero_shot" / "cache.json"
            if zs_eval.exists():
                entry.setdefault("zero_shot", {})["evaluation"] = read_json(zs_eval)
            if zs_cache.exists():
                entry.setdefault("zero_shot", {})["cache"] = read_json(zs_cache)

            tuned_root = base / "tuned"
            evals: list[dict[str, Any]] = []
            caches: list[dict[str, Any]] = []
            seed_ids: list[int] = []
            if tuned_root.exists():
                for seed_dir in sorted(tuned_root.glob("seed_*")):
                    try:
                        seed = int(seed_dir.name.split("_", 1)[1])
                    except Exception:
                        continue
                    eval_path = seed_dir / "evaluation_test.json"
                    cache_path = seed_dir / "cache.json"
                    if eval_path.exists():
                        evals.append(read_json(eval_path))
                        seed_ids.append(seed)
                    if cache_path.exists():
                        caches.append(read_json(cache_path))

            if evals or caches:
                entry["tuned"] = {
                    "seeds": sorted(set(seed_ids)),
                    "evaluation": _aggregate_objects(evals) if evals else {},
                    "cache": _aggregate_objects(caches) if caches else {},
                }
            summary[dataset][model] = entry
    return summary


def write_aggregated_summary(
    results_dir: str | Path,
    model_keys: list[str],
    datasets: list[str],
) -> Path:
    results_dir = Path(results_dir)
    summary = aggregate_results(results_dir, model_keys, datasets)
    out = results_dir / "aggregated" / "summary.json"
    write_json(out, summary)
    return out
