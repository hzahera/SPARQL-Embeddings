from __future__ import annotations

from pathlib import Path
from typing import Any

from .utils import mean_std, read_json, write_json


# ---------------------------------------------------------------------------
# Generic numeric aggregation
# ---------------------------------------------------------------------------


def _numeric_leaf_paths(
    obj: Any,
    prefix: str = "",
) -> dict[str, float]:
    out: dict[str, float] = {}

    if isinstance(obj, dict):
        for key, value in obj.items():
            child = (
                f"{prefix}.{key}"
                if prefix
                else str(key)
            )
            out.update(
                _numeric_leaf_paths(
                    value,
                    child,
                )
            )

    elif (
        isinstance(obj, (int, float))
        and not isinstance(obj, bool)
    ):
        out[prefix] = float(obj)

    return out


def _aggregate_objects(
    objects: list[dict[str, Any]],
) -> dict[str, dict[str, float]]:
    """
    Existing generic aggregation used for retrieval/alignment evaluation.

    Each numeric leaf is aggregated independently over training seeds.
    """
    paths: dict[str, list[float]] = {}

    for obj in objects:
        for key, value in _numeric_leaf_paths(obj).items():
            paths.setdefault(key, []).append(value)

    result: dict[str, dict[str, float]] = {}

    for key, values in paths.items():
        mean, std = mean_std(values)

        result[key] = {
            "mean": mean,
            "std": std,
            "n": len(values),
        }

    return result


# ---------------------------------------------------------------------------
# Online-cache aggregation
# ---------------------------------------------------------------------------


def _aggregate_metric_rows(
    rows: list[dict[str, Any]],
) -> dict[str, dict[str, float]]:
    """
    Aggregate top-level numeric cache metrics.

    Unlike _aggregate_objects(), this intentionally does not recursively
    aggregate nested metadata and does not treat stream_seed as a metric.
    """
    values_by_key: dict[str, list[float]] = {}

    for row in rows:
        for key, value in row.items():
            if key in {
                "stream_seed",
                "capacity",
            }:
                continue

            if isinstance(value, bool):
                continue

            if isinstance(value, (int, float)):
                values_by_key.setdefault(
                    key,
                    [],
                ).append(float(value))

    out: dict[str, dict[str, float]] = {}

    for key, values in values_by_key.items():
        if not values:
            continue

        mean, std = mean_std(values)

        out[key] = {
            "mean": mean,
            "std": std,
            "n": len(values),
        }

    return out


def _is_online_cache_result(
    cache: dict[str, Any],
) -> bool:
    definition = cache.get("definition", {})

    return (
        isinstance(definition, dict)
        and definition.get("mode")
        == "online bounded semantic cache"
        and isinstance(
            cache.get("capacities"),
            dict,
        )
    )


def _aggregate_online_caches(
    caches: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Aggregate tuned online-cache experiments.

    Every tuned training seed already contains multiple stream-order runs.
    We combine all per-stream observations across training seeds.

    Example:
        5 model-training seeds
        x 5 stream-order seeds
        = 25 observations per method/capacity

    This captures variation from both:
      - fine-tuning seed
      - workload ordering

    without recursively averaging already-computed std values.
    """
    if not caches:
        return {}

    first = caches[0]

    result: dict[str, Any] = {
        "definition": first.get(
            "definition",
            {},
        ),
        "training_seed_runs": len(caches),
        "primary_capacity": first.get(
            "definition",
            {},
        ).get(
            "primary_capacity"
        ),
    }

    # ---------------------------------------------------------------
    # Aggregate selected threshold across model-training seeds.
    # ---------------------------------------------------------------

    thresholds = [
        float(cache["selected_threshold"])
        for cache in caches
        if isinstance(
            cache.get("selected_threshold"),
            (int, float),
        )
    ]

    if thresholds:
        mean, std = mean_std(thresholds)

        result["selected_threshold"] = {
            "mean": mean,
            "std": std,
            "n": len(thresholds),
        }

    # ---------------------------------------------------------------
    # Capacity-specific test results.
    # ---------------------------------------------------------------

    capacity_keys: set[str] = set()

    for cache in caches:
        capacities = cache.get(
            "capacities",
            {},
        )

        if isinstance(capacities, dict):
            capacity_keys.update(
                capacities.keys()
            )

    result["capacities"] = {}

    for capacity in sorted(
        capacity_keys,
        key=lambda x: int(x),
    ):
        capacity_result: dict[str, Any] = {
            "test": {}
        }

        method_names: set[str] = set()

        for cache in caches:
            test_obj = (
                cache.get(
                    "capacities",
                    {},
                )
                .get(
                    capacity,
                    {},
                )
                .get(
                    "test",
                    {},
                )
            )

            if isinstance(test_obj, dict):
                method_names.update(
                    test_obj.keys()
                )

        for method in sorted(method_names):
            all_rows: list[dict[str, Any]] = []

            for cache in caches:
                method_obj = (
                    cache.get(
                        "capacities",
                        {},
                    )
                    .get(
                        capacity,
                        {},
                    )
                    .get(
                        "test",
                        {},
                    )
                    .get(
                        method,
                        {},
                    )
                )

                if not isinstance(
                    method_obj,
                    dict,
                ):
                    continue

                per_stream = method_obj.get(
                    "per_stream",
                    [],
                )

                if isinstance(
                    per_stream,
                    list,
                ):
                    all_rows.extend(
                        row
                        for row in per_stream
                        if isinstance(
                            row,
                            dict,
                        )
                    )

            capacity_result["test"][
                method
            ] = {
                "aggregate": (
                    _aggregate_metric_rows(
                        all_rows
                    )
                ),
                "n_observations": len(
                    all_rows
                ),
            }

        result["capacities"][
            capacity
        ] = capacity_result

    return result


# ---------------------------------------------------------------------------
# Main aggregation
# ---------------------------------------------------------------------------


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
            base = (
                results_dir
                / "raw"
                / dataset
                / model
            )

            entry: dict[str, Any] = {}

            # -------------------------------------------------------
            # Zero-shot
            # -------------------------------------------------------

            zs_eval = (
                base
                / "zero_shot"
                / "evaluation_test.json"
            )

            zs_cache = (
                base
                / "zero_shot"
                / "cache.json"
            )

            if zs_eval.exists():
                entry.setdefault(
                    "zero_shot",
                    {},
                )["evaluation"] = read_json(
                    zs_eval
                )

            if zs_cache.exists():
                entry.setdefault(
                    "zero_shot",
                    {},
                )["cache"] = read_json(
                    zs_cache
                )

            # -------------------------------------------------------
            # Fine-tuned seeds
            # -------------------------------------------------------

            tuned_root = (
                base
                / "tuned"
            )

            evals: list[
                dict[str, Any]
            ] = []

            caches: list[
                dict[str, Any]
            ] = []

            eval_seed_ids: list[int] = []
            cache_seed_ids: list[int] = []

            if tuned_root.exists():
                for seed_dir in sorted(
                    tuned_root.glob(
                        "seed_*"
                    )
                ):
                    try:
                        seed = int(
                            seed_dir.name.split(
                                "_",
                                1,
                            )[1]
                        )
                    except Exception:
                        continue

                    eval_path = (
                        seed_dir
                        / "evaluation_test.json"
                    )

                    cache_path = (
                        seed_dir
                        / "cache.json"
                    )

                    if eval_path.exists():
                        evals.append(
                            read_json(
                                eval_path
                            )
                        )
                        eval_seed_ids.append(
                            seed
                        )

                    if cache_path.exists():
                        caches.append(
                            read_json(
                                cache_path
                            )
                        )
                        cache_seed_ids.append(
                            seed
                        )

            # -------------------------------------------------------
            # Fine-tuned evaluation aggregation
            # -------------------------------------------------------

            if evals or caches:
                tuned_entry: dict[
                    str,
                    Any,
                ] = {}

                all_seed_ids = sorted(
                    set(
                        eval_seed_ids
                        + cache_seed_ids
                    )
                )

                tuned_entry[
                    "seeds"
                ] = all_seed_ids

                if evals:
                    tuned_entry[
                        "evaluation"
                    ] = _aggregate_objects(
                        evals
                    )
                else:
                    tuned_entry[
                        "evaluation"
                    ] = {}

                # ---------------------------------------------------
                # Cache aggregation
                # ---------------------------------------------------

                if caches:
                    if all(
                        _is_online_cache_result(
                            cache
                        )
                        for cache in caches
                    ):
                        tuned_entry[
                            "cache"
                        ] = _aggregate_online_caches(
                            caches
                        )
                    else:
                        # Backward compatibility with old
                        # candidate-cache result files.
                        tuned_entry[
                            "cache"
                        ] = _aggregate_objects(
                            caches
                        )
                else:
                    tuned_entry[
                        "cache"
                    ] = {}

                entry[
                    "tuned"
                ] = tuned_entry

            summary[
                dataset
            ][model] = entry

    return summary


def write_aggregated_summary(
    results_dir: str | Path,
    model_keys: list[str],
    datasets: list[str],
) -> Path:
    results_dir = Path(results_dir)

    summary = aggregate_results(
        results_dir,
        model_keys,
        datasets,
    )

    out = (
        results_dir
        / "aggregated"
        / "summary.json"
    )

    write_json(
        out,
        summary,
    )

    return out