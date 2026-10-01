#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from result_aware.config import DEFAULT_CONFIG_DIR, load_experiment_config, load_registry
from result_aware.results import write_aggregated_summary


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=str(DEFAULT_CONFIG_DIR / "experiment.yaml"))
    p.add_argument("--config-dir", default=str(DEFAULT_CONFIG_DIR))
    p.add_argument("--models", nargs="+", default=["all"])
    p.add_argument("--datasets", nargs="+", default=["all"])
    args = p.parse_args()

    exp = load_experiment_config(args.config)
    registry = load_registry(args.config_dir)
    models = list(registry["models"]) if args.models == ["all"] else args.models
    datasets = list(exp["datasets"]) if args.datasets == ["all"] else args.datasets
    results_dir = Path(exp.get("output_dir", "results")).resolve()
    out = write_aggregated_summary(results_dir, models, datasets)
    print(out)


if __name__ == "__main__":
    main()
