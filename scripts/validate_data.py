#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from result_aware.config import DEFAULT_CONFIG_DIR, load_experiment_config, resolve_dataset_paths
from result_aware.data import load_records, validate_records, validate_split_disjointness
from result_aware.utils import write_json


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=str(DEFAULT_CONFIG_DIR / "experiment.yaml"))
    p.add_argument("--dataset", choices=["dbpedia", "wikidata"], required=True)
    p.add_argument("--output", default=None)
    args = p.parse_args()

    exp = load_experiment_config(args.config)
    paths = resolve_dataset_paths(exp, args.dataset, base_dir=Path(args.config).resolve().parents[1])
    records = {split: load_records(path) for split, path in paths.items()}
    report = {
        split: validate_records(recs) for split, recs in records.items()
    }
    report["split_disjointness"] = validate_split_disjointness(
        records["train"], records["valid"], records["test"]
    )
    report["paths"] = {k: str(v) for k, v in paths.items()}
    print(report)
    if args.output:
        write_json(args.output, report)


if __name__ == "__main__":
    main()
