#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import shutil
from pathlib import Path

import torch

from result_aware.cache_eval import evaluate_semantic_cache
from result_aware.config import (
    DEFAULT_CONFIG_DIR,
    load_experiment_config,
    load_model_config,
    load_registry,
    resolve_dataset_paths,
)
from result_aware.data import load_records, validate_records, validate_split_disjointness
from result_aware.evaluation import evaluate_records
from result_aware.model import HFEmbeddingModel
from result_aware.results import write_aggregated_summary
from result_aware.trainer import train_one_seed
from result_aware.utils import ensure_dir, write_json


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run zero-shot, tuned, retrieval, alignment and cache experiments.")
    p.add_argument("--config", default=str(DEFAULT_CONFIG_DIR / "experiment.yaml"))
    p.add_argument("--config-dir", default=str(DEFAULT_CONFIG_DIR))
    p.add_argument("--models", nargs="+", default=["all"], help="Registry keys or 'all'.")
    p.add_argument("--datasets", nargs="+", default=["dbpedia", "wikidata"])
    p.add_argument("--seeds", nargs="+", type=int, default=None)
    p.add_argument("--device", default="cuda")
    p.add_argument("--force", action="store_true")
    p.add_argument("--skip-zero-shot", action="store_true")
    p.add_argument("--skip-training", action="store_true")
    p.add_argument("--skip-cache", action="store_true")
    p.add_argument("--batch-size", type=int, default=None, help="Optional runtime override.")
    p.add_argument("--grad-accum", type=int, default=None, help="Optional runtime override.")
    p.add_argument("--encode-batch-size", type=int, default=None)
    return p.parse_args()


def release(model=None) -> None:
    if model is not None:
        del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main() -> None:
    args = parse_args()
    exp = load_experiment_config(args.config)
    registry = load_registry(args.config_dir)
    all_models = list(registry["models"].keys())
    models = all_models if args.models == ["all"] else args.models
    for m in models:
        if m not in registry["models"]:
            raise SystemExit(f"Unknown model {m!r}; choose from {all_models}")
    datasets = list(exp["datasets"].keys()) if args.datasets == ["all"] else args.datasets
    seeds = args.seeds or list(exp.get("seeds", [42, 43, 44, 45, 46]))

    training_cfg = dict(exp["training"])
    eval_cfg = dict(exp["evaluation"])
    cache_cfg = dict(exp.get("semantic_cache", {}))
    if args.batch_size is not None:
        training_cfg["batch_size"] = args.batch_size
    if args.grad_accum is not None:
        training_cfg["gradient_accumulation_steps"] = args.grad_accum
    if args.encode_batch_size is not None:
        eval_cfg["encode_batch_size"] = args.encode_batch_size
        cache_cfg["encode_batch_size"] = args.encode_batch_size

    results_dir = Path(exp.get("output_dir", "results"))
    if not results_dir.is_absolute():
        results_dir = (Path.cwd() / results_dir).resolve()
    ensure_dir(results_dir)

    for dataset_name in datasets:
        paths = resolve_dataset_paths(exp, dataset_name, base_dir=Path(args.config).resolve().parents[1])
        print(f"\n=== Loading {dataset_name} ===")
        train_records = load_records(paths["train"])
        valid_records = load_records(paths["valid"])
        test_records = load_records(paths["test"])

        data_audit = {
            "train": validate_records(train_records),
            "valid": validate_records(valid_records),
            "test": validate_records(test_records),
            "split_disjointness": validate_split_disjointness(train_records, valid_records, test_records),
            "paths": {k: str(v) for k, v in paths.items()},
        }
        audit_path = results_dir / "raw" / dataset_name / "data_audit.json"
        write_json(audit_path, data_audit)
        if not all(data_audit[s]["passed"] for s in ("train", "valid", "test")):
            raise RuntimeError(f"Dataset validation failed; see {audit_path}")
        if not data_audit["split_disjointness"]["passed"]:
            raise RuntimeError(f"Cross-split query leakage detected; see {audit_path}")

        for model_key in models:
            print(f"\n========== {dataset_name} / {model_key} ==========")
            model_cfg = load_model_config(model_key, args.config_dir)
            base = results_dir / "raw" / dataset_name / model_key
            ensure_dir(base)
            write_json(
                base / "experiment_manifest.json",
                {
                    "dataset": dataset_name,
                    "model_key": model_key,
                    "model_config": model_cfg,
                    "training_config": training_cfg,
                    "evaluation_config": eval_cfg,
                    "semantic_cache_config": cache_cfg,
                    "seeds": seeds,
                    "data_paths": {k: str(v) for k, v in paths.items()},
                },
            )

            if not args.skip_zero_shot:
                zs_dir = base / "zero_shot"
                if args.force and zs_dir.exists():
                    shutil.rmtree(zs_dir)
                ensure_dir(zs_dir)
                eval_test_path = zs_dir / "evaluation_test.json"
                eval_valid_path = zs_dir / "evaluation_validation.json"
                cache_path = zs_dir / "cache.json"
                need_model = (
                    not eval_test_path.exists()
                    or not eval_valid_path.exists()
                    or (not args.skip_cache and not cache_path.exists())
                )
                if need_model:
                    print("Loading zero-shot model...")
                    model = HFEmbeddingModel(model_cfg, device=args.device)
                    if not eval_valid_path.exists():
                        write_json(
                            eval_valid_path,
                            evaluate_records(model, valid_records, eval_cfg, "validation"),
                        )
                    if not eval_test_path.exists():
                        write_json(
                            eval_test_path,
                            evaluate_records(model, test_records, eval_cfg, "test"),
                        )
                    if not args.skip_cache and cache_cfg.get("enabled", True) and not cache_path.exists():
                        write_json(
                            cache_path,
                            evaluate_semantic_cache(
                                model, valid_records, test_records, eval_cfg, cache_cfg
                            ),
                        )
                    release(model)
                else:
                    print("Zero-shot outputs already exist; skipping.")

            if args.skip_training:
                continue

            for seed in seeds:
                seed_dir = base / "tuned" / f"seed_{seed}"
                if args.force and seed_dir.exists():
                    shutil.rmtree(seed_dir)
                ensure_dir(seed_dir)
                checkpoint = seed_dir / "best_model"
                training_summary = seed_dir / "training_summary.json"
                if not checkpoint.exists() or not training_summary.exists():
                    print(f"Training seed {seed}...")
                    train_one_seed(
                        model_cfg=model_cfg,
                        train_records=train_records,
                        valid_records=valid_records,
                        training_cfg=training_cfg,
                        eval_cfg=eval_cfg,
                        seed=seed,
                        output_dir=seed_dir,
                        device=args.device,
                    )
                    release()
                else:
                    print(f"Seed {seed}: checkpoint already exists; skipping training.")

                eval_valid_path = seed_dir / "evaluation_validation.json"
                eval_test_path = seed_dir / "evaluation_test.json"
                cache_path = seed_dir / "cache.json"
                need_tuned_model = (
                    not eval_valid_path.exists()
                    or not eval_test_path.exists()
                    or (not args.skip_cache and cache_cfg.get("enabled", True) and not cache_path.exists())
                )
                if need_tuned_model:
                    print(f"Evaluating tuned seed {seed} from best checkpoint...")
                    tuned = HFEmbeddingModel(model_cfg, device=args.device, checkpoint_path=checkpoint)
                    if not eval_valid_path.exists():
                        write_json(
                            eval_valid_path,
                            evaluate_records(tuned, valid_records, eval_cfg, "validation"),
                        )
                    if not eval_test_path.exists():
                        write_json(
                            eval_test_path,
                            evaluate_records(tuned, test_records, eval_cfg, "test"),
                        )
                    if not args.skip_cache and cache_cfg.get("enabled", True) and not cache_path.exists():
                        write_json(
                            cache_path,
                            evaluate_semantic_cache(
                                tuned, valid_records, test_records, eval_cfg, cache_cfg
                            ),
                        )
                    release(tuned)

    summary_path = write_aggregated_summary(results_dir, models, datasets)
    print(f"\nAggregated summary: {summary_path}")


if __name__ == "__main__":
    main()
