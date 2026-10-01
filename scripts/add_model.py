#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from result_aware.config import DEFAULT_CONFIG_DIR, load_registry


def main() -> None:
    p = argparse.ArgumentParser(description="Register another Hugging Face transformer without changing benchmark code.")
    p.add_argument("--key", required=True, help="Short registry key, e.g. modernbert")
    p.add_argument("--hf-id", required=True, help="Hugging Face model id or local path")
    p.add_argument("--display-name", required=True)
    p.add_argument("--category", choices=["text", "code", "general", "sparql"], default="text")
    p.add_argument("--architecture", choices=["encoder", "decoder"], required=True)
    p.add_argument("--pooling", choices=["mean", "cls", "last_token"], required=True)
    p.add_argument("--prefix", default="")
    p.add_argument("--padding-side", choices=["left", "right"], default="right")
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--trust-remote-code", action="store_true")
    p.add_argument("--citation", default=None)
    p.add_argument("--config-dir", default=str(DEFAULT_CONFIG_DIR))
    args = p.parse_args()

    config_dir = Path(args.config_dir)
    registry_path = config_dir / "model_registry.yaml"
    registry = load_registry(config_dir)
    if args.key in registry["models"]:
        raise SystemExit(f"Model key {args.key!r} already exists")

    model_rel = f"models/{args.key}.yaml"
    model_path = config_dir / model_rel
    model_cfg = {
        "name": args.key,
        "model_name_or_path": args.hf_id,
        "architecture": args.architecture,
        "pooling": args.pooling,
        "prefix": args.prefix,
        "max_length": args.max_length,
        "padding_side": args.padding_side,
        "trust_remote_code": bool(args.trust_remote_code),
        "torch_dtype": "float32",
        "input_transform": "identity",
    }
    model_path.parent.mkdir(parents=True, exist_ok=True)
    with open(model_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(model_cfg, f, sort_keys=False)

    registry["models"][args.key] = {
        "config": model_rel,
        "display_name": args.display_name,
        "category": args.category,
        "citation": args.citation,
    }
    with open(registry_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(registry, f, sort_keys=False, allow_unicode=True)

    print(f"Registered {args.key} in {registry_path}")
    print(f"Created {model_path}")


if __name__ == "__main__":
    main()
