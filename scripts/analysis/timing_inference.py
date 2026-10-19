#!/usr/bin/env python3
"""
Inference cost for the paper (Section 3.2): encoding latency per query and nearest-neighbor
search time over the cache, for every encoder, plus the hardware used.

For each encoder, the fine-tuned checkpoint of one seed is loaded and the test queries of one graph
are encoded one at a time (batch size 1, as for an incoming query). Search time is measured for one
query vector against K cached vectors (K = 128, the cache size of RQ3, and all test queries).

Run from the repository root:
  export PYTHONPATH="$PWD/src:$PYTHONPATH"
  python scripts/analysis/timing_inference.py --dataset wikidata
Output: printed table + results/analysis/timing_inference_<dataset>.json
"""
import argparse
import json
import platform
import statistics as st
import subprocess
import time
from pathlib import Path

import torch

from result_aware.config import load_model_config
from result_aware.data import load_records, unique_queries
from result_aware.model import HFEmbeddingModel

MODELS = ["mpnet", "e5-base-v2", "bge-base", "gte-base", "qwen3-embedding",
          "gemma", "codebert", "graphcodebert", "qwen2.5-coder", "spbert"]


def sync(device):
    if device.startswith("cuda"):
        torch.cuda.synchronize()


def hardware():
    info = {"python": platform.python_version(), "torch": torch.__version__,
            "cpu": platform.processor() or platform.machine()}
    try:
        info["cpu"] = next(l.split(":", 1)[1].strip() for l in open("/proc/cpuinfo") if l.startswith("model name"))
    except Exception:
        pass
    if torch.cuda.is_available():
        info["gpu"] = torch.cuda.get_device_name(0)
        info["cuda"] = torch.version.cuda
    try:
        info["nvidia_smi"] = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
                                             "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
    except Exception:
        pass
    return info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="wikidata", choices=["dbpedia", "wikidata"])
    ap.add_argument("--models", nargs="+", default=MODELS)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n", type=int, default=200, help="number of timed queries")
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    records = load_records(f"datasets/{args.dataset}-splits/{args.dataset}-test.json")
    texts = unique_queries(records)
    timed = texts[: args.n]
    hw = hardware()
    print("Hardware:", json.dumps(hw, indent=1))
    print(f"{args.dataset}: {len(texts)} test queries, timing {len(timed)} (batch size 1)\n")

    out = {"dataset": args.dataset, "hardware": hw, "timed_queries": len(timed), "models": {}}
    print(f"{'model':16s} {'encode median ms':>17s} {'p90 ms':>8s} {'search K=128 ms':>16s} {'search all ms':>14s}")
    for key in args.models:
        ckpt = Path("results/raw") / args.dataset / key / "tuned" / f"seed_{args.seed}" / "best_model"
        if not ckpt.exists():
            print(f"{key:16s} no checkpoint at {ckpt}, skipped")
            continue
        try:
            model = HFEmbeddingModel(load_model_config(key), device=args.device, checkpoint_path=ckpt)
        except Exception as e:
            print(f"{key:16s} could not load ({type(e).__name__}: {e}), skipped")
            continue
        model.eval()
        with torch.no_grad():
            for q in texts[: args.warmup]:
                model.encode([q], batch_size=1)
            enc = []
            for q in timed:
                sync(args.device); t0 = time.perf_counter()
                model.encode([q], batch_size=1)
                sync(args.device); enc.append((time.perf_counter() - t0) * 1000)

            allv = model.encode(texts, batch_size=64).to(args.device)
            qv = allv[0]
            search = {}
            for name, cache in (("k128", allv[:128]), ("all", allv)):
                for _ in range(20):
                    (cache @ qv).argmax()
                ts = []
                for _ in range(200):
                    sync(args.device); t0 = time.perf_counter()
                    int((cache @ qv).argmax())
                    sync(args.device); ts.append((time.perf_counter() - t0) * 1000)
                search[name] = st.median(ts)

        p90 = sorted(enc)[int(0.9 * len(enc)) - 1]
        out["models"][key] = {"encode_ms_median": st.median(enc), "encode_ms_p90": p90,
                              "search_ms_k128": search["k128"], "search_ms_all": search["all"],
                              "cache_all_size": len(texts)}
        print(f"{key:16s} {st.median(enc):17.2f} {p90:8.2f} {search['k128']:16.3f} {search['all']:14.3f}")
        del model
        torch.cuda.empty_cache()

    dest = Path("results/analysis") / f"timing_inference_{args.dataset}.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(dest, "w"), indent=2)
    print(f"\nSaved {dest}")


if __name__ == "__main__":
    main()
