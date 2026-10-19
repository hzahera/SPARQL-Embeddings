#!/usr/bin/env python3
"""Compare every per-anchor file (protocol B) with the stored evaluation_test.json of its run.
Run from the repository root:  python scripts/analysis/check_per_anchor.py"""
import json
from pathlib import Path

bad = 0
for ds in ("dbpedia", "wikidata"):
    for mdir in sorted(p for p in Path(f"results/per_anchor/{ds}").iterdir() if p.is_dir()):
        for f in sorted(mdir.glob("*.json")):
            if f.name.endswith("_full.json"):
                continue
            tag = f.stem
            ev = (Path("results/raw") / ds / mdir.name /
                  ("zero_shot" if tag == "zero_shot" else f"tuned/{tag}") / "evaluation_test.json")
            if not ev.exists():
                print(f"{ds:9s} {mdir.name:16s} {tag:10s} no evaluation_test.json"); continue
            ref = json.load(open(ev))["ranking"]
            sc = json.load(open(f))
            diffs = {m: sum(v[m] for v in sc.values()) / len(sc) - ref[m] for m in ("hits@1", "mrr@5", "ndcg@5")}
            if max(abs(d) for d in diffs.values()) > 1e-6:
                bad += 1
                print(f"{ds:9s} {mdir.name:16s} {tag:10s} MISMATCH " + "  ".join(f"{m} {d:+.4f}" for m, d in diffs.items()))
print("all per-anchor files match their evaluation_test.json" if bad == 0 else f"{bad} file(s) do not match")
