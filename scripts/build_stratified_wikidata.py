#!/usr/bin/env python3
"""Build the stratified Wikidata RQ1 test set (500 natural anchors).

A single set spanning the natural lexical-similarity range, reported PER cos-bin
(<0.3 / 0.3-0.5 / 0.5-0.8 / >=0.8) exactly like the DBpedia stratified table.
The aggregate is realistic; the <0.3 stratum isolates the regime where only a
result-aware / structural model can retrieve. No anchors are hand-picked for
difficulty and none are padded with filler -- the bins fall out of the natural
distribution, so there is no cherry-pick or dilution artifact.

Bins are NOT stored here: they are computed at eval time from the label-expanded
anchor->positive cosine (`eval_rq1_stratified.py --stratify`), so the binning
uses the exact same expanded text the text baselines read and needs no label map
at build time.

Spec
----
* Source     : held-out Wikidata TEST split only (no train/val -> no leakage).
* Anchor     : any test anchor with >= 1 result-equivalent positive.
* Positive   : answer_jaccard > 0.8, capped at CAP/anchor (strongest first).
* Negatives  : the anchor's shipped negatives (answer_jaccard = 0), up to CAP.
* N_ANCHORS  : 500, sampled deterministically (seed 42) from the qualifying pool.
* No lexical or result-size filter -- the natural distribution is the point.

Output: datasets/wikidata_testset_RQ1_stratified.json

    python scripts/build_stratified_wikidata.py
"""
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRIP = ROOT / "datasets" / "wikidata-triplets-dataset.json"
SPLITS = ROOT / "datasets" / "splits_wikidata.json"
OUT = ROOT / "datasets" / "wikidata_testset_RQ1_stratified.json"

JAC_MIN = 0.80
CAP = 20
N_ANCHORS = 500
SEED = 42


def main():
    trip = json.load(open(TRIP))
    spl = json.load(open(SPLITS))
    test = [trip[i] for i in spl["test_idx"]]

    pool = []
    for e in test:
        pos = sorted((p for p in e["positive_queries"]
                      if p.get("answer_jaccard", 0) > JAC_MIN),
                     key=lambda p: -p["answer_jaccard"])[:CAP]
        if not pos:
            continue
        pool.append({
            "anchor_query": e["anchor_query"],
            "positive_queries": [
                {"query": p["query"], "answer_jaccard": round(float(p["answer_jaccard"]), 4)}
                for p in pos],
            "negative_queries": [
                {"query": n["query"],
                 "answer_jaccard": round(float(n.get("answer_jaccard", 0.0)), 4)}
                for n in e["negative_queries"][:CAP]],
        })

    if len(pool) < N_ANCHORS:
        print(f"WARNING: only {len(pool)} qualifying anchors (< {N_ANCHORS}); "
              "using all.")
    random.Random(SEED).shuffle(pool)
    out = pool[:N_ANCHORS]
    json.dump(out, open(OUT, "w"), ensure_ascii=False, indent=1)

    import numpy as np
    npos = [len(r["positive_queries"]) for r in out]
    nneg = [len(r["negative_queries"]) for r in out]
    corpus = set()
    for r in out:
        corpus.add(" ".join(r["anchor_query"].split()))
        corpus.update(" ".join(p["query"].split()) for p in r["positive_queries"])
        corpus.update(" ".join(n["query"].split()) for n in r["negative_queries"])
    print(f"wrote {len(out)} anchors to {OUT}")
    print(f"  qualifying pool: {len(pool)}  ->  sampled: {len(out)} (seed {SEED})")
    print(f"  positives/anchor: min {min(npos)} mean {np.mean(npos):.2f} max {max(npos)} "
          f"(total {sum(npos)})")
    print(f"  negatives/anchor: mean {np.mean(nneg):.2f}")
    print(f"  positive jaccard: all > {JAC_MIN}")
    print(f"  unique corpus queries: {len(corpus)}")
    print(f"  bins are computed at eval time: "
          f"eval_rq1_stratified.py --split {OUT.name} --stratify")


if __name__ == "__main__":
    main()
