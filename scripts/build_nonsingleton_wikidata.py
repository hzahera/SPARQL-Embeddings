#!/usr/bin/env python3
"""Build the non-singleton Wikidata RQ1 test set (substantive result-equivalence).

Wikidata's answer_jaccard is bimodal (0 or 1) because most queries return a
single entity: two singletons are either identical (jaccard 1) or disjoint
(jaccard 0), so `jaccard > 0.8` collapses to "same one answer" and never tests
partial overlap. This set removes that degeneracy: both the anchor and every
positive must return at least MIN_RESULT_SIZE (=5) rows, so a `jaccard > 0.8`
positive is a genuine many-element set intersection, not two identical singletons.
This also drops the COUNT(...) aggregate queries (which return exactly one row).

No lexical filter is applied -- this is the natural lexical distribution over
non-singleton, result-equivalent pairs.

Spec
----
* Source     : held-out Wikidata TEST + VALID splits (SPLIT_KEYS). Train is
               excluded -> no leakage: the trained models (SparqlCE, E5-FT) never
               saw these anchors as training targets. VALID was used only for
               checkpoint selection (mild optimism, disclosed) and is included to
               reach a usable anchor count -- the oracle covers too few TEST
               anchors alone for a large non-singleton set.
* Anchor     : result set >= MIN_RESULT_SIZE rows (non-singleton).
* Positive   : answer_jaccard > 0.8 (result-equivalent) AND result set >=
               MIN_RESULT_SIZE rows, capped at CAP/anchor.
* Anchor kept if it has >= MIN_POS such positives.
* Negatives  : the anchor's shipped negatives (answer_jaccard = 0), matching
               wikidata_testset_RQ1.json. Not size-filtered -- they are absent
               from the cached oracle and, being jaccard=0, never contributed to
               the bimodality.

Result-set sizes come from the cached execution oracle
result_sets_wikidata.json, streamed once with ijson (torch-free).

Output: datasets/wikidata_testset_RQ1_nonsingleton.json

    python scripts/build_nonsingleton_wikidata.py
"""
import json
from pathlib import Path

import ijson
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TRIP = ROOT / "datasets" / "wikidata-triplets-dataset.json"
SPLITS = ROOT / "datasets" / "splits_wikidata.json"
RESULTS = ROOT / "datasets" / "result_sets_wikidata.json"
OUT = ROOT / "datasets" / "wikidata_testset_RQ1_nonsingleton.json"

JAC_MIN = 0.80
MIN_RESULT_SIZE = 5          # anchor & positives must return >= this many rows
CAP = 20
MIN_POS = 1
SEED = 42
SPLIT_KEYS = ["test_idx", "valid_idx"]   # held-out only; train excluded (leak-free)

_norm = lambda s: " ".join(s.split())


def load_result_sizes(wanted):
    """Stream result_sets_wikidata.json once, returning {query: n_rows} for the
    queries in `wanted`. Keys absent from the oracle are left out -> size 0 ->
    filtered (conservative: an unknown-size query is never kept)."""
    if not RESULTS.exists():
        raise SystemExit(
            f"{RESULTS} not found — the non-singleton filter needs the execution "
            "oracle (result set sizes).")
    sizes = {}
    with open(RESULTS, "rb") as f:
        for k, v in ijson.kvitems(f, ""):
            if k in wanted:
                sizes[k] = len(v)
    return sizes


def main():
    trip = json.load(open(TRIP))
    spl = json.load(open(SPLITS))
    idxs = [i for key in SPLIT_KEYS for i in spl[key]]
    test = [trip[i] for i in idxs]
    print(f"source splits: {'+'.join(SPLIT_KEYS)}  ({len(test)} anchors)")

    wanted = set()
    for e in test:
        wanted.add(e["anchor_query"])
        wanted.update(p["query"] for p in e["positive_queries"])
    print(f"streaming {RESULTS.name} for {len(wanted)} query sizes ...")
    sizes = load_result_sizes(wanted)
    size = lambda q: sizes.get(q, 0)
    print(f"  sizes resolved: {len(sizes)}/{len(wanted)} "
          f"(queries absent from oracle are treated as singletons)")

    out, npos, all_sz = [], [], []
    for e in test:
        a = e["anchor_query"]
        if size(a) < MIN_RESULT_SIZE:                     # non-singleton anchor
            continue
        pos = []
        for p in e["positive_queries"]:
            j = p.get("answer_jaccard")
            if j is None or j <= JAC_MIN:
                continue
            if size(p["query"]) < MIN_RESULT_SIZE:        # non-singleton positive
                continue
            pos.append((p["query"], float(j)))
        if len(pos) < MIN_POS:
            continue
        pos = sorted(pos, key=lambda t: -t[1])[:CAP]      # strongest overlap first
        out.append({
            "anchor_query": a,
            "anchor_result_size": size(a),
            "positive_queries": [
                {"query": q, "answer_jaccard": round(j, 4), "result_size": size(q)}
                for (q, j) in pos],
            "negative_queries": [
                {"query": n["query"],
                 "answer_jaccard": round(float(n.get("answer_jaccard", 0.0)), 4)}
                for n in e["negative_queries"]],
        })
        npos.append(len(pos))
        all_sz.append(size(a))
        all_sz += [size(q) for q, _ in pos]

    if not out:
        raise SystemExit(
            f"No anchors qualify with result set >= {MIN_RESULT_SIZE} and a "
            f"jaccard>{JAC_MIN} positive — loosen MIN_RESULT_SIZE or check oracle "
            "coverage.")
    json.dump(out, open(OUT, "w"), ensure_ascii=False, indent=1)

    corpus = set()
    for r in out:
        corpus.add(_norm(r["anchor_query"]))
        corpus.update(_norm(p["query"]) for p in r["positive_queries"])
        corpus.update(_norm(n["query"]) for n in r["negative_queries"])
    sz = np.array(all_sz)
    print(f"\nwrote {len(out)} anchors to {OUT}")
    print(f"  test anchors scanned: {len(test)}  ->  qualifying "
          f"(size>={MIN_RESULT_SIZE}, >={MIN_POS} positive jaccard>{JAC_MIN}): {len(out)}")
    print(f"  positives/anchor: min {min(npos)} mean {np.mean(npos):.2f} max {max(npos)} "
          f"(total {sum(npos)})")
    print(f"  result-set size (anchor+positives): min {sz.min()} median "
          f"{int(np.median(sz))} mean {sz.mean():.0f} max {sz.max()}  (all >= {MIN_RESULT_SIZE})")
    print(f"  positive jaccard: min "
          f"{min(p['answer_jaccard'] for r in out for p in r['positive_queries']):.3f}  "
          f"(all > {JAC_MIN})")
    print(f"  negatives/anchor: mean "
          f"{np.mean([len(r['negative_queries']) for r in out]):.1f}  (shipped, jaccard 0)")
    print(f"  unique corpus queries: {len(corpus)}")


if __name__ == "__main__":
    main()
