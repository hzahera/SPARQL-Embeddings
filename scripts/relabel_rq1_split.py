#!/usr/bin/env python3
"""Make an RQ1 split's relevance judgments COMPLETE with respect to its own pool.

The problem
-----------
An RQ1 split's positives come from the triplet file's precomputed positive lists,
capped at --cap per anchor. They were never derived from the split's actual
candidate pool. But retrieval is scored against that whole pool, and the pool is
built from every anchor's queries -- so anchor A's pool contains anchor B's
positives, and some of those are result-equivalent to A as well.

Measured on datasets/dbpedia-testsplit-RQ1.json: 1,103 pool queries across
102/212 anchors (48.1%) have answer-set Jaccard > 0.8 to their anchor while not
being labelled positive for it. By the benchmark's own criterion those are
correct answers being scored as errors.

Incomplete judgments do not bias all systems equally: a model that is better at
surfacing genuine result-equivalence is penalised more than one that ranks by
surface form, because it is the one that puts unlabelled-but-correct queries in
the top-k. That makes the comparison this benchmark exists to run unreliable in
precisely the direction that matters.

The fix
-------
Relabel exhaustively: for every anchor, execute the Jaccard criterion against
every query in the pool using the cached answer sets, and label EVERY query over
the threshold as a positive. The judgment set then matches what the metric
assumes -- complete relevance over the candidate set.

This only ADDS labels; it never removes an existing positive (a pair kept by the
original build stays, even if the oracle has no entry for it). Anchors whose own
answer set is missing from the oracle cannot be relabelled and are reported.

Leakage is re-checked afterwards: a newly added positive can reintroduce a pair
the model trained on, so R3 is re-applied to the added labels unless
--keep-leaked is passed.

Usage
-----
    # DBpedia: small oracle, seconds
    python scripts/relabel_rq1_split.py --kg dbpedia \
        --split datasets/dbpedia-testsplit-RQ1.json --report-only

    # Wikidata: streams the ~6GB oracle once, minutes
    python scripts/relabel_rq1_split.py --kg wikidata \
        --split datasets/wikidata-testsplit-RQ1.json \
        --out datasets/wikidata-testsplit-RQ1-complete.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

ORACLE = {"dbpedia": "result_sets_dbpedia.json",
          "wikidata": "result_sets_wikidata.json"}
TRIPLETS = {"dbpedia": "dbpedia-triplets-dataset.json",
            "wikidata": "wikidata-triplets-dataset.json"}
SPLITS = {"dbpedia": "splits_dbpedia.json", "wikidata": "splits_wikidata.json"}

STREAM_THRESHOLD = 512 * 1024 * 1024      # stream anything bigger than this


def norm(s):
    return " ".join(str(s).split())


def _q(entry):
    if isinstance(entry, dict):
        return str(entry.get("query") or entry.get("sparql") or "")
    return str(entry or "")


def as_answer_set(rows):
    """Normalize a cached result to a hashable set of row strings.

    Scalar payloads (a bare count) carry no set structure, so they are treated as
    unusable rather than as a one-element set -- comparing two such values yields
    Jaccard 1.0 whenever two numbers coincide, which is the collision R2 exists
    to remove.
    """
    if rows is None or isinstance(rows, (int, float, str)):
        return None
    try:
        return frozenset(json.dumps(r, sort_keys=True) if isinstance(r, (dict, list))
                         else str(r) for r in rows)
    except TypeError:
        return None


def load_answer_sets(kg, wanted, oracle=None):
    """{normalized_query: frozenset(rows)} for the queries in `wanted`."""
    path = ROOT / "datasets" / (oracle or ORACLE[kg])
    if not path.exists():
        raise SystemExit(f"{path} not found -- the exhaustive relabelling needs the "
                         f"execution oracle (see datasets/README.md).")
    out = {}
    if path.stat().st_size > STREAM_THRESHOLD:
        try:
            import ijson
        except ImportError:
            raise SystemExit("pip install ijson -- needed to stream the large oracle.")
        print(f"streaming {path.name} ({path.stat().st_size / 1e9:.1f} GB) for "
              f"{len(wanted)} queries ...", flush=True)
        with open(path, "rb") as f:
            for k, v in ijson.kvitems(f, ""):
                nk = norm(k)
                if nk in wanted and nk not in out:
                    s = as_answer_set(v)
                    if s:
                        out[nk] = s
    else:
        for k, v in json.load(open(path)).items():
            nk = norm(k)
            if nk in wanted:
                s = as_answer_set(v)
                if s:
                    out[nk] = s
    return out


def trained_pairs(kg, which="train+valid"):
    """Same reconstruction as scripts/check_leakage.py."""
    trip = json.load(open(ROOT / "datasets" / TRIPLETS[kg]))
    spl = json.load(open(ROOT / "datasets" / SPLITS[kg]))
    groups = []
    for item in trip:
        a = item.get("anchor_query")
        if not a:
            continue
        pos = [_q(p) for p in item.get("positive_queries", []) if _q(p)]
        neg = [_q(n) for n in item.get("negative_queries", []) if _q(n)]
        if not pos or not neg:
            continue
        groups.append((norm(a), [norm(p) for p in pos]))
    keys = ["train_idx"] + (["valid_idx"] if which == "train+valid" else [])
    pairs = set()
    for k in keys:
        for i in spl.get(k, []):
            if i < len(groups):
                a, pos = groups[i]
                pairs.update(frozenset((a, p)) for p in pos)
    return pairs


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kg", required=True, choices=sorted(ORACLE))
    ap.add_argument("--split", required=True)
    ap.add_argument("--jac-min", type=float, default=0.80,
                    help="a pool query is a positive iff Jaccard > this (default "
                         "0.80, matching the split's construction threshold)")
    ap.add_argument("--keep-leaked", action="store_true",
                    help="do NOT re-apply the memorized-pair filter to newly added "
                         "positives (only for measuring the effect)")
    ap.add_argument("--leak-splits", default="train+valid",
                    choices=["train", "train+valid"])
    ap.add_argument("--results", default=None, metavar="FILENAME",
                    help="execution-oracle filename under datasets/ (default: the "
                         "KG's result_sets_*.json). Use the same oracle the split "
                         "was built from -- mixing snapshots makes Jaccard wrong.")
    ap.add_argument("--out", default=None)
    ap.add_argument("--report-only", action="store_true")
    args = ap.parse_args()

    split = json.load(open(args.split))
    pool = set()
    for r in split:
        pool.add(norm(r["anchor_query"]))
        pool.update(norm(_q(p)) for p in r.get("positive_queries", []))
        pool.update(norm(_q(n)) for n in r.get("negative_queries", []))
    print(f"split: {Path(args.split).name}  anchors={len(split)}  pool={len(pool)}")

    ans = load_answer_sets(args.kg, pool, args.results)
    print(f"oracle coverage: {len(ans)}/{len(pool)} pool queries "
          f"({len(ans) / max(len(pool), 1):.1%}) have a usable answer set")

    trained = set() if args.keep_leaked else trained_pairs(args.kg, args.leak_splits)
    pool_sorted = sorted(pool)

    out, added_tot, unresolved, added_leaked = [], 0, 0, 0
    added_per, excl_per = [], []
    for r in split:
        a = norm(r["anchor_query"])
        existing = {norm(_q(p)): p for p in r.get("positive_queries", [])}
        ra = ans.get(a)
        added, excluded = [], []
        if ra is None:
            unresolved += 1
        else:
            for q in pool_sorted:
                if q == a or q in existing:
                    continue
                rq = ans.get(q)
                if rq is None:
                    continue
                j = len(ra & rq) / len(ra | rq)
                if j > args.jac_min:
                    if not args.keep_leaked and frozenset((a, q)) in trained:
                        # Result-equivalent but a memorized training pair: it can
                        # be neither labelled (reintroduces leakage) nor left in
                        # the ranking (scores a correct retrieval as an error).
                        # Mark it unjudged -- evaluate_rq1.build_excluded_maps
                        # drops it from this anchor's ranking.
                        added_leaked += 1
                        excluded.append({"query": q, "answer_jaccard": round(j, 4),
                                         "reason": "equivalent_but_trained_on"})
                        continue
                    added.append({"query": q, "answer_jaccard": round(j, 4),
                                  "relabelled": True})
        added_tot += len(added)
        added_per.append(len(added))
        excl_per.append(len(excluded))
        rec = dict(r)
        rec["positive_queries"] = list(r.get("positive_queries", [])) + added
        if excluded:
            rec["excluded_queries"] = list(r.get("excluded_queries", [])) + excluded
        out.append(rec)

    n_before = sum(len(r.get("positive_queries", [])) for r in split)
    n_after = sum(len(r["positive_queries"]) for r in out)
    ap_arr = np.array(added_per)
    print(f"\nRELABELLING")
    print(f"  positives before : {n_before} (mean {n_before / len(split):.2f}/anchor)")
    print(f"  positives added  : {added_tot} "
          f"({int((ap_arr > 0).sum())}/{len(split)} anchors affected, "
          f"{(ap_arr > 0).mean():.1%})")
    print(f"  positives after  : {n_after} (mean {n_after / len(split):.2f}/anchor)")
    if added_tot:
        print(f"  added per affected anchor: mean {ap_arr[ap_arr > 0].mean():.2f} "
              f"max {ap_arr.max()}")
    ex_arr = np.array(excl_per)
    if not args.keep_leaked:
        print(f"\n  UNJUDGED (equivalent but a memorized training pair): {added_leaked}")
        print(f"    anchors affected: {int((ex_arr > 0).sum())}/{len(split)} "
              f"({(ex_arr > 0).mean():.1%})"
              + (f", mean {ex_arr[ex_arr > 0].mean():.2f} max {ex_arr.max()} each"
                 if added_leaked else ""))
        print(f"    written to each anchor's `excluded_queries`; the evaluator drops")
        print(f"    them from that anchor's ranking instead of scoring them wrong.")
    if unresolved:
        print(f"  anchors NOT relabellable (no usable answer set): {unresolved} "
              f"-- their labels are unchanged and may still be incomplete")

    if args.report_only:
        print("\n--report-only: nothing written")
        return
    if not args.out:
        raise SystemExit("--out required unless --report-only")
    out_path = Path(args.out)
    tmp = out_path.with_name(out_path.name + ".partial")
    json.dump(out, open(tmp, "w"), ensure_ascii=False, indent=1)
    tmp.replace(out_path)
    print(f"\nwrote {len(out)} anchors -> {out_path}")
    print("Re-audit before use:")
    print(f"  python scripts/check_leakage.py --kg {args.kg} --split {out_path}")


if __name__ == "__main__":
    main()
