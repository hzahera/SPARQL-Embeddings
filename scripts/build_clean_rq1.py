#!/usr/bin/env python3
"""Build the CLEAN natural RQ1 retrieval split -- one identical protocol per KG.

Motivation
----------
The Wikidata RQ1 sets currently in `datasets/` are either (a) natural but
label-noisy, or (b) clean only because they were filtered on anchor--positive
lexical similarity (`_adversarial`, cos < 0.3). Filtering on lexical similarity
selects for *difficulty*, which makes a "SparqlCE wins" claim look
outcome-selected however honestly it was arrived at.

This script instead removes pairs whose *label is invalid* and applies NO
lexical filter at all, so the resulting split keeps the natural
lexical-similarity distribution -- exactly like the DBpedia set. Crucially the
same rules run on BOTH knowledge graphs, so the Wikidata split cannot be
characterised as tuned: whatever is done to Wikidata is done to DBpedia too.

The two label-validity rules (both fixed a priori, neither referring to any
model's output):

R1. NON-SINGLETON RESULTS. Anchor and positive must each return
    >= --min-result-size rows. `answer_jaccard > 0.8` between two single-row
    results only means "the same one answer", which collapses to a coin flip
    between identical (J=1) and disjoint (J=0) and never tests set overlap.
    On Wikidata this also removes the external-id lookups whose equivalence is
    decided by which entity an opaque catalogue id happens to resolve to --
    a fact about the graph's contents that is absent from the query text and
    therefore unlearnable by ANY query encoder.

R2. NO SCALAR-AGGREGATE COLLISIONS. Drop pairs where both queries return a
    single aggregate row (COUNT/SUM/AVG/MIN/MAX with no GROUP BY). Their
    `answer_jaccard = 1.0` records that two *numbers* coincided, not that the
    queries are result-equivalent. In `wikidata_testset_RQ1.json` 40.2% of all
    anchor--positive pairs are of this form. R1 removes most of them via the
    row count; R2 makes the rule explicit and still applies where the execution
    oracle has no entry for a query.

Everything else matches the existing DBpedia RQ1 protocol: positives
`answer_jaccard > --jac-min`, capped at --cap per anchor, strongest overlap
first; the shipped `answer_jaccard = 0` distractors as negatives; held-out
splits only.

Usage
-----
    # DBpedia -- runs anywhere (result_sets_dbpedia.json is small)
    python scripts/build_clean_rq1.py --kg dbpedia

    # Wikidata -- needs the ~6GB execution oracle, run on the cluster
    python scripts/build_clean_rq1.py --kg wikidata

Then evaluate both with the SAME command, changing only --split:

    python scripts/eval_rq1_stratified.py --stratify \
      --split datasets/{dbpedia,wikidata}_testset_RQ1_clean.json \
      --st intfloat/e5-base-v2 --st checkpoints/e5_square_<kg>/best_model \
      --sparqlce checkpoints_b128/<kg>/seed42_full/best_model.pt
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

KG = {
    "dbpedia": {
        "triplets": "dbpedia-triplets-dataset.json",
        "splits": "splits_dbpedia.json",
        "results": "result_sets_dbpedia.json",
        "out": "dbpedia_testset_RQ1_clean.json",
    },
    "wikidata": {
        "triplets": "wikidata-triplets-dataset.json",
        "splits": "splits_wikidata.json",
        "results": "result_sets_wikidata.json",
        "out": "wikidata_testset_RQ1_clean.json",
    },
}

# A scalar aggregate: an aggregate in the projection and no GROUP BY, so the
# query returns exactly one row whose value is a number.
_AGG_RE = re.compile(r'\b(COUNT|SUM|AVG|MIN|MAX|GROUP_CONCAT|SAMPLE)\s*\(', re.IGNORECASE)
_GROUP_BY_RE = re.compile(r'\bGROUP\s+BY\b', re.IGNORECASE)


def norm(s):
    return " ".join(str(s).split())


def _q(entry):
    """Query text of a positive/negative entry (dict in the triplet files, plain
    string in some built splits)."""
    if isinstance(entry, dict):
        return str(entry.get("query") or entry.get("sparql") or "")
    return str(entry or "")


def _j(entry, default=1.0):
    """answer_jaccard of an entry; `default` when the split does not carry it."""
    if isinstance(entry, dict) and entry.get("answer_jaccard") is not None:
        return float(entry["answer_jaccard"])
    return default


def is_scalar_aggregate(query):
    """True if the query projects an aggregate and does not GROUP BY."""
    head = re.split(r'\bWHERE\b', query, maxsplit=1, flags=re.IGNORECASE)[0]
    return bool(_AGG_RE.search(head)) and not _GROUP_BY_RE.search(query)


def load_result_sizes(path, wanted):
    """Return {normalized_query: n_rows}.

    Keys are normalized on both sides: the cached oracle and the triplet file
    do not agree byte-for-byte on whitespace (on DBpedia an exact-key lookup
    resolves 0/472 anchors, a normalized one resolves 396/472), and a missed
    lookup silently reads as "size 0" and filters the query out.
    """
    if not path.exists():
        raise SystemExit(
            f"{path} not found -- the non-singleton rule needs the execution "
            f"oracle. For Wikidata this file is ~6GB and must be run on the "
            f"cluster (see datasets/README.md).")
    sizes = {}
    if path.stat().st_size > 512 * 1024 * 1024:      # stream the big Wikidata oracle
        try:
            import ijson
        except ImportError:
            raise SystemExit("pip install ijson -- needed to stream the large oracle.")
        with open(path, "rb") as f:
            for k, v in ijson.kvitems(f, ""):
                nk = norm(k)
                if nk in wanted:
                    sizes[nk] = len(v)
    else:
        for k, v in json.load(open(path)).items():
            nk = norm(k)
            if nk in wanted:
                sizes[nk] = len(v)
    return sizes


def lexical_profile(records):
    """Mean anchor->positive TF-IDF cosine per anchor, on raw query text.

    Reported only to DEMONSTRATE the split is unfiltered -- it is never used to
    include or exclude anything. (The paper's bins are computed at eval time on
    label-expanded text by eval_rq1_stratified.py --stratify; this is the
    build-time sanity check that the natural spread survived.)
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.preprocessing import normalize as sk_norm

    corpus, seen = [], {}
    for r in records:
        for q in [r["anchor_query"]] + [p["query"] for p in r["positive_queries"]]:
            if norm(q) not in seen:
                seen[norm(q)] = len(corpus)
                corpus.append(norm(q))
    X = sk_norm(TfidfVectorizer(analyzer="word", ngram_range=(1, 2))
                .fit_transform(corpus), norm="l2")
    means = []
    for r in records:
        ai = seen[norm(r["anchor_query"])]
        cs = [float(X[ai].multiply(X[seen[norm(p["query"])]]).sum())
              for p in r["positive_queries"]]
        means.append(float(np.mean(cs)) if cs else 0.0)
    return np.array(means)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kg", choices=sorted(KG), required=True)
    ap.add_argument("--min-result-size", type=int, default=5,
                    help="R1: min rows for anchor and each positive (default 5). "
                         "Set 0 to DISABLE R1 entirely -- no execution oracle is "
                         "then needed, and R2 alone is applied. On Wikidata this "
                         "is usually what you want: see --from-split below.")
    ap.add_argument("--from-split", default=None, metavar="PATH",
                    help="apply the rules to an already-built RQ1 split instead "
                         "of the raw triplets (e.g. "
                         "datasets/wikidata_testset_RQ1_stratified.json). The "
                         "source split's anchors and negatives are preserved, so "
                         "leakage properties are inherited unchanged. Combine "
                         "with --min-result-size 0 for an oracle-free build.")
    ap.add_argument("--jac-min", type=float, default=0.80,
                    help="positive iff answer_jaccard > this (default 0.80)")
    ap.add_argument("--cap", type=int, default=20, help="max positives per anchor")
    ap.add_argument("--min-pos", type=int, default=1, help="min positives to keep an anchor")
    ap.add_argument("--splits", default="test", choices=["test", "test+valid"],
                    help="held-out splits to draw anchors from; train is NEVER used")
    ap.add_argument("--keep-scalar-aggregates", action="store_true",
                    help="disable R2 (kept only for ablating the rule's effect)")
    ap.add_argument("--target-n", type=int, default=None,
                    help="downsample to exactly N anchors, deterministically "
                         "(seed --sample-seed). Sampling is UNIFORM over the "
                         "qualifying anchors -- it never looks at lexical "
                         "similarity, result size or any model score, so it "
                         "cannot bias difficulty. Omit to keep every anchor.")
    ap.add_argument("--sample-seed", type=int, default=42)
    ap.add_argument("--report-only", action="store_true",
                    help="print the composition report without writing the split")
    ap.add_argument("--out", default=None, help="override output path")
    args = ap.parse_args()

    cfg = KG[args.kg]
    if args.from_split:
        src = Path(args.from_split)
        if not src.exists():
            raise SystemExit(f"{src} not found.")
        held = json.load(open(src))
        print(f"[{args.kg}] source: {src.name}  ({len(held)} anchors; leakage "
              f"properties inherited from that split)")
    else:
        trip_path = ROOT / "datasets" / cfg["triplets"]
        if not trip_path.exists():
            raise SystemExit(f"{trip_path} not found (see datasets/README.md).")
        trip = json.load(open(trip_path))
        spl = json.load(open(ROOT / "datasets" / cfg["splits"]))
        keys = ["test_idx"] + (["valid_idx"] if args.splits == "test+valid" else [])
        idxs = [i for k in keys for i in spl[k]]
        held = [trip[i] for i in idxs]
        print(f"[{args.kg}] source: {'+'.join(keys)}  "
              f"({len(held)} anchors, train excluded)")

    if args.min_result_size <= 0:
        print(f"[{args.kg}] R1 DISABLED (--min-result-size 0): no execution oracle "
              f"used; R2 (scalar-aggregate collisions) applied alone.")
        sizes = {}
        size = lambda q: args.min_result_size      # always passes the R1 check
    else:
        wanted = set()
        for e in held:
            wanted.add(norm(e["anchor_query"]))
            wanted.update(norm(_q(p)) for p in e["positive_queries"])
        print(f"[{args.kg}] resolving result-set sizes for {len(wanted)} queries ...")
        sizes = load_result_sizes(ROOT / "datasets" / cfg["results"], wanted)
        cov = len(sizes) / max(len(wanted), 1)
        print(f"[{args.kg}]   oracle coverage: {len(sizes)}/{len(wanted)} "
              f"({cov * 100:.1f}%); uncovered queries are treated "
              f"as size 0 and dropped (conservative)")
        if cov < 0.75:
            print(f"[{args.kg}]   WARNING: coverage below 75%. Most 'anchor_small' "
                  f"drops below are UNKNOWN size, not verified-small, so R1 is "
                  f"largely selecting for oracle presence. On Wikidata R1 also "
                  f"correlates with high lexical similarity (multi-row queries are "
                  f"the near-duplicate BIND/path template families), which skews "
                  f"the split toward the easy >=0.8 bin. Consider "
                  f"--min-result-size 0 --from-split <natural split>.")
        size = lambda q: sizes.get(norm(q), 0)

    out = []
    drop = {"anchor_small": 0, "anchor_scalar_agg": 0,
            "pos_jaccard": 0, "pos_small": 0, "pos_scalar_agg": 0, "no_pos": 0}
    for e in held:
        a = e["anchor_query"]
        a_scalar = is_scalar_aggregate(a)
        if size(a) < args.min_result_size:                       # R1
            drop["anchor_small"] += 1
            continue
        if a_scalar and not args.keep_scalar_aggregates:         # R2
            drop["anchor_scalar_agg"] += 1
            continue
        pos = []
        for p in e["positive_queries"]:
            pq = _q(p)
            # A split that does not record answer_jaccard was already built with
            # a jaccard threshold, so treat a missing value as passing rather
            # than silently discarding every positive.
            j = _j(p, default=1.0)
            if not pq or j <= args.jac_min:
                drop["pos_jaccard"] += 1
                continue
            if size(pq) < args.min_result_size:                  # R1
                drop["pos_small"] += 1
                continue
            # R2 -- a scalar-aggregate positive can only coincide numerically
            if not args.keep_scalar_aggregates and is_scalar_aggregate(pq):
                drop["pos_scalar_agg"] += 1
                continue
            pos.append((pq, j))
        if len(pos) < args.min_pos:
            drop["no_pos"] += 1
            continue
        pos = sorted(pos, key=lambda t: -t[1])[:args.cap]
        out.append({
            "anchor_query": a,
            "anchor_result_size": size(a),
            "positive_queries": [
                {"query": q, "answer_jaccard": round(j, 4), "result_size": size(q)}
                for q, j in pos],
            "negative_queries": [
                {"query": _q(n), "answer_jaccard": round(_j(n, default=0.0), 4)}
                for n in e.get("negative_queries", []) if _q(n)],
        })

    if not out:
        raise SystemExit(
            f"No anchors qualify. Loosen --min-result-size (currently "
            f"{args.min_result_size}) or check oracle coverage above.")

    qualified = len(out)
    if args.target_n and args.target_n < qualified:
        rng = np.random.default_rng(args.sample_seed)
        keep = sorted(rng.choice(qualified, size=args.target_n, replace=False).tolist())
        out = [out[i] for i in keep]
        print(f"  downsampled {qualified} -> {len(out)} anchors "
              f"(uniform, seed {args.sample_seed})")
    elif args.target_n:
        print(f"  NOTE: only {qualified} anchors qualify, below --target-n "
              f"{args.target_n}; keeping all. Lower --min-result-size or use "
              f"--splits test+valid to raise the yield.")

    out_path = Path(args.out) if args.out else ROOT / "datasets" / cfg["out"]
    if args.report_only:
        print(f"  --report-only: not writing {out_path}")
    else:
        json.dump(out, open(out_path, "w"), ensure_ascii=False, indent=1)

    npos = [len(r["positive_queries"]) for r in out]
    sz = np.array([r["anchor_result_size"] for r in out]
                  + [p["result_size"] for r in out for p in r["positive_queries"]])
    corpus = set()
    for r in out:
        corpus.add(norm(r["anchor_query"]))
        corpus.update(norm(p["query"]) for p in r["positive_queries"])
        corpus.update(norm(n["query"]) for n in r["negative_queries"])

    verb = "reported" if args.report_only else "wrote"
    print(f"\n{verb} {len(out)} anchors -> {out_path}")
    print(f"  dropped: {drop}")
    print(f"  positives/anchor: min {min(npos)} mean {np.mean(npos):.2f} "
          f"max {max(npos)} (total {sum(npos)})")
    if args.min_result_size > 0:
        print(f"  result sizes: min {sz.min()} median {int(np.median(sz))} "
              f"max {sz.max()} (all >= {args.min_result_size})")
    else:
        print("  result sizes: not resolved (R1 disabled, no oracle read)")
    print(f"  unique corpus queries: {len(corpus)}")

    try:
        prof = lexical_profile(out)
    except ImportError:
        print("  (install scikit-learn to print the lexical-similarity profile)")
        return
    bins = [("<0.3", (prof < 0.3)), ("0.3-0.5", (prof >= 0.3) & (prof < 0.5)),
            ("0.5-0.8", (prof >= 0.5) & (prof < 0.8)), (">=0.8", prof >= 0.8)]
    print(f"\n  UNFILTERED lexical spread (raw text; proves no cos filter was applied):")
    print(f"    mean {prof.mean():.3f}  median {np.median(prof):.3f}  "
          f"min {prof.min():.3f}  max {prof.max():.3f}")
    for name, mask in bins:
        print(f"    {name:>8}: {int(mask.sum()):4d} anchors ({mask.mean() * 100:4.1f}%)")


if __name__ == "__main__":
    main()
