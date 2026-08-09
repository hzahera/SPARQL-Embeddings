#!/usr/bin/env python3
"""Build an RQ1 corpus whose positives are VERIFIED answer-preserving rewrites.

Why
---
Natural DBpedia logs cannot supply the structural axis. Measured on the shipped
corpus, 61-72% of result-equivalent pairs share a structural template and differ
only in their constants -- a fact about the graph, not the query. Filtering for
structurally richer queries makes it worse, not better: requiring >=2 triple
patterns and a join vertex collapses 534 templates to 39 and takes the axis-S
share from 38.6% to 0.0%, because the multi-pattern queries in these logs are a
few machine-generated families repeated with different constants.

So the structural axis is constructed rather than mined. A rewrite of a query is
the same query written differently: axis S by definition, with a label that does
not depend on what any entity happens to denote.

Two operators are DELIBERATELY EXCLUDED
---------------------------------------
`variable_renaming` and `filter_true` are baked into the graph builder --
`canonicalize_variables()` and `is_noop_filter_text()` make those rewrites
produce bit-identical graphs, so SparqlCE would score them perfectly by
preprocessing while a text baseline sees two different strings. Including them
manufactures a win that measures normalisation, not learning. They are genuine
architectural properties and `scripts/verify_invariances.py` is where they
belong. Pass --allow-baked-in to override; the script says loudly what that
means.

Answers are VERIFIED, not assumed
---------------------------------
The operators are intended to preserve answers, but intent is not evidence.
`redundant_optional` introduces a fresh variable, which adds a column, which
changes every row key and drives Jaccard to 0. So a rewrite becomes a positive
only after the oracle confirms it, and the report breaks the survival rate down
per operator -- an operator that rarely survives is a bug, not a hard case.

Two stages, because verification needs execution
------------------------------------------------
    # 1. generate rewrites and emit everything needing execution
    python scripts/build_rq1_rewrites.py --in datasets/dbpedia-triplets-dataset.json \
        --out-manifest datasets/dbpedia-rewrites-manifest.json

    # 2. execute (only the rewrites are new; anchors are already in your oracle)
    python scripts/build_result_sets.py --kg dbpedia \
        --split datasets/dbpedia-rewrites-manifest.json \
        --out datasets/result_sets_dbpedia_rewrites.json --pause 0.1

    # 3. verify and build the triplets
    python scripts/build_rq1_rewrites.py --in datasets/dbpedia-triplets-dataset.json \
        --result-sets datasets/result_sets_dbpedia_rewrites.json \
        --also-result-sets datasets/result_sets_dbpedia_v2.json \
        --out datasets/dbpedia-rewrites-triplets.json

The output is the SQuARE triplet schema, so build_component_splits.py,
build_rq1_twoaxis.py and both trainers consume it unchanged.
"""
import argparse
import collections
import hashlib
import importlib.util
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sparqlce.sparql_utils import normalize_query, structural_template

# Operators the graph builder normalises away, so a pair produced by them is
# identical for SparqlCE before any training happens.
BAKED_IN = {"variable_renaming", "filter_true"}


def _load_operators():
    """Import the RQ2 operators from the existing generator.

    Reused rather than reimplemented: RQ1 and RQ2 must apply the SAME rewrites,
    or a robustness claim and a retrieval claim are about different operators.
    """
    path = ROOT / "scripts" / "generate_hard_robustness_dataset.py"
    spec = importlib.util.spec_from_file_location("_rq2ops", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    ops = {
        "variable_renaming":  mod.apply_variable_renaming,
        "triple_shuffle":     mod.apply_triple_shuffle,
        "union_duplication":  mod.apply_union_duplication,
        "subquery_wrapping":  mod.apply_subquery_wrapping,
        "redundant_optional": mod.apply_redundant_optional,
        "filter_true":        mod.apply_filter_true,
        "bgp_split":          mod.apply_bgp_split,
        "distinct_toggle":    mod.apply_distinct_toggle,
    }
    return ops, mod.is_valid_sparql


def load_queries(path, field=None):
    text = Path(path).read_text(encoding="utf-8").strip()
    records = json.loads(text) if text.startswith("[") else \
        [json.loads(l) for l in text.splitlines() if l.strip()]
    keys = [field] if field else ["anchor_query", "query", "sparql_query", "text"]
    out = []
    for r in records:
        if isinstance(r, str):
            out.append(r)
        elif isinstance(r, dict):
            for k in keys:
                if k and r.get(k):
                    out.append(str(r[k]))
                    break
    return list(dict.fromkeys(normalize_query(q) for q in out if q and q.strip()))


def load_oracle(paths):
    """Merge oracles; later files do not override earlier ones."""
    out = {}
    for p in paths:
        p = Path(p) if Path(p).is_absolute() else ROOT / p
        if not p.exists():
            raise SystemExit(f"{p} not found.")
        meta = p.with_name(p.stem + "_meta.json")
        bad = ({normalize_query(q) for q in json.load(open(meta)).get("truncated", [])}
               if meta.exists() else set())
        for k, v in json.load(open(p)).items():
            nk = normalize_query(k)
            if nk not in bad and nk not in out:
                out[nk] = frozenset(v)
    return out


def jaccard(a, b):
    if not a or not b:
        return 0.0
    inter = len(a & b)
    return inter / (len(a) + len(b) - inter)


def _pick(cands, anchor, exclude, n):
    if n <= 0 or not cands:
        return []
    seq = list(cands)
    off = int(hashlib.sha1(anchor.encode()).hexdigest(), 16) % len(seq)
    out = []
    for i in range(len(seq)):
        c = seq[(off + i) % len(seq)]
        if c != anchor and c not in exclude:
            out.append(c)
            if len(out) == n:
                break
    return out


def generate(anchors, ops, is_valid, enabled, seed):
    """{anchor: {operator: rewrite}} for rewrites that are valid and RESHAPED.

    A rewrite whose structural_template equals the anchor's is not axis S -- it
    is the same shape with different tokens -- so it is dropped here rather than
    inflating the corpus with pairs that do not test what it exists to test.
    """
    random.seed(seed)
    out, stats = {}, collections.Counter()
    for a in anchors:
        ta = structural_template(a)
        got = {}
        for name in enabled:
            try:
                r = normalize_query(ops[name](a))
            except Exception:
                stats[f"{name}:error"] += 1
                continue
            if not r or r == a:
                stats[f"{name}:noop"] += 1
                continue
            if not is_valid(r):
                stats[f"{name}:invalid"] += 1
                continue
            if structural_template(r) == ta:
                stats[f"{name}:same_template"] += 1
                continue
            got[name] = r
            stats[f"{name}:generated"] += 1
        if got:
            out[a] = got
    return out, stats


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="inp", required=True, help="source of anchor queries")
    ap.add_argument("--field", default=None)
    ap.add_argument("--operators", default=None,
                    help="comma-separated; default is every operator except the "
                         "two the graph builder normalises away")
    ap.add_argument("--allow-baked-in", action="store_true",
                    help="also use variable_renaming and filter_true. The graph "
                         "builder makes these bit-identical, so SparqlCE scores "
                         "them perfectly without learning anything.")
    ap.add_argument("--max-anchors", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-manifest", default=None,
                    help="stage 1: write every query needing execution")
    ap.add_argument("--result-sets", default=None,
                    help="stage 2: oracle covering the REWRITES")
    ap.add_argument("--also-result-sets", default=None,
                    help="stage 2: an existing oracle covering the ANCHORS, so they "
                         "need not be re-executed")
    ap.add_argument("--jac-min", type=float, default=0.99,
                    help="a rewrite must reproduce the anchor's answers this "
                         "closely (default 0.99: these are supposed to be exact, "
                         "so anything lower means the operator is not "
                         "answer-preserving and should be investigated)")
    ap.add_argument("--min-rows", type=int, default=2)
    ap.add_argument("--hard-negs", type=int, default=20)
    ap.add_argument("--easy-negs", type=int, default=10)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    ops, is_valid = _load_operators()
    if args.operators:
        enabled = [o.strip() for o in args.operators.split(",")]
        for o in enabled:
            if o not in ops:
                raise SystemExit(f"unknown operator {o!r}; choose from {sorted(ops)}")
    else:
        enabled = [o for o in ops if o not in BAKED_IN]
    baked = [o for o in enabled if o in BAKED_IN]
    if baked and not args.allow_baked_in:
        raise SystemExit(
            f"{baked} are normalised away by the graph builder "
            f"(canonicalize_variables / is_noop_filter_text), so SparqlCE embeds "
            f"those pairs identically before training and would score them 1.0 by "
            f"construction while a text baseline sees different strings. Report "
            f"them via scripts/verify_invariances.py instead, or pass "
            f"--allow-baked-in if you really intend this.")
    if baked:
        print(f"!!! --allow-baked-in: {baked} give SparqlCE a preprocessing win. "
              f"Do not report the result as retrieval quality.")
    print(f"operators: {enabled}")

    anchors = load_queries(args.inp, args.field)
    if args.max_anchors:
        anchors = anchors[:args.max_anchors]
    print(f"anchors: {len(anchors)}")

    rewrites, stats = generate(anchors, ops, is_valid, enabled, args.seed)
    n_rw = sum(len(v) for v in rewrites.values())
    print(f"generated {n_rw} candidate rewrites over {len(rewrites)} anchors")
    for k in sorted(stats):
        print(f"    {k}: {stats[k]}")

    # ── stage 1 ───────────────────────────────────────────────────────────────
    if args.out_manifest:
        pool = list(dict.fromkeys(
            list(rewrites) + [r for v in rewrites.values() for r in v.values()]))
        out_path = Path(args.out_manifest)
        json.dump([{"anchor_query": q, "positive_queries": [], "negative_queries": []}
                   for q in pool], open(out_path, "w"), ensure_ascii=False)
        print(f"\nwrote {len(pool)} queries -> {out_path}")
        print(f"next:\n  python scripts/build_result_sets.py --kg dbpedia \\\n"
              f"    --split {out_path} \\\n"
              f"    --out datasets/result_sets_dbpedia_rewrites.json --pause 0.1")
        return

    if not args.result_sets:
        raise SystemExit("give --out-manifest (stage 1) or --result-sets (stage 2).")

    # ── stage 2: verify ───────────────────────────────────────────────────────
    paths = [args.result_sets] + ([args.also_result_sets] if args.also_result_sets else [])
    oracle = load_oracle(paths)
    print(f"\noracle: {len(oracle)} queries from {len(paths)} file(s)")

    verified, vstats = {}, collections.Counter()
    for a, byop in rewrites.items():
        ra = oracle.get(a)
        if ra is None or len(ra) < args.min_rows:
            vstats["anchor_unusable"] += 1
            continue
        keep = {}
        for name, r in byop.items():
            rr = oracle.get(r)
            if rr is None:
                vstats[f"{name}:not_executed"] += 1
                continue
            j = jaccard(ra, rr)
            if j > args.jac_min:
                keep[name] = (r, j)
                vstats[f"{name}:VERIFIED"] += 1
            else:
                vstats[f"{name}:answers_changed"] += 1
        if keep:
            verified[a] = keep

    print(f"\nverification (a rewrite is a positive only if the oracle agrees):")
    for k in sorted(vstats):
        print(f"    {k}: {vstats[k]}")
    per_op = collections.Counter()
    for k, v in vstats.items():
        if k.endswith(":VERIFIED"):
            per_op[k.split(":")[0]] = v
    if not verified:
        raise SystemExit("no rewrite preserved its anchor's answers -- check the "
                         "operators and the oracle before going further.")

    # ── stage 2: negatives, from the same pool ────────────────────────────────
    pool = [q for q in dict.fromkeys(
        list(verified) + [r for v in verified.values() for r, _ in v.values()])
        if len(oracle.get(q, ())) >= args.min_rows]
    tmpl = {q: structural_template(q) for q in pool}
    by_tmpl = collections.defaultdict(list)
    for q in pool:
        by_tmpl[tmpl[q]].append(q)
    postings = collections.defaultdict(list)
    for i, q in enumerate(pool):
        for row in oracle[q]:
            postings[row].append(i)

    records, axis = [], collections.Counter()
    for a, keep in verified.items():
        if a not in tmpl:
            continue
        touched = set()
        for row in oracle[a]:
            for i in postings.get(row, ()):
                touched.add(pool[i])
        blocked = touched | {r for r, _ in keep.values()}
        hard = _pick(by_tmpl.get(tmpl[a], ()), a, blocked, args.hard_negs)
        easy = _pick([q for q in pool if tmpl[q] != tmpl[a]], a, blocked, args.easy_negs)
        if not hard and not easy:
            continue
        for _ in keep:
            axis["S"] += 1
        records.append({
            "anchor_query": a,
            "positive_queries": [
                {"query": r, "answer_jaccard": round(j, 4), "axis": "S",
                 "operator": name}
                for name, (r, j) in sorted(keep.items())],
            "negative_queries": (
                [{"query": q, "answer_jaccard": 0.0, "hardness": "hard"} for q in hard]
                + [{"query": q, "answer_jaccard": 0.0, "hardness": "easy"} for q in easy]),
        })

    npos = [len(r["positive_queries"]) for r in records]
    corpus = set()
    for r in records:
        corpus.add(r["anchor_query"])
        corpus.update(p["query"] for p in r["positive_queries"])
        corpus.update(n["query"] for n in r["negative_queries"])
    print(f"\nanchors {len(records)} | positives {sum(npos)} "
          f"({sum(npos)/max(len(records),1):.2f}/anchor) | corpus {len(corpus)}")
    print(f"AXIS SPLIT: S {axis['S']} (100.0%)  C 0 -- structural by construction")
    print(f"verified positives by operator: {dict(per_op)}")

    out_path = Path(args.out) if args.out else ROOT / "datasets" / "rewrites-triplets.json"
    json.dump(records, open(out_path, "w"), ensure_ascii=False)
    print(f"\nwrote {out_path}")
    print("next:\n  python scripts/build_component_splits.py --kg dbpedia "
          "--result-sets <oracle> --out datasets/splits_dbpedia_rewrites.json")


if __name__ == "__main__":
    main()
