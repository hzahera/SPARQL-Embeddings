#!/usr/bin/env python3
"""Build the RQ1 retrieval split on a component partition, judged exhaustively.

This supersedes `build_clean_rq1.py` for any KG that has a component split
(`build_component_splits.py`) and full oracle coverage of its corpus. It differs
in three ways that all follow from those two inputs.

1. Leakage is structural, not filtered
--------------------------------------
`build_clean_rq1.py` needs rule R3 to delete anchor-positive pairs that also
occur in training. On a component partition no such pair exists: anchors and
their positives are always in the same component, and a component lies wholly
inside one split. This script asserts that instead of enforcing it -- if a
memorized pair turns up, the partition is wrong and the build fails.

2. Judgments are exhaustive, so "unlabelled but correct" cannot happen
----------------------------------------------------------------------
The shipped positive lists were computed against the whole corpus, never against
the split's own candidate pool, so anchor A's pool contained anchor B's
positives -- some of them result-equivalent to A and scored as errors anyway. On
the old DBpedia split that was 1,103 queries across 48% of anchors, and it
penalised precisely the model that was best at surfacing true equivalence.

Here every anchor is judged against every pool query with the oracle, and the
verdict is one of three, never "absent":

    Jaccard > --jac-min      -> positive
    Jaccard == 0             -> negative
    0 < Jaccard <= --jac-min -> EXCLUDED (unjudged; partial overlap is neither)
    not resolvable in oracle -> kept out of the pool entirely (absence from the
                                oracle is not evidence of anything, and a query
                                that is never labelled never gets ranked)

Excluded queries are dropped from that anchor's ranking, the standard treatment
of unjudged documents in a pooled IR evaluation. Crucially, on a component split
a newly discovered positive is still inside the test split, so relabelling adds
no leakage -- the old build had to discard 547 such discoveries because
labelling them would have reintroduced training pairs.

3. Negatives are hard, and there are more than 21 of them
----------------------------------------------------------
The shipped triplets carry 24 distinct negatives for the ENTIRE 4,713-group
corpus, all lexically unrelated to their anchors. A benchmark built from them
cannot punish a purely lexical matcher, which is the one failure mode
"result-aware embedding" claims to fix. This script instead prefers negatives
that share the anchor's `structural_template` and differ only in their
constants, so the ranking contains queries that are nearly identical as text and
provably result-disjoint. On DBpedia 81.5% of same-template pairs are disjoint,
so they are abundant.

The two axes
------------
Result-equivalence here has two sources, and pooling them hides which one a
score comes from:

  * **Axis S** -- different template, same answers: a structural rewrite.
    A property of the QUERY, and what result-awareness should mean.
  * **Axis C** -- same template, different constants with the same extension.
    A property of the GRAPH; inferable only through knowledge about constants.

Three files are written. In each axis file the OTHER axis's positives become
excluded rather than negative: they are correct answers, so counting them as
errors would understate the model, but they are off-axis, so counting them as
hits would defeat the stratification.

Usage
-----
    python scripts/build_rq1_twoaxis.py --kg dbpedia \
        --splits datasets/splits_dbpedia_components.json \
        --result-sets datasets/result_sets_dbpedia_v2.json \
        --out datasets/dbpedia-testsplit-RQ1.json
"""
import argparse
import collections
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sparqlce.sparql_utils import normalize_query, structural_template

KG = {
    "dbpedia": {"triplets": "dbpedia-triplets-dataset.json",
                "splits": "splits_dbpedia_components.json",
                "results": "result_sets_dbpedia_v2.json",
                "out": "dbpedia-testsplit-RQ1.json"},
    "wikidata": {"triplets": "wikidata-triplets-dataset.json",
                 "splits": "splits_wikidata_components.json",
                 "results": "result_sets_wikidata_v2.json",
                 "out": "wikidata-testsplit-RQ1.json"},
}


def _q(entry):
    if isinstance(entry, dict):
        return str(entry.get("query") or entry.get("sparql") or "")
    return str(entry or "")


def _pick(candidates, anchor, exclude, n):
    """Up to `n` of `candidates`, starting from a stable per-anchor offset.

    The rotation is what stops every anchor from being handed the same first few
    queries. `hashlib` rather than `hash()` because the latter is salted per
    process, which would make the split non-reproducible across runs.
    """
    if n <= 0 or not candidates:
        return []
    seq = list(candidates)
    off = int(hashlib.sha1(anchor.encode()).hexdigest(), 16) % len(seq)
    out = []
    for i in range(len(seq)):
        q = seq[(off + i) % len(seq)]
        if q != anchor and q not in exclude:
            out.append(q)
            if len(out) == n:
                break
    return out


def truncated_queries(oracle_path):
    """Normalized queries whose stored answer set is a PREFIX, not a set.

    `build_result_sets.py --max-rows` caps what it stores and records the
    offenders in a `<stem>_meta.json` sidecar. The oracle itself is a plain
    {query: rows} map, so a truncated entry is indistinguishable from a complete
    one -- and comparing a prefix against a full set understates Jaccard by an
    unknown amount, which silently turns real positives into negatives. Such
    queries must be dropped, not judged.
    """
    meta = oracle_path.with_name(oracle_path.stem + "_meta.json")
    if not meta.exists():
        return set(), meta
    payload = json.load(open(meta))
    return {normalize_query(q) for q in payload.get("truncated", [])}, meta


def load_oracle(path):
    """{normalized_query: frozenset(rows)}, excluding truncated answer sets."""
    if not path.exists():
        raise SystemExit(
            f"{path} not found. Build it first:\n"
            f"  python scripts/build_result_sets.py --kg <kg> "
            f"--split datasets/<kg>-triplets-dataset.json --out {path}")
    bad, meta = truncated_queries(path)
    out = {}
    if path.stat().st_size > 512 * 1024 * 1024:
        try:
            import ijson
        except ImportError:
            raise SystemExit("pip install ijson -- needed to stream the large oracle.")
        with open(path, "rb") as f:
            for k, v in ijson.kvitems(f, ""):
                nk = normalize_query(k)
                if nk not in bad:
                    out[nk] = frozenset(v)
    else:
        for k, v in json.load(open(path)).items():
            nk = normalize_query(k)
            if nk not in bad:
                out[nk] = frozenset(v)
    if bad:
        print(f"  dropped {len(bad)} truncated answer sets (per {meta.name}); "
              f"a capped result is a prefix, not a set, so its Jaccard is wrong")
    elif not meta.exists():
        print(f"  NOTE: no {meta.name} sidecar; assuming no answer set was "
              f"truncated. If this oracle came from --max-rows, verify that.")
    return out


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kg", choices=sorted(KG), default="dbpedia")
    ap.add_argument("--splits", default=None, help="component split file")
    ap.add_argument("--result-sets", default=None, help="execution oracle")
    ap.add_argument("--jac-min", type=float, default=0.80,
                    help="positive iff Jaccard > this (default 0.80)")
    ap.add_argument("--min-rows", type=int, default=2,
                    help="a query must return >= this many rows to be judged "
                         "(default 2: a one-row answer set makes Jaccard a coin "
                         "flip between 1.0 and 0, not a test of set overlap)")
    ap.add_argument("--hard-negs", type=int, default=20,
                    help="max same-template result-disjoint negatives per anchor")
    ap.add_argument("--easy-negs", type=int, default=10,
                    help="max different-template result-disjoint negatives, kept "
                         "so the pool is not exclusively adversarial")
    ap.add_argument("--min-pos", type=int, default=1)
    ap.add_argument("--from-split", default="test", choices=["test", "valid", "train"],
                    help="which component split supplies anchors AND the pool "
                         "(default test)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--report-only", action="store_true")
    args = ap.parse_args()

    cfg = KG[args.kg]
    trip = json.load(open(ROOT / "datasets" / cfg["triplets"]))
    spl_path = Path(args.splits) if args.splits else ROOT / "datasets" / cfg["splits"]
    if not spl_path.is_absolute():
        spl_path = ROOT / spl_path
    spl = json.load(open(spl_path))
    if "partition" not in spl:
        raise SystemExit(
            f"{spl_path.name} is not a component split (no 'partition' key). "
            f"This builder's leak-freeness argument depends on one; use "
            f"build_component_splits.py, or build_clean_rq1.py for a group split.")
    orc_path = Path(args.result_sets) if args.result_sets else ROOT / "datasets" / cfg["results"]
    if not orc_path.is_absolute():
        orc_path = ROOT / orc_path
    oracle = load_oracle(orc_path)
    print(f"oracle {orc_path.name}: {len(oracle)} queries")

    idx = spl[f"{args.from_split}_idx"]
    held = [trip[i] for i in idx]
    print(f"[{args.kg}] anchors from {args.from_split} components: {len(held)}")

    # ── 1. the candidate pool: every query of the held-out components ─────────
    # Restricting the pool to this split is what keeps relabelling safe: any
    # equivalence discovered below is between two held-out queries, so it can
    # never restate a training pair.
    pool = []
    seen = set()
    for e in held:
        for q in [e["anchor_query"]] + [_q(p) for p in e.get("positive_queries", [])]:
            nq = normalize_query(q)
            if nq and nq not in seen:
                seen.add(nq)
                pool.append(nq)
    print(f"[{args.kg}] candidate pool before negatives: {len(pool)}")

    judged = [q for q in pool if len(oracle.get(q, ())) >= args.min_rows]
    print(f"[{args.kg}]   resolvable at >= {args.min_rows} rows: {len(judged)}"
          f" ({len(judged)/max(len(pool),1):.1%})")
    if len(judged) < 0.5 * len(pool):
        print(f"[{args.kg}]   WARNING: under half the pool is judgeable. Most "
              f"exclusions below are UNKNOWN, not verified-partial. Rebuild the "
              f"oracle over the whole corpus before trusting these numbers.")

    tmpl = {q: structural_template(q) for q in pool}
    by_tmpl = collections.defaultdict(list)
    for q in judged:
        by_tmpl[tmpl[q]].append(q)

    # dict.fromkeys: dedupe while keeping order. Two groups whose anchor text
    # differs only in whitespace collapse to one query here, and emitting both
    # would put duplicate records in the split.
    anchors = list(dict.fromkeys(normalize_query(e["anchor_query"]) for e in held))
    anchors = [a for a in anchors if len(oracle.get(a, ())) >= args.min_rows]
    print(f"[{args.kg}] judgeable anchors: {len(anchors)}")

    # ── 2. exhaustive judgment, via an inverted index over answer rows ────────
    # Judging every anchor against every pool query is quadratic, which on the
    # train split is 3,769 x 3,769 set intersections. But almost all of those
    # pairs are disjoint, and a disjoint pair needs no intersection computed: a
    # row -> queries index gives exactly the candidates sharing at least one row,
    # and everything absent from it is Jaccard 0 by construction. That turns the
    # quadratic sweep into one pass over the postings of each anchor's rows.
    postings = collections.defaultdict(list)
    for qi, q in enumerate(judged):
        for row in oracle[q]:
            postings[row].append(qi)
    print(f"[{args.kg}] inverted index: {len(postings)} distinct answer rows")

    records = []
    axis_tot = collections.Counter()
    drop = collections.Counter()
    for a in anchors:
        ans = oracle[a]
        ta = tmpl[a]
        overlap = collections.Counter()
        for row in ans:
            for qi in postings.get(row, ()):
                overlap[qi] += 1

        pos, exc = [], []
        touched = set()
        for qi, inter in overlap.items():
            cand = judged[qi]
            touched.add(cand)
            if cand == a:
                continue
            j = inter / (len(ans) + len(oracle[cand]) - inter)
            if j > args.jac_min:
                axis = "S" if tmpl[cand] != ta else "C"
                pos.append({"query": cand, "answer_jaccard": round(j, 4),
                            "axis": axis, "result_size": len(oracle[cand])})
                axis_tot[axis] += 1
            else:
                exc.append(cand)              # partial overlap -> unjudged
        # Queries the oracle cannot resolve are deliberately NOT listed here.
        # `excluded_queries` removes a query from the anchor's ranking, and a
        # ranking only contains queries that reached the pool -- which is built
        # from labelled anchors/positives/negatives. An unresolvable query is
        # never labelled as anything, so it never enters the pool, and listing it
        # would be a no-op that overstates how much was excluded.

        if len(pos) < args.min_pos:
            drop["no_positives"] += 1
            continue

        # Negatives are the provably disjoint remainder: judgeable, non-anchor,
        # and never touched by the postings sweep (untouched == shares no answer
        # row == Jaccard 0). Same-template candidates are the hard ones.
        #
        # Taken in list order these would be nearly the SAME queries for every
        # anchor -- which is exactly the shipped corpus's defect, 24 negatives
        # shared across 4,713 groups. Rotating the candidate list by a stable
        # per-anchor offset spreads the choice over the whole pool instead, so
        # different anchors contribute different queries to the ranking, while
        # staying deterministic.
        hard = _pick(by_tmpl.get(ta, ()), a, touched, args.hard_negs)
        easy = _pick([q for q in judged if tmpl[q] != ta], a, touched, args.easy_negs)
        records.append({
            "anchor_query": a,
            "anchor_result_size": len(ans),
            "anchor_template": ta,
            "positive_queries": sorted(pos, key=lambda p: -p["answer_jaccard"]),
            "negative_queries": (
                [{"query": q, "answer_jaccard": 0.0, "hardness": "hard"} for q in hard]
                + [{"query": q, "answer_jaccard": 0.0, "hardness": "easy"} for q in easy]),
            "excluded_queries": sorted(exc),
        })

    if not records:
        raise SystemExit("no anchor survived; loosen --min-rows or widen the oracle.")

    # ── 4. the property the component split buys us ───────────────────────────
    # Only meaningful for a HELD-OUT split. With --from-split train the judged
    # pairs are the training pairs, so "leakage" is the intended state and the
    # check would always fire.
    if args.from_split == "train":
        print("\n--from-split train: skipping the memorization check (these pairs "
              "ARE the training data). This file is for TRAINING, not reporting.")
    else:
        trained = set()
        for i in spl["train_idx"] + (spl["valid_idx"] if args.from_split == "test" else []):
            a = normalize_query(trip[i]["anchor_query"])
            for p in trip[i].get("positive_queries", []):
                if _q(p):
                    trained.add(frozenset((a, normalize_query(_q(p)))))
        leaked = sum(1 for r in records for p in r["positive_queries"]
                     if frozenset((r["anchor_query"], p["query"])) in trained)
        print(f"\nmemorized positive pairs after exhaustive relabelling: "
              f"{leaked} (must be 0)")
        assert leaked == 0, (
            "a judged pair was trained on -- the component partition is not sound")

    # ── 5. emit pooled + one file per axis ────────────────────────────────────
    out_path = Path(args.out) if args.out else ROOT / "datasets" / cfg["out"]
    written = []

    def emit(path, axis):
        """axis=None -> pooled. Otherwise keep that axis's positives and move the
        other axis's to excluded: they are correct, so they must not be scored as
        errors, but they are off-axis, so they must not be scored as hits."""
        out = []
        for r in records:
            keep = [p for p in r["positive_queries"] if axis is None or p["axis"] == axis]
            if not keep:
                continue
            off = [p["query"] for p in r["positive_queries"] if p not in keep]
            out.append({**r, "positive_queries": keep,
                        "excluded_queries": sorted(set(r["excluded_queries"]) | set(off)),
                        "axis": axis or "pooled"})
        if not args.report_only:
            json.dump(out, open(path, "w"), ensure_ascii=False, indent=1)
        written.append((path.name, axis or "pooled", out))
        return out

    emit(out_path, None)
    # The axis split is a REPORTING device; a training file wants every positive
    # it can get, so the per-axis files are only written for held-out splits.
    if args.from_split != "train":
        emit(out_path.with_name(out_path.stem + "-axisS.json"), "S")
        emit(out_path.with_name(out_path.stem + "-axisC.json"), "C")

    # The pool is what the evaluator actually ranks: the union of every labelled
    # query across the file (build_global_corpus). That is NOT the same as the
    # set of judgeable queries -- a judgeable query only reaches the pool if some
    # anchor labelled it -- so it is measured here rather than assumed.
    corpus = set()
    for r in records:
        corpus.add(r["anchor_query"])
        corpus.update(p["query"] for p in r["positive_queries"])
        corpus.update(n["query"] for n in r["negative_queries"])
    missed = set(judged) - corpus
    print(f"\ncandidate pool: {len(corpus)} unique queries "
          f"(report this next to any MRR from these files)")
    print(f"  of {len(judged)} judgeable queries; {len(missed)} never got labelled "
          f"by any anchor and so are absent from the ranking")
    if missed:
        print(f"  -> raise --easy-negs/--hard-negs to pull them in, or accept the "
              f"smaller pool and report it")
    print(f"positives by axis: S={axis_tot['S']}  C={axis_tot['C']}")
    print(f"dropped anchors: {dict(drop)}")
    print(f"\n{'file':<44}{'axis':>7}{'anchors':>9}{'pos':>7}{'pos/anc':>9}"
          f"{'excl/anc':>10}")
    for name, axis, out in written:
        if not out:
            print(f"{name:<44}{axis:>7}{0:>9}{'-':>7}{'-':>9}{'-':>10}")
            continue
        np_ = [len(r["positive_queries"]) for r in out]
        ne = [len(r["excluded_queries"]) for r in out]
        print(f"{name:<44}{axis:>7}{len(out):>9}{sum(np_):>7}"
              f"{sum(np_)/len(out):>9.2f}{sum(ne)/len(out):>10.1f}")
    nh = sum(len([n for n in r['negative_queries'] if n['hardness'] == 'hard'])
             for r in records)
    print(f"\nhard negatives: {nh} ({nh/len(records):.1f}/anchor), "
          f"easy: {sum(len(r['negative_queries']) for r in records) - nh}")
    if args.report_only:
        print("\n--report-only: nothing written")


if __name__ == "__main__":
    main()
