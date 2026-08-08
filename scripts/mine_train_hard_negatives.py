#!/usr/bin/env python3
"""Mine real hard negatives for the TRAINING groups of an index split.

Why
---
`dbpedia-triplets-dataset.json` ships 20 negatives per group drawn from just
**24 distinct queries** across the whole 4,713-group corpus, none of them
lexically related to their anchors. Nothing in that training signal asks either
model to separate two queries that *look* alike but return different answers --
which is the entire distinction "result-aware embedding" claims to make. Both
trainers consume `negative_queries` (SparqlCE under `--hard-negatives`,
finetune_e5_baseline.py one per anchor), so both are equally starved by it.

With a full execution oracle the real negatives are cheap to find: take queries
sharing the anchor's `structural_template`, keep the ones whose answer set is
provably disjoint, and they are near-identical as text and verifiably
result-different. On DBpedia 81.5% of same-template pairs qualify.

What this does NOT touch
------------------------
Only `negative_queries`, and only for groups in the training and validation
splits. Positives are untouched, so the eval split's leak-freeness argument is
unchanged: `build_clean_rq1.py`'s R3 rule filters *positive* pairs, and it was
computed against exactly these groups' positives. Test groups are left
byte-identical so a rebuilt eval split is unaffected.

Record order and count are preserved, so the integer indices in
`splits_{kg}.json` still address the same queries and
`train_sparqlce.py`'s alignment assertion still passes.

Negatives are drawn only from queries that appear in the TRAINING groups, never
from test-only queries -- mining the whole corpus would put held-out query text
into training as a side effect.

Usage
-----
    python scripts/mine_train_hard_negatives.py --kg dbpedia \
        --result-sets datasets/result_sets_dbpedia_v2.json \
        --out datasets/dbpedia-triplets-hardneg.json

Then train against it with `--data`, keeping `--splits` as-is.
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
                "splits": "splits_dbpedia.json",
                "results": "result_sets_dbpedia_v2.json"},
    "wikidata": {"triplets": "wikidata-triplets-dataset.json",
                 "splits": "splits_wikidata.json",
                 "results": "result_sets_wikidata_v2.json"},
}


def _q(entry):
    if isinstance(entry, dict):
        return str(entry.get("query") or entry.get("sparql") or "")
    return str(entry or "")


def load_oracle(path):
    """{normalized_query: frozenset(rows)}, minus truncated answer sets.

    A truncated set is a prefix, so it cannot prove disjointness -- two queries
    might share rows that were cut off. Such queries are dropped rather than
    treated as usable.
    """
    if not path.exists():
        raise SystemExit(
            f"{path} not found. Build it with:\n"
            f"  python scripts/build_result_sets.py --kg <kg> "
            f"--split datasets/<kg>-triplets-dataset.json --out {path}")
    meta = path.with_name(path.stem + "_meta.json")
    bad = ({normalize_query(q) for q in json.load(open(meta)).get("truncated", [])}
           if meta.exists() else set())
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
        print(f"  dropped {len(bad)} truncated answer sets (per {meta.name})")
    return out


def _pick(candidates, anchor, exclude, n):
    """Up to `n` candidates, from a stable per-anchor offset.

    Rotating rather than taking the head is what stops every anchor being handed
    the same few queries -- which is the defect being fixed here. hashlib, not
    hash(), because the latter is salted per process.
    """
    if n <= 0 or not candidates:
        return []
    seq = list(candidates)
    off = int(hashlib.sha1(anchor.encode()).hexdigest(), 16) % len(seq)
    out = []
    for i in range(len(seq)):
        c = seq[(off + i) % len(seq)]
        if c != anchor and c not in exclude:
            out.append(c)
            if len(out) == n:
                break
    return out


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kg", choices=sorted(KG), default="dbpedia")
    ap.add_argument("--result-sets", default=None)
    ap.add_argument("--splits", default=None)
    ap.add_argument("--hard-negs", type=int, default=20,
                    help="same-template, result-disjoint negatives per anchor")
    ap.add_argument("--easy-negs", type=int, default=10,
                    help="different-template disjoint negatives, so the batch is "
                         "not exclusively adversarial")
    ap.add_argument("--keep-original", action="store_true",
                    help="append the shipped negatives instead of replacing them")
    ap.add_argument("--out", default=None)
    ap.add_argument("--report-only", action="store_true")
    args = ap.parse_args()

    cfg = KG[args.kg]
    trip = json.load(open(ROOT / "datasets" / cfg["triplets"]))
    spl_path = Path(args.splits) if args.splits else ROOT / "datasets" / cfg["splits"]
    if not spl_path.is_absolute():
        spl_path = ROOT / spl_path
    spl = json.load(open(spl_path))
    orc_path = Path(args.result_sets) if args.result_sets else ROOT / "datasets" / cfg["results"]
    if not orc_path.is_absolute():
        orc_path = ROOT / orc_path
    oracle = load_oracle(orc_path)
    print(f"oracle {orc_path.name}: {len(oracle)} queries")

    train_ids = set(spl["train_idx"]) | set(spl.get("valid_idx", []))
    print(f"[{args.kg}] enriching {len(train_ids)} train+valid groups "
          f"of {len(trip)} (test groups untouched)")

    # ── the universe negatives may be drawn from ──────────────────────────────
    # Training-group queries only. The corpus at large includes test-only
    # queries, and putting those in front of the model during training -- even
    # as negatives -- is held-out text leaking into the training run.
    universe = []
    seen = set()
    for i in sorted(train_ids):
        item = trip[i]
        for q in [item.get("anchor_query")] + [_q(p) for p in item.get("positive_queries", [])]:
            nq = normalize_query(q or "")
            if nq and nq not in seen and nq in oracle:
                seen.add(nq)
                universe.append(nq)
    print(f"[{args.kg}] candidate universe: {len(universe)} resolvable training queries")

    tmpl = {q: structural_template(q) for q in universe}
    by_tmpl = collections.defaultdict(list)
    for q in universe:
        by_tmpl[tmpl[q]].append(q)
    multi = sum(1 for v in by_tmpl.values() if len(v) > 1)
    print(f"[{args.kg}] {len(by_tmpl)} templates, {multi} with >1 query "
          f"(only those can yield same-template negatives)")

    # Inverted index: a candidate sharing no answer row with the anchor is
    # disjoint by construction, so only the postings need examining.
    postings = collections.defaultdict(list)
    for qi, q in enumerate(universe):
        for row in oracle[q]:
            postings[row].append(qi)

    out = [dict(item) for item in trip]        # shallow copy; untouched groups stay identical
    stats = collections.Counter()
    n_hard, n_easy, n_anchor = 0, 0, 0
    distinct_negs = set()

    for i in sorted(train_ids):
        item = trip[i]
        a_raw = item.get("anchor_query")
        if not a_raw:
            stats["no_anchor"] += 1
            continue
        a = normalize_query(a_raw)
        if a not in oracle:
            stats["anchor_unresolvable"] += 1
            continue

        touched = set()
        for row in oracle[a]:
            for qi in postings.get(row, ()):
                touched.add(universe[qi])
        # Positives must never be offered as negatives even if the oracle
        # disagrees with the shipped label.
        blocked = touched | {normalize_query(_q(p)) for p in item.get("positive_queries", [])}

        ta = tmpl.get(a) or structural_template(a_raw)
        hard = _pick(by_tmpl.get(ta, ()), a, blocked, args.hard_negs)
        easy = _pick([q for q in universe if tmpl[q] != ta], a, blocked, args.easy_negs)
        if not hard and not easy:
            stats["no_negatives_found"] += 1
            continue

        negs = ([{"query": q, "answer_jaccard": 0.0, "hardness": "hard"} for q in hard]
                + [{"query": q, "answer_jaccard": 0.0, "hardness": "easy"} for q in easy])
        if args.keep_original:
            negs += [dict(n) for n in item.get("negative_queries", [])]
        out[i] = {**item, "negative_queries": negs}

        n_hard += len(hard)
        n_easy += len(easy)
        n_anchor += 1
        distinct_negs.update(hard)
        distinct_negs.update(easy)
        stats["enriched"] += 1

    # ── invariants the downstream pipeline depends on ─────────────────────────
    assert len(out) == len(trip), "record count changed -- split indices would misalign"
    for i, (o, t) in enumerate(zip(out, trip)):
        assert o.get("anchor_query") == t.get("anchor_query"), f"anchor changed at {i}"
        assert o.get("positive_queries") == t.get("positive_queries"), \
            f"positives changed at {i} -- the eval split's R3 guarantee assumes they did not"
    untouched = [i for i in range(len(trip)) if i not in train_ids]
    assert all(out[i] == trip[i] for i in untouched), "a test group was modified"
    print(f"\ninvariants OK: {len(out)} records, positives unchanged, "
          f"{len(untouched)} test groups byte-identical")

    orig_negs = {normalize_query(_q(n)) for it in trip
                 for n in it.get("negative_queries", []) if _q(n)}
    print(f"enriched {n_anchor} groups | drops: {dict(stats) if stats else '{}'}")
    print(f"hard {n_hard} ({n_hard/max(n_anchor,1):.1f}/anchor)  "
          f"easy {n_easy} ({n_easy/max(n_anchor,1):.1f}/anchor)")
    print(f"distinct negative queries: {len(distinct_negs)} "
          f"(shipped corpus had {len(orig_negs)})")

    if args.report_only:
        print("\n--report-only: nothing written")
        return
    out_path = Path(args.out) if args.out else \
        ROOT / "datasets" / f"{args.kg}-triplets-hardneg.json"
    json.dump(out, open(out_path, "w"), ensure_ascii=False)
    print(f"\nwrote {out_path}")
    # os.path.relpath, not Path.is_relative_to: the latter is 3.9+ and this
    # runs on 3.7 in places.
    import os
    print(f"train with:  --data {os.path.relpath(out_path, ROOT)}")


if __name__ == "__main__":
    main()
