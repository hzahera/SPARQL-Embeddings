#!/usr/bin/env python3
"""Partition the result-equivalence graph by CONNECTED COMPONENT.

Why the anchor-group split had to go
------------------------------------
`splits_dbpedia.json` partitions anchor *groups* 3,770/471/472. That cannot be
leak-free, and no amount of post-hoc filtering makes it so. Measure the corpus:

    4,713 groups, 39,520 positives, and **100% of those positives are
    themselves anchors** of other groups. Only 14 non-anchor queries exist.

So the dataset is not 4,713 independent items. It is ONE symmetric
result-equivalence relation over a closed set of 4,713 queries. Splitting the
groups puts query *q* in train (as an anchor) and in test (as a positive of some
held-out anchor *a*), which restates the training pair {a, q} at evaluation
time. That is exactly the 84.2% pair leakage measured on the sets these splits
replaced, and `build_clean_rq1.py`'s R3 rule is a post-hoc repair for it: it
deletes the leaked pairs afterwards, shrinking the split and leaving the
awkward framing "we filtered the leakage out".

Partitioning CONNECTED COMPONENTS of that relation removes the problem at the
source. If every query of a component lands in the same split, then no
anchor-positive pair can cross a split boundary -- not because they were
filtered, but because a crossing pair would be an edge between two components,
which by definition does not exist. R3 becomes unnecessary rather than
successful, and the script asserts the property instead of assuming it.

The graph is well-conditioned for this: 1,118 components, none a singleton,
largest 161 queries (3.4% of the corpus), so an 80/10/10 pack is comfortable.

Stratifying the pack
--------------------
Component packing by query count alone is not enough. Result-equivalence in
SQuARE-DBpedia arises two ways, and they measure different things:

  * **Axis S (structural).** Different `structural_template`, same answers --
    one query is a rewrite of the other. A fact about the QUERY, and the thing
    "result-aware embedding" should mean.
  * **Axis C (constant).** Same template, different constants that happen to
    share an extension. A fact about the GRAPH: no encoder can infer it from
    query text except through world knowledge about the constants.

Axis-S anchors are scarce (522 of 4,713 once single-row coincidences are
excluded) and clustered in a few large components, so a size-only pack strands
them: the naive greedy puts 474 in train and leaves 28 for test. This script
therefore packs on three quantities at once -- queries, axis-S anchors, axis-C
anchors -- always assigning the next component to whichever split is furthest
below its target in *relative* terms.

Single-row answer sets are excluded from both axes (`--min-rows`, default 2):
`answer_jaccard > 0.8` between two one-row results is not a test of set overlap,
it is two singletons coinciding. 57.3% of same-template and 34.8% of
different-template oracle-covered pairs are of that form.

Usage
-----
    python scripts/build_component_splits.py --kg dbpedia \
        --result-sets datasets/result_sets_dbpedia_v2.json \
        --out datasets/splits_dbpedia_components.json

`--result-sets` is optional: without it the axes cannot be measured and the pack
falls back to balancing query count only (it will say so loudly). With an oracle
covering only part of the corpus the axis counts are lower bounds, which biases
the stratification but never the leak-freeness -- that holds unconditionally.
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
                "results": "result_sets_dbpedia.json",
                "out": "splits_dbpedia_components.json"},
    "wikidata": {"triplets": "wikidata-triplets-dataset.json",
                 "results": "result_sets_wikidata.json",
                 "out": "splits_wikidata_components.json"},
}


def _q(entry):
    if isinstance(entry, dict):
        return str(entry.get("query") or entry.get("sparql") or "")
    return str(entry or "")


class UnionFind:
    def __init__(self):
        self.parent = {}

    def find(self, x):
        self.parent.setdefault(x, x)
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:            # path compression
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb


def load_oracle(path):
    """{normalized_query: n_rows}, excluding truncated sets. Streams if large.

    A query whose answer set hit `build_result_sets.py --max-rows` is stored as a
    prefix and listed in the `<stem>_meta.json` sidecar. Its row count is a
    floor, not a size, so counting it toward an axis would classify an anchor on
    evidence that does not exist.
    """
    if not path.exists():
        return None
    meta = path.with_name(path.stem + "_meta.json")
    bad = ({normalize_query(q) for q in json.load(open(meta)).get("truncated", [])}
           if meta.exists() else set())
    sizes = {}
    if path.stat().st_size > 512 * 1024 * 1024:
        try:
            import ijson
        except ImportError:
            raise SystemExit("pip install ijson -- needed to stream the large oracle.")
        with open(path, "rb") as f:
            for k, v in ijson.kvitems(f, ""):
                nk = normalize_query(k)
                if nk not in bad:
                    sizes[nk] = len(v)
    else:
        for k, v in json.load(open(path)).items():
            nk = normalize_query(k)
            if nk not in bad:
                sizes[nk] = len(v)
    if bad:
        print(f"  dropped {len(bad)} truncated answer sets (per {meta.name})")
    return sizes


def classify_anchor(item, sizes, min_rows):
    """(has_axis_S, has_axis_C) for one triplet group.

    An axis is credited only when BOTH sides of the pair are resolvable in the
    oracle and return >= min_rows. An unresolvable pair is not evidence of
    anything, so it is counted toward neither axis rather than guessed.
    """
    anchor = item.get("anchor_query") or ""
    na = normalize_query(anchor)
    if sizes is None:
        # No oracle: fall back to structure alone, so the pack still separates
        # rewrite-bearing components from purely parametric ones.
        ta = structural_template(anchor)
        has_s = any(structural_template(_q(p)) != ta
                    for p in item.get("positive_queries", []) if _q(p))
        return has_s, not has_s
    if sizes.get(na, 0) < min_rows:
        return False, False
    ta = structural_template(anchor)
    has_s = has_c = False
    for p in item.get("positive_queries", []):
        pq = _q(p)
        if not pq or sizes.get(normalize_query(pq), 0) < min_rows:
            continue
        if structural_template(pq) != ta:
            has_s = True
        else:
            has_c = True
    return has_s, has_c


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kg", choices=sorted(KG), default="dbpedia")
    ap.add_argument("--triplets", default=None,
                    help="corpus to partition, overriding the KG default. "
                         "Required for a derived corpus such as "
                         "dbpedia-rewrites-triplets.json -- passing only "
                         "--result-sets silently partitions the DEFAULT "
                         "corpus and produces a split of the wrong data.")
    ap.add_argument("--result-sets", default=None,
                    help="execution oracle path (default: the KG's "
                         "result_sets_*.json under datasets/). Without it the "
                         "axis stratification degrades -- see the docstring.")
    ap.add_argument("--min-rows", type=int, default=2,
                    help="both sides of a pair must return >= this many rows to "
                         "count toward an axis (default 2: excludes single-row "
                         "coincidences, which are not set overlap)")
    ap.add_argument("--ratios", default="0.8,0.1,0.1",
                    help="train,valid,test fractions (default 0.8,0.1,0.1)")
    ap.add_argument("--seed", type=int, default=42,
                    help="recorded in the output and used only to break ties "
                         "deterministically; the pack itself is not random")
    ap.add_argument("--max-sweeps", type=int, default=50,
                    help="cap on local-refinement sweeps (default 50; it "
                         "normally converges in a handful)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--report-only", action="store_true")
    args = ap.parse_args()

    cfg = KG[args.kg]
    trip_path = Path(args.triplets) if args.triplets else ROOT / "datasets" / cfg["triplets"]
    if not trip_path.is_absolute():
        trip_path = ROOT / trip_path
    trip = json.load(open(trip_path))
    print(f"corpus: {trip_path.name} ({len(trip)} groups)")
    ratios = [float(x) for x in args.ratios.split(",")]
    if len(ratios) != 3 or abs(sum(ratios) - 1.0) > 1e-6:
        raise SystemExit(f"--ratios must be three fractions summing to 1, got {args.ratios}")

    orc_path = Path(args.result_sets) if args.result_sets else ROOT / "datasets" / cfg["results"]
    if not orc_path.is_absolute():
        orc_path = ROOT / orc_path
    sizes = load_oracle(orc_path)
    if sizes is None:
        print(f"WARNING: no oracle at {orc_path}. Axes are approximated from "
              f"structure alone and single-row coincidences cannot be excluded. "
              f"Leak-freeness is unaffected.")
    else:
        print(f"oracle: {orc_path.name} ({len(sizes)} queries)")

    # ── 1. the result-equivalence graph ───────────────────────────────────────
    uf = UnionFind()
    edges = 0
    for item in trip:
        a = item.get("anchor_query")
        if not a:
            continue
        na = normalize_query(a)
        uf.find(na)
        for p in item.get("positive_queries", []):
            pq = _q(p)
            if pq:
                uf.union(na, normalize_query(pq))
                edges += 1

    members = collections.defaultdict(set)
    for q in list(uf.parent):
        members[uf.find(q)].add(q)
    print(f"equivalence graph: {len(uf.parent)} queries, {edges} edges, "
          f"{len(members)} components "
          f"(largest {max(len(v) for v in members.values())})")

    # ── 2. per-component payload ──────────────────────────────────────────────
    comp_groups = collections.defaultdict(list)     # root -> [group index]
    n_s = collections.Counter()
    n_c = collections.Counter()
    for i, item in enumerate(trip):
        a = item.get("anchor_query")
        if not a:
            continue
        root = uf.find(normalize_query(a))
        comp_groups[root].append(i)
        has_s, has_c = classify_anchor(item, sizes, args.min_rows)
        n_s[root] += int(has_s)
        n_c[root] += int(has_c)

    tot = {"queries": len(uf.parent),
           "s": sum(n_s.values()),
           "c": sum(n_c.values())}
    print(f"axis-S anchors: {tot['s']}   axis-C anchors: {tot['c']}   "
          f"(min-rows {args.min_rows})")
    if tot["s"] == 0:
        print("WARNING: zero axis-S anchors -- stratification reduces to size only.")

    # ── 3. stratified pack: greedy seed + local refinement ────────────────────
    # Axis-S is badly lumpy -- on DBpedia the largest single component holds 22%
    # of all axis-S anchors and the top three hold 39%. A one-pass greedy cannot
    # recover from an early misplacement (a component assigned first sees every
    # bin equally empty), so it is only the seed; a local search then moves
    # components one at a time while that reduces the objective.
    #
    # The objective is SQUARED RELATIVE DEVIATION, summed over bins and
    # quantities. Squaring matters: the deficit-maximising rule this replaced was
    # blind to overshoot, so a bin that had already blown its axis-S budget kept
    # looking attractive on the quantities it was still short of.
    bins = ["train", "valid", "test"]
    target = {b: {k: tot[k] * r for k in tot} for b, r in zip(bins, ratios)}
    weight = {"queries": 1.0, "s": 3.0, "c": 1.5}

    def cost(cur):
        return sum(weight[k] * ((cur[b][k] - target[b][k]) / max(target[b][k], 1.0)) ** 2
                   for b in bins for k in tot)

    def tiebreak(root):
        return hashlib.sha1(f"{args.seed}:{root}".encode()).hexdigest()

    payload = {r: {"queries": len(members[r]), "s": n_s[r], "c": n_c[r]}
               for r in members}
    # Place the scarcest components first: a component with many axis-S anchors
    # has the fewest viable homes, so it must choose before the filler does.
    order = sorted(members, key=lambda r: (-n_s[r], -n_c[r], -len(members[r]), tiebreak(r)))

    cur = {b: {k: 0 for k in tot} for b in bins}
    assign = {}
    for root in order:
        best, best_cost = None, None
        for b in bins:
            for k in tot:
                cur[b][k] += payload[root][k]
            trial = cost(cur)
            for k in tot:
                cur[b][k] -= payload[root][k]
            if best_cost is None or trial < best_cost - 1e-12:
                best, best_cost = b, trial
        assign[root] = best
        for k in tot:
            cur[best][k] += payload[root][k]

    # Refinement: sweep components in a fixed order, moving any whose relocation
    # strictly improves the objective. Converges quickly and is deterministic.
    for sweep in range(args.max_sweeps):
        moved = 0
        for root in order:
            src = assign[root]
            base = cost(cur)
            best, best_cost = src, base
            for b in bins:
                if b == src:
                    continue
                for k in tot:
                    cur[src][k] -= payload[root][k]
                    cur[b][k] += payload[root][k]
                trial = cost(cur)
                for k in tot:
                    cur[src][k] += payload[root][k]
                    cur[b][k] -= payload[root][k]
                if trial < best_cost - 1e-12:
                    best, best_cost = b, trial
            if best != src:
                for k in tot:
                    cur[src][k] -= payload[root][k]
                    cur[best][k] += payload[root][k]
                assign[root] = best
                moved += 1
        if not moved:
            break
    print(f"pack: converged after {sweep} refinement sweeps, "
          f"objective {cost(cur):.4f}")

    # ── 4. emit, in the shape the trainers already read ───────────────────────
    idx = {b: sorted(i for r, gs in comp_groups.items() if assign[r] == b for i in gs)
           for b in bins}

    # ── 5. the property this whole script exists for ──────────────────────────
    split_of = {}
    for root, b in assign.items():
        for q in members[root]:
            split_of[q] = b
    crossing = 0
    for item in trip:
        a = item.get("anchor_query")
        if not a:
            continue
        sa = split_of[normalize_query(a)]
        for p in item.get("positive_queries", []):
            pq = _q(p)
            if pq and split_of[normalize_query(pq)] != sa:
                crossing += 1
    print(f"\ncross-split anchor-positive pairs: {crossing} (must be 0)")
    assert crossing == 0, "component pack leaked a pair -- the graph build is wrong"

    print(f"{'split':<8}{'groups':>8}{'queries':>9}{'axis-S':>8}{'axis-C':>8}"
          f"{'comps':>7}")
    for b in bins:
        ncomp = sum(1 for r in assign if assign[r] == b)
        print(f"{b:<8}{len(idx[b]):>8}{cur[b]['queries']:>9}{cur[b]['s']:>8}"
              f"{cur[b]['c']:>8}{ncomp:>7}")
    print(f"{'target':<8}{'':>8}"
          + "".join(f"{target['test'][k]:>9.0f}" if k == 'queries'
                    else f"{target['test'][k]:>8.0f}" for k in ("queries", "s", "c"))
          + "   (test)")

    out = {
        "seed": args.seed,
        "train_idx": idx["train"],
        "valid_idx": idx["valid"],
        "test_idx": idx["test"],
        "partition": "connected-component of the result-equivalence graph",
        "cross_split_pairs": crossing,
        "min_rows": args.min_rows,
        "ratios": dict(zip(bins, ratios)),
        "oracle": orc_path.name if sizes is not None else None,
        "n_components": {b: sum(1 for r in assign if assign[r] == b) for b in bins},
        "axis_counts": {b: {"queries": cur[b]["queries"], "S": cur[b]["s"],
                            "C": cur[b]["c"]} for b in bins},
        "component_of_group": {str(i): assign[uf.find(normalize_query(trip[i]["anchor_query"]))]
                               for i in range(len(trip)) if trip[i].get("anchor_query")},
    }
    out_path = Path(args.out) if args.out else ROOT / "datasets" / cfg["out"]
    if args.report_only:
        print(f"\n--report-only: not writing {out_path}")
    else:
        json.dump(out, open(out_path, "w"), indent=1)
        print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
