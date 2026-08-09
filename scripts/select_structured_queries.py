#!/usr/bin/env python3
"""Select SPARQL queries whose STRUCTURE the graph encoder can actually use.

Why this exists
---------------
The DBpedia half of SQuARE cannot show a graph encoder doing anything, and the
reason is measurable rather than speculative. Over 4,713 anchors:

    triple patterns   1: 1,957   2: 2,142   3+: 628   (of 4,727 unique)
    GROUP BY / COUNT / BIND / aggregates      0.0%
    no join vertex at all                     49.1%
    graphs with diameter <= 1, clique joins   38.2%   (Wikidata: 4.2%)

A second GATv2 layer has nothing to propagate over a diameter-1 graph, and
71.6% of the result-equivalent pairs are the same template with a different
constant -- a fact about the graph's contents, not about the query. Every
configuration we trained landed within noise of a plain text encoder, which is
what that profile predicts.

This script selects a corpus that does not have those properties, so the
hypothesis can be tested rather than argued about. It filters on the real
`SPARQLGraphBuilder`/`DBpediaGraphBuilder` output -- the same graphs the model
trains on -- not on a proxy.

What it does NOT do
-------------------
Execute anything, or build pairs. Structure is decided from the query text;
whether two queries are result-equivalent needs the execution oracle, which is
`build_result_sets.py`'s job. This writes a manifest that script accepts
directly, so the pipeline is:

    select_structured_queries.py  ->  manifest
    build_result_sets.py          ->  oracle (answer sets)
    <pair mining>                 ->  SQuARE triplets
    build_component_splits.py     ->  leak-free splits

Input
-----
Any JSON or JSONL file of queries. Recognised shapes:

    ["SELECT ...", ...]                        plain list
    [{"query": "SELECT ..."}, ...]             SQuARE / LSQ export
    [{"sparql_query": "SELECT ..."}, ...]      LC-QuAD 1.0
    [{"anchor_query": ..., "positive_queries": [...]}, ...]   this repo's triplets

Use --field to name the key explicitly when auto-detection guesses wrong.

Usage
-----
    python scripts/select_structured_queries.py \
        --in datasets/lsq2_dbpedia_dump.json \
        --out datasets/dbpedia-structured-manifest.json \
        --compare datasets/dbpedia-triplets-dataset.json
"""
import argparse
import collections
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sparqlce.sparql_utils import normalize_query, structural_template

# A query form whose answers are not a comparable SET is useless here: ASK
# returns a boolean (two ASKs agreeing is not result-equivalence), DESCRIBE and
# CONSTRUCT return graphs whose shape depends on the server. build_result_sets.py
# already rejects ASK at execution time; catching it now avoids paying for it.
_FORM_RE = re.compile(r'\b(SELECT|ASK|CONSTRUCT|DESCRIBE)\b', re.IGNORECASE)
_VAR_RE = re.compile(r'\?[A-Za-z_]\w*')
_AGG_RE = re.compile(r'\b(COUNT|SUM|AVG|MIN|MAX|GROUP_CONCAT|SAMPLE)\s*\(', re.IGNORECASE)

FEATURES = {
    "OPTIONAL":  re.compile(r'\bOPTIONAL\b', re.I),
    "UNION":     re.compile(r'\bUNION\b', re.I),
    "FILTER":    re.compile(r'\bFILTER\b', re.I),
    "GROUP BY":  re.compile(r'\bGROUP\s+BY\b', re.I),
    "ORDER BY":  re.compile(r'\bORDER\s+BY\b', re.I),
    "LIMIT":     re.compile(r'\bLIMIT\b', re.I),
    "aggregate": _AGG_RE,
    "BIND":      re.compile(r'\bBIND\b', re.I),
    "VALUES":    re.compile(r'\bVALUES\b', re.I),
    "MINUS":     re.compile(r'\bMINUS\b', re.I),
    "subquery":  re.compile(r'\{\s*SELECT\b', re.I),
}

# Property paths need the IRIs stripped first, or every "http://..." scores as a
# path operator and the feature reads 100% on any corpus.
_PATH_RE = re.compile(r'[\w>)]\s*[/|]\s*[\w<(]|[\w>)]\s*[*+?]\s*[\s.;,}\]]')


def has_property_path(q):
    body = _LITERAL_RE.sub('""', where_body(q))
    body = _IRI_RE.sub("<C>", body)
    return bool(_PATH_RE.search(body))


_IRI_RE = re.compile(r'<[^>]*>')
_LITERAL_RE = re.compile(r'"[^"]*"')


def query_form(q):
    m = _FORM_RE.search(q)
    return m.group(1).upper() if m else "UNKNOWN"


def where_body(q):
    parts = re.split(r'\bWHERE\b', q, maxsplit=1, flags=re.IGNORECASE)
    return parts[1] if len(parts) > 1 else q


def n_triple_patterns(q):
    """Rough count: statements separated by '.' or ';' inside the WHERE body.

    Deliberately approximate -- it is a filter, not a measurement, and the
    diameter check below is the one that decides anything.
    """
    body = where_body(q)
    body = re.sub(r'\bFILTER\s*\([^)]*\)', ' ', body, flags=re.I)
    stmts = [s for s in re.split(r'[.;]', body) if s.strip(" \t\n{}")]
    return len(stmts)


def join_vertices(q):
    """(count, mean_degree) over variables appearing in >1 triple pattern.

    This is LSQ 2.0's `lsqv:joinVertexCount` / `joinVertexDegreeMean` computed
    locally, so no dependency on their endpoint -- which is currently down.
    A query with no join vertex is a star of depth 1: nothing for message
    passing to carry.
    """
    body = where_body(q)
    body = re.sub(r'\bFILTER\s*\([^)]*\)', ' ', body, flags=re.I)
    stmts = [s for s in re.split(r'[.;]', body) if s.strip(" \t\n{}")]
    deg = collections.Counter()
    for s in stmts:
        for v in set(_VAR_RE.findall(s)):
            deg[v] += 1
    joins = {v: d for v, d in deg.items() if d > 1}
    mean = sum(joins.values()) / len(joins) if joins else 0.0
    return len(joins), mean


def graph_diameter(q, builder):
    """Diameter of the graph the ENCODER sees, via the real builder.

    The point of the whole exercise: a diameter-1 graph makes the second GATv2
    layer a no-op, so those queries cannot demonstrate anything about graph
    encoding however complex they look as text. Returns -1 when the query
    cannot be built.
    """
    try:
        import networkx as nx
        G = nx.Graph(builder.build_graph(q))
    except Exception:
        return -1
    if G.number_of_nodes() <= 1:
        return 0
    best = 0
    for comp in nx.connected_components(G):
        sub = G.subgraph(comp)
        if sub.number_of_nodes() > 1:
            best = max(best, nx.diameter(sub))
    return best


def load_queries(path, field=None):
    """Queries from a JSON/JSONL file, tolerating the shapes we actually meet."""
    text = Path(path).read_text(encoding="utf-8").strip()
    if text.startswith("["):
        records = json.loads(text)
    else:
        records = [json.loads(line) for line in text.splitlines() if line.strip()]

    keys = [field] if field else ["query", "sparql_query", "sparql", "text",
                                  "queryString", "anchor_query"]
    out = []
    for r in records:
        if isinstance(r, str):
            out.append(r)
            continue
        if not isinstance(r, dict):
            continue
        for k in keys:
            if k and r.get(k):
                out.append(str(r[k]))
                break
        # this repo's triplets carry more queries inside the record
        for sub in ("positive_queries", "negative_queries"):
            for e in r.get(sub, []) or []:
                q = e.get("query") if isinstance(e, dict) else e
                if q:
                    out.append(str(q))
    return out


def profile(queries, builder=None, label=""):
    """Print the same profile for any corpus, so candidates are comparable."""
    n = len(queries)
    if not n:
        print(f"  {label}: empty")
        return
    tp = collections.Counter(min(n_triple_patterns(q), 6) for q in queries)
    jv = [join_vertices(q)[0] for q in queries]
    print(f"  {label}: {n} queries, {len(set(structural_template(q) for q in queries))} templates")
    print(f"    triple patterns (capped 6): {dict(sorted(tp.items()))}")
    print(f"    join vertices: 0 -> {sum(1 for x in jv if x == 0)/n:.1%}, "
          f">=1 -> {sum(1 for x in jv if x >= 1)/n:.1%}, "
          f">=2 -> {sum(1 for x in jv if x >= 2)/n:.1%}")
    feats = {k: sum(1 for q in queries if rx.search(q)) / n for k, rx in FEATURES.items()}
    feats["path"] = sum(1 for q in queries if has_property_path(q)) / n
    print("    features: " + "  ".join(f"{k} {v:.1%}" for k, v in feats.items() if v > 0))
    if builder is not None:
        d = [graph_diameter(q, builder) for q in queries]
        ok = [x for x in d if x >= 0]
        if ok:
            print(f"    builder diameter <=1: {sum(1 for x in ok if x <= 1)/len(ok):.1%} "
                  f"(median {sorted(ok)[len(ok)//2]}); unbuildable {len(d)-len(ok)}")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="inp", required=True,
                    help="JSON/JSONL file of candidate queries")
    ap.add_argument("--field", default=None, help="query field name (auto-detected otherwise)")
    ap.add_argument("--out", default=None,
                    help="manifest for build_result_sets.py --split")
    ap.add_argument("--compare", default=None,
                    help="a second corpus to profile alongside, e.g. the current "
                         "datasets/dbpedia-triplets-dataset.json, so the selection "
                         "can be judged against what it replaces")
    ap.add_argument("--min-triples", type=int, default=2,
                    help="minimum triple patterns (default 2: 53%% of the current "
                         "DBpedia corpus has exactly 1)")
    ap.add_argument("--min-join-vertices", type=int, default=1,
                    help="minimum variables shared by >1 pattern (default 1). 0 "
                         "means a star with no join -- nothing to propagate.")
    ap.add_argument("--min-diameter", type=int, default=2,
                    help="minimum diameter of the graph the ENCODER builds "
                         "(default 2). This is the binding constraint: at <=1 a "
                         "second GATv2 layer is a no-op. Set 0 to skip (much "
                         "faster, but drops the guarantee).")
    ap.add_argument("--join-hubs", action="store_true", default=True,
                    help="build graphs with join hubs, matching --join-hubs "
                         "training (default on; clique joins collapse diameter)")
    ap.add_argument("--arch", choices=["hybrid", "dbpedia"], default="hybrid",
                    help="which builder to measure with; use the one you train with")
    ap.add_argument("--require-any", default=None,
                    help="comma-separated features, keep a query with ANY of them, "
                         "e.g. 'GROUP BY,aggregate,UNION,OPTIONAL'")
    ap.add_argument("--max-queries", type=int, default=None,
                    help="cap the output (deterministic: keeps the most connected "
                         "queries first, ties by template for diversity)")
    ap.add_argument("--report-only", action="store_true")
    args = ap.parse_args()

    raw = load_queries(args.inp, args.field)
    queries = list(dict.fromkeys(normalize_query(q) for q in raw if q and q.strip()))
    print(f"loaded {len(raw)} queries from {args.inp} -> {len(queries)} unique")

    builder = None
    if args.min_diameter > 0 or True:
        if args.arch == "dbpedia":
            from sparqlce.dbpedia import DBpediaGraphBuilder
            builder = DBpediaGraphBuilder()
        else:
            from sparqlce.graph_builder import SPARQLGraphBuilder
            builder = SPARQLGraphBuilder(join_hubs=args.join_hubs)

    print("\nBEFORE selection:")
    profile(queries, builder if args.min_diameter > 0 else None, "candidates")
    if args.compare:
        cmp_q = list(dict.fromkeys(normalize_query(q) for q in load_queries(args.compare)))
        profile(cmp_q, builder if args.min_diameter > 0 else None,
                f"current ({Path(args.compare).name})")

    want = [f.strip() for f in args.require_any.split(",")] if args.require_any else []
    for f in want:
        if f not in FEATURES and f != "path":
            raise SystemExit(f"unknown feature {f!r}; "
                             f"choose from {sorted(list(FEATURES) + ['path'])}")

    kept, drop = [], collections.Counter()
    for q in queries:
        if query_form(q) != "SELECT":
            drop[f"form_{query_form(q).lower()}"] += 1
            continue
        if n_triple_patterns(q) < args.min_triples:
            drop["too_few_triples"] += 1
            continue
        jvc, jvd = join_vertices(q)
        if jvc < args.min_join_vertices:
            drop["no_join_vertex"] += 1
            continue
        if want and not any(has_property_path(q) if f == "path"
                            else FEATURES[f].search(q) for f in want):
            drop["missing_required_feature"] += 1
            continue
        d = graph_diameter(q, builder) if args.min_diameter > 0 else 99
        if d < 0:
            drop["unbuildable"] += 1
            continue
        if d < args.min_diameter:
            drop["diameter_too_small"] += 1
            continue
        kept.append((q, jvc, jvd, d))

    print(f"\ndropped: {dict(drop) if drop else '{}'}")
    if not kept:
        raise SystemExit("nothing survived; loosen --min-diameter / --min-triples.")

    # Deterministic cap: most-connected first, but round-robin over templates so
    # one dominant family cannot fill the quota (the failure that produced a
    # corpus where one template covered 34% of anchors).
    if args.max_queries and args.max_queries < len(kept):
        by_t = collections.defaultdict(list)
        for rec in sorted(kept, key=lambda r: (-r[1], -r[3], r[0])):
            by_t[structural_template(rec[0])].append(rec)
        order, sel = sorted(by_t, key=lambda t: (-len(by_t[t]), t)), []
        i = 0
        while len(sel) < args.max_queries and any(by_t[t] for t in order):
            t = order[i % len(order)]
            if by_t[t]:
                sel.append(by_t[t].pop(0))
            i += 1
        kept = sel
        print(f"capped to {len(kept)} (round-robin over "
              f"{len({structural_template(r[0]) for r in kept})} templates)")

    sel_q = [r[0] for r in kept]
    print("\nAFTER selection:")
    profile(sel_q, builder if args.min_diameter > 0 else None, "selected")

    if args.report_only:
        print("\n--report-only: nothing written")
        return
    out_path = Path(args.out) if args.out else ROOT / "datasets" / "structured-manifest.json"
    # build_result_sets.py reads {anchor_query, positive_queries, negative_queries}
    # records and unions every query it finds, so this shape feeds it directly.
    json.dump([{"anchor_query": q, "positive_queries": [], "negative_queries": []}
               for q in sel_q], open(out_path, "w"), ensure_ascii=False)
    print(f"\nwrote {len(sel_q)} queries -> {out_path}")
    print(f"\nnext:\n  python scripts/build_result_sets.py --kg dbpedia \\\n"
          f"    --split {out_path} \\\n"
          f"    --out datasets/result_sets_dbpedia_structured.json --pause 0.1")


if __name__ == "__main__":
    main()
