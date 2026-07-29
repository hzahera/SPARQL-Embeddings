#!/usr/bin/env python3
"""Execute a split's queries and cache their answer sets — the execution oracle.

Why this exists
---------------
`result_sets_wikidata.json` holds only 2,229 queries (it is 6.4 GB because
individual result sets run to millions of rows, not because it is broad). Of the
448 queries in the pool of `wikidata-testsplit-RQ1.json`, 200 are absent, so
41% of anchors cannot have their relevance judgments verified or completed. That
blocks two things the RQ1 evaluation needs:

  * R1, the non-singleton rule (`build_clean_rq1.py --min-result-size`). 45% of
    the *resolvable* Wikidata queries return a single row, where Jaccard > 0.8
    is a coin flip between 1.0 and 0 rather than a test of set overlap.
  * Exhaustive relabelling (`relabel_rq1_split.py`). On DBpedia, where coverage
    is 96%, that found 556 unlabelled result-equivalent pool queries affecting
    40% of anchors — correct answers scored as errors, penalising whichever
    model is better at finding true equivalence.

Snapshot consistency
--------------------
Answer sets are only comparable when they come from ONE graph snapshot. The
cached sets predate this run by years and Wikidata has moved since, so by
default this re-executes **every** query in the pool rather than only the gaps,
and writes a fresh file instead of merging into the old one. `--only-missing`
merges into an existing oracle instead — faster, but then Jaccard is computed
across snapshots and small overlaps become untrustworthy.

Truncation
----------
Some LSQ queries return millions of rows. `--max-rows` caps what is stored, but
a truncated set makes Jaccard wrong in an unpredictable direction, so any query
hitting the cap is recorded with `"truncated": true` and consumers must treat it
as unusable rather than as a set. Raising the cap trades disk for coverage; it
does not make truncated sets safe to compare.

Resumability
------------
Results are checkpointed to `<out>.partial` every `--checkpoint-every` queries
and reloaded on restart, so an interrupted run continues. Failures (timeout,
malformed query, endpoint error) are recorded with their reason instead of being
retried forever — a query that cannot be executed must be dropped from the
split, not silently treated as empty.

Usage
-----
    # dry run: how many queries, how many already cached
    python scripts/build_result_sets.py --kg wikidata \
        --split datasets/wikidata-testsplit-RQ1.json --dry-run

    # full re-execution of the pool (recommended; hours for Wikidata)
    python scripts/build_result_sets.py --kg wikidata \
        --split datasets/wikidata-testsplit-RQ1.json \
        --out datasets/result_sets_wikidata_v2.json

Then rebuild and relabel against the new oracle:
    python scripts/build_clean_rq1.py --kg wikidata --min-result-size 5 \
        --results result_sets_wikidata_v2.json --out datasets/wikidata-testsplit-RQ1.json
    python scripts/relabel_rq1_split.py --kg wikidata \
        --split datasets/wikidata-testsplit-RQ1.json --out <...>
"""
import argparse
import collections
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Same self-hosted mirrors the rest of the pipeline uses (see
# scripts/build_wikidata_labels.py). Public endpoints rate-limit aggressively and
# will not survive a few thousand LSQ queries.
ENDPOINTS = {
    "wikidata": "https://wikidata.data.dice-research.org/sparql",
    "dbpedia": "https://dbpedia.data.dice-research.org/sparql",
}
USER_AGENT = "SPARQLCE-oracle-builder/1.0 (research; contact via repo)"


def norm(s):
    return " ".join(str(s).split())


def _q(entry):
    if isinstance(entry, dict):
        return str(entry.get("query") or entry.get("sparql") or "")
    return str(entry or "")


def pool_of(split_path):
    """Every distinct query the split ranks: anchors, positives, negatives."""
    split = json.load(open(split_path))
    pool = set()
    for r in split:
        pool.add(norm(r["anchor_query"]))
        pool.update(norm(_q(p)) for p in r.get("positive_queries", []))
        pool.update(norm(_q(n)) for n in r.get("negative_queries", []))
    return split, sorted(pool)


def row_key(binding, order):
    """One result row as a stable string, so answer sets can be set-compared.

    Sorted by variable name rather than by the endpoint's column order: two
    equivalent queries may project the same values under a different SELECT
    order, and a row's identity should not depend on that.
    """
    parts = []
    for var in order:
        cell = binding.get(var)
        parts.append("" if cell is None else f"{cell.get('type','')}:{cell.get('value','')}")
    return "\t".join(parts)


def execute(endpoint, query, timeout, max_rows, max_retries=4):
    """Run one query. Returns (rows|None, error_reason|None, truncated)."""
    url = endpoint + "?" + urllib.parse.urlencode({"query": query, "format": "json"})
    req = urllib.request.Request(
        url, headers={"User-Agent": USER_AGENT,
                      "Accept": "application/sparql-results+json"})
    delay = 2.0
    for attempt in range(max_retries):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                payload = json.load(r)
            break
        except urllib.error.HTTPError as e:
            # 400 is the endpoint rejecting the query itself -- retrying cannot
            # help, and these anchors must be dropped from the split.
            if e.code == 400:
                return None, f"http400 (endpoint rejected the query)", False
            if e.code in (429, 500, 502, 503, 504) and attempt < max_retries - 1:
                wait = float(e.headers.get("Retry-After", delay))
                time.sleep(wait)
                delay = min(delay * 2, 120)
                continue
            return None, f"http{e.code}", False
        except (urllib.error.URLError, TimeoutError) as e:
            if attempt < max_retries - 1:
                time.sleep(delay)
                delay = min(delay * 2, 120)
                continue
            return None, f"unreachable/timeout ({e})", False
        except ValueError as e:
            return None, f"malformed response ({e})", False
    else:
        return None, "exhausted retries", False

    # ASK returns a boolean, not bindings. A boolean carries no set structure --
    # two ASKs agreeing is not answer-set equivalence -- so it is not an oracle
    # entry, and is reported rather than stored as a one-element set.
    if "boolean" in payload:
        return None, "ASK query (no answer set)", False
    res = payload.get("results", {})
    bindings = res.get("bindings", [])
    order = payload.get("head", {}).get("vars", [])
    order = sorted(order)
    truncated = len(bindings) > max_rows
    rows = [row_key(b, order) for b in bindings[:max_rows]]
    return rows, None, truncated


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kg", required=True, choices=sorted(ENDPOINTS))
    ap.add_argument("--split", required=True,
                    help="file whose query pool should be executed. Any file of "
                         "{anchor_query, positive_queries, negative_queries} "
                         "records works, so pass an RQ1 split to cover just that "
                         "pool, or the KG's *-triplets-dataset.json to cover the "
                         "WHOLE corpus (4,727 queries on DBpedia) -- which is "
                         "what split construction and relabelling need.")
    ap.add_argument("--out", default=None,
                    help="output oracle path (required unless --dry-run)")
    ap.add_argument("--endpoint", default=None, help="override the SPARQL endpoint")
    ap.add_argument("--timeout", type=int, default=120, help="per-query seconds")
    ap.add_argument("--max-rows", type=int, default=100000,
                    help="cap on stored rows; queries exceeding it are marked "
                         "truncated and are NOT safe to compare (default 100000)")
    ap.add_argument("--pause", type=float, default=0.2,
                    help="seconds between queries, to be a good citizen")
    ap.add_argument("--checkpoint-every", type=int, default=25)
    ap.add_argument("--abort-after", type=int, default=25,
                    help="abort if the first N queries ALL fail (default 25) -- "
                         "catches a broken endpoint or TLS trust store before it "
                         "costs hours and yields an empty oracle")
    ap.add_argument("--only-missing", metavar="ORACLE",
                    help="execute only queries absent from this existing oracle and "
                         "merge into it. Faster, but mixes graph snapshots -- see "
                         "the module docstring.")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    endpoint = args.endpoint or ENDPOINTS[args.kg]
    split, pool = pool_of(args.split)
    print(f"split: {Path(args.split).name}  anchors={len(split)}  pool={len(pool)}")

    have = {}
    if args.only_missing:
        src = Path(args.only_missing)
        if not src.is_absolute():
            src = ROOT / "datasets" / src
        print(f"reading existing oracle {src.name} ...", flush=True)
        if src.stat().st_size > 512 * 1024 * 1024:
            import ijson
            with open(src, "rb") as f:
                for k, v in ijson.kvitems(f, ""):
                    nk = norm(k)
                    if nk in set(pool):
                        have[nk] = v
        else:
            have = {norm(k): v for k, v in json.load(open(src)).items()
                    if norm(k) in set(pool)}
        print(f"  {len(have)}/{len(pool)} pool queries already cached")
        print("  WARNING: merging re-executed results with cached ones compares "
              "answer sets across graph snapshots.")

    todo = [q for q in pool if q not in have]
    print(f"to execute: {len(todo)} queries against {endpoint}")
    if args.dry_run:
        print("--dry-run: nothing executed")
        return
    if not args.out:
        raise SystemExit("--out required unless --dry-run")

    out_path = Path(args.out)
    part = out_path.with_name(out_path.name + ".partial")
    results, failures, truncated = dict(have), {}, []
    if part.exists():
        state = json.load(open(part))
        results.update(state.get("results", {}))
        failures.update(state.get("failures", {}))
        truncated = state.get("truncated", [])
        todo = [q for q in todo if q not in results and q not in failures]
        print(f"resuming from {part.name}: {len(results)} done, "
              f"{len(failures)} failed, {len(todo)} left")

    def checkpoint():
        json.dump({"results": results, "failures": failures,
                   "truncated": truncated}, open(part, "w"))

    t0 = time.time()
    n_new = 0
    for i, q in enumerate(todo, 1):
        rows, err, trunc = execute(endpoint, q, args.timeout, args.max_rows)
        if err is not None:
            failures[q] = err
        else:
            results[q] = rows
            n_new += 1
            if trunc:
                truncated.append(q)

        # Fail fast on a broken environment. Every transport error retries with
        # exponential backoff, so a misconfiguration that fails *all* of them --
        # no CA bundle, wrong endpoint, no route -- otherwise burns hours and
        # then writes an oracle with zero answer sets, which downstream reads as
        # "every query returns nothing" rather than as a failure.
        if i >= args.abort_after and n_new == 0:
            reasons = collections.Counter(failures.values()).most_common(3)
            raise SystemExit(
                f"aborting: first {i} queries all failed against {endpoint}.\n"
                f"  most common reasons: {reasons}\n"
                f"  nothing was written. Check endpoint reachability and TLS "
                f"trust (a python.org macOS build needs "
                f"'Install Certificates.command'), then re-run.")
        if i % args.checkpoint_every == 0 or i == len(todo):
            checkpoint()
            rate = i / max(time.time() - t0, 1e-9)
            eta = (len(todo) - i) / rate if rate else 0
            print(f"  {i}/{len(todo)}  ok={len(results)} fail={len(failures)} "
                  f"trunc={len(truncated)}  {rate * 60:.1f} q/min  "
                  f"ETA {eta / 60:.0f} min", flush=True)
        time.sleep(args.pause)

    # The oracle itself is a plain {query: rows} map, so every existing consumer
    # (build_clean_rq1, relabel_rq1_split) reads it unchanged. Truncated and
    # failed queries go in a sidecar rather than being silently indistinguishable
    # from short or empty results.
    json.dump(results, open(out_path, "w"))
    side = out_path.with_name(out_path.stem + "_meta.json")
    json.dump({"endpoint": endpoint, "split": str(args.split),
               "max_rows": args.max_rows, "executed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               "n_results": len(results), "failures": failures,
               "truncated": truncated}, open(side, "w"), indent=1)
    part.unlink(missing_ok=True)

    empty = sum(1 for v in results.values() if not v)
    single = sum(1 for v in results.values() if len(v) == 1)
    print(f"\nwrote {len(results)} answer sets -> {out_path}")
    print(f"  metadata (failures, truncation) -> {side.name}")
    print(f"  coverage of pool: {len(results)}/{len(pool)} ({len(results)/len(pool):.1%})")
    print(f"  empty results: {empty} | single-row: {single} "
          f"({single/max(len(results),1):.1%}) | truncated: {len(truncated)}")
    if failures:
        from collections import Counter
        print(f"  failures: {len(failures)} -> {dict(Counter(failures.values()))}")
        print("  Anchors whose query failed must be DROPPED from the split, not "
              "treated as empty-result.")
    print(f"\nNext: rebuild with R1 enabled, then relabel exhaustively:")
    print(f"  python scripts/build_clean_rq1.py --kg {args.kg} --min-result-size 5 \\")
    print(f"    --results {out_path.name} --out datasets/{args.kg}-testsplit-RQ1.json")
    print(f"  python scripts/relabel_rq1_split.py --kg {args.kg} "
          f"--split datasets/{args.kg}-testsplit-RQ1.json --out <...>")


if __name__ == "__main__":
    main()
