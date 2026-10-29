#!/usr/bin/env python3
"""
Rebuild leakage-free DBpedia semantic triplets from an existing triplet dataset.

Input (either format is accepted via --input):
  unique_queries.json
    ["SELECT ...", "SELECT ...", ...]

  dbpedia-triplets-dataset.json
    [
      {
        "anchor_query": "...",
        "positive_queries": [{"query": "...", "answer_jaccard": ...}, ...],
        "negative_queries": [{"query": "...", "answer_jaccard": ...}, ...]
      },
      ...
    ]

What this script does:
  1. Reads every UNIQUE query string (from the list, or from
     anchors/positives/negatives) and drops scalar-aggregate queries.
  2. Re-executes every unique query on the configured DBpedia SPARQL endpoint.
  3. Stores a resumable execution cache.
  4. Recomputes answer-set Jaccard from the NEW endpoint results.
  5. Builds positive-connected components so positive/result-similar queries can
     never be split across train/validation/test.
  6. Creates leakage-free train/validation/test query partitions.
  7. Rebuilds anchor -> positives/negatives INSIDE each partition only.
  8. Runs hard assertions for exact query overlap, normalized-query overlap,
     result-equivalent leakage, and positive-edge leakage.
  9. Writes:
       dbpedia-train-clean.json
       dbpedia-valid-clean.json
       dbpedia-test-clean.json
       split_report.json
       unique_queries.json
       query_execution_cache.jsonl

Default labels reproduce the original mining idea:
  positive: Jaccard >= 0.80
  negative: Jaccard <= 0.10

Requirements:
  pip install requests tqdm

Pipeline position (see datasets/DATASET.md):
  Stage 1  01_query_collection_dbpedia.ipynb  -> sources/successful_queries_dbpedia.json
  Stage 2  02_triplet_mining_dbpedia.ipynb    -> sources/dbpedia-triplets-dataset.json
  Stage 3  this script                    -> unique_queries.json, query_execution_cache.jsonl,
                                              dbpedia-{train,valid,test}-clean.json, split_report.json

How to run (from datasets/dbpedia-splits/):
  Full pipeline, from the Stage 2 output:
    python ../scripts/rebuild_dbpedia_clean_splits.py --input sources/dbpedia-triplets-dataset.json --output-dir .
  Reproduce the released dataset from the frozen query list and the cached answers
  (no endpoint access needed when the cache is present):
    python ../scripts/rebuild_dbpedia_clean_splits.py --input unique_queries.json --output-dir <new_dir>
    (copy query_execution_cache.jsonl into <new_dir> first)
  The released split files are the *-clean.json outputs renamed to dbpedia-{train,valid,test}.json.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import statistics
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import requests
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Query normalization / hashing
# ---------------------------------------------------------------------------

def normalize_query(q: str) -> str:
    """
    Conservative normalization used ONLY for duplicate/leakage detection.
    It does not rewrite SPARQL semantics.
    """
    q = q.strip()
    q = re.sub(r"\s+", " ", q)
    return q


def query_hash(q: str) -> str:
    return hashlib.sha256(normalize_query(q).encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Scalar-aggregate filter
# ---------------------------------------------------------------------------

_AGG = re.compile(r"\b(COUNT|SUM|AVG|MIN|MAX|SAMPLE|GROUP_CONCAT)\s*\(", re.I)
_GROUP_BY = re.compile(r"\bGROUP\s+BY\b", re.I)


def is_scalar_aggregate(q: str) -> bool:
    """
    True for SELECT queries that project an aggregate (in the outer query or a
    sub-SELECT) and contain no GROUP BY, i.e. queries that return a single
    aggregate row such as SELECT (COUNT(?x) AS ?n) WHERE { ... }.

    Answer-set Jaccard is not meaningful for these queries: two unrelated
    queries become result-equivalent (J = 1) whenever their single values
    coincide. They are therefore excluded before execution.
    """
    # Drop IRIs first: they may contain '#', '{' or keywords such as COUNT.
    body = re.sub(r"<[^<>\s]*>", "<>", q)
    if _GROUP_BY.search(body):
        return False
    # Check the projection of the outer query and of any sub-SELECT, so that
    # a scalar aggregate wrapped in an outer SELECT is also caught.
    for m in re.finditer(r"\bSELECT\b(.*?)(\bWHERE\b|\{)", body, re.I | re.S):
        if _AGG.search(m.group(1)):
            return True
    return False


# ---------------------------------------------------------------------------
# Input extraction
# ---------------------------------------------------------------------------

def extract_unique_queries(input_path: Path) -> List[str]:
    with input_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(
            "Expected the input JSON root to be a list of query strings "
            "(unique_queries.json) or of anchor/positive/negative records."
        )

    by_norm: Dict[str, str] = {}

    def add(q: Any) -> None:
        if not isinstance(q, str) or not q.strip():
            return
        n = normalize_query(q)
        by_norm.setdefault(n, q.strip())

    for item in data:
        # unique_queries.json: a plain list of query strings.
        if isinstance(item, str):
            add(item)
            continue
        # Triplet dataset: anchor/positive/negative records.
        if not isinstance(item, dict):
            continue

        add(item.get("anchor_query"))

        for key in ("positive_queries", "negative_queries"):
            for candidate in item.get(key, []) or []:
                if isinstance(candidate, str):
                    add(candidate)
                elif isinstance(candidate, dict):
                    add(candidate.get("query"))

    return list(by_norm.values())


# ---------------------------------------------------------------------------
# SPARQL execution
# ---------------------------------------------------------------------------

def canonical_cell(binding: Dict[str, Any]) -> Tuple[str, str, str, str]:
    """
    Preserve RDF value identity more carefully than value-only comparison.
    """
    return (
        str(binding.get("type", "")),
        str(binding.get("value", "")),
        str(binding.get("xml:lang", binding.get("lang", ""))),
        str(binding.get("datatype", "")),
    )


def canonical_row(binding: Dict[str, Any], vars_: Sequence[str]) -> str:
    """
    Canonical result row. Variable NAMES are not stored, so simple variable
    renaming does not automatically make result rows different. Position follows
    the endpoint's projected variable order.
    """
    row = []
    for v in vars_:
        if v in binding:
            row.append(canonical_cell(binding[v]))
        else:
            row.append(("", "", "", ""))
    return json.dumps(row, ensure_ascii=False, separators=(",", ":"))


@dataclass
class ExecutionResult:
    query: str
    query_hash: str
    status: str
    answers: List[str]
    n_answers: int
    error: Optional[str] = None
    elapsed_s: Optional[float] = None


def execute_one(
    query: str,
    endpoint: str,
    timeout_s: int,
    retries: int,
    user_agent: str,
) -> ExecutionResult:
    qh = query_hash(query)

    headers = {
        "Accept": "application/sparql-results+json",
        "User-Agent": user_agent,
    }

    last_error = None

    for attempt in range(retries + 1):
        started = time.time()
        try:
            # GET is convenient and reproducible for this public endpoint.
            # If a query is too long, retry using POST.
            params = {
                "query": query,
                "format": "application/sparql-results+json",
            }

            r = requests.get(
                endpoint,
                params=params,
                headers=headers,
                timeout=timeout_s,
            )

            if r.status_code in (414, 431):
                r = requests.post(
                    endpoint,
                    data=params,
                    headers=headers,
                    timeout=timeout_s,
                )

            r.raise_for_status()
            payload = r.json()

            bindings = payload.get("results", {}).get("bindings")
            vars_ = payload.get("head", {}).get("vars", [])

            if bindings is None:
                return ExecutionResult(
                    query=query,
                    query_hash=qh,
                    status="unsupported_non_select",
                    answers=[],
                    n_answers=0,
                    error="Response does not contain results.bindings",
                    elapsed_s=time.time() - started,
                )

            answers = sorted(
                {
                    canonical_row(binding, vars_)
                    for binding in bindings
                }
            )

            return ExecutionResult(
                query=query,
                query_hash=qh,
                status="success" if answers else "empty",
                answers=answers,
                n_answers=len(answers),
                elapsed_s=time.time() - started,
            )

        except Exception as e:
            last_error = f"{type(e).__name__}: {e}"
            if attempt < retries:
                # Mild exponential backoff; do not hammer the endpoint.
                time.sleep(min(2 ** attempt, 8))

    return ExecutionResult(
        query=query,
        query_hash=qh,
        status="error",
        answers=[],
        n_answers=0,
        error=last_error,
    )


def load_execution_cache(path: Path) -> Dict[str, ExecutionResult]:
    cache: Dict[str, ExecutionResult] = {}
    if not path.exists():
        return cache

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            cache[obj["query_hash"]] = ExecutionResult(**obj)
    return cache


def append_execution_cache(path: Path, result: ExecutionResult) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(result.__dict__, ensure_ascii=False) + "\n")
        f.flush()


def execute_all(
    queries: Sequence[str],
    cache_path: Path,
    endpoint: str,
    workers: int,
    timeout_s: int,
    retries: int,
    user_agent: str,
) -> Dict[str, ExecutionResult]:
    cache = load_execution_cache(cache_path)

    todo = [q for q in queries if query_hash(q) not in cache]
    print(f"Execution cache already contains: {len(cache)} queries")
    print(f"Queries still to execute:         {len(todo)}")

    if not todo:
        return cache

    # Endpoint-friendly concurrency. Increase only if the endpoint tolerates it.
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {
            ex.submit(
                execute_one,
                q,
                endpoint,
                timeout_s,
                retries,
                user_agent,
            ): q
            for q in todo
        }

        for fut in tqdm(as_completed(futures), total=len(futures), desc="Executing"):
            result = fut.result()
            # Main thread writes the cache, avoiding concurrent file writes.
            append_execution_cache(cache_path, result)
            cache[result.query_hash] = result

    return cache


# ---------------------------------------------------------------------------
# Jaccard and efficient overlap discovery
# ---------------------------------------------------------------------------

def jaccard(a: Set[str], b: Set[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    inter = len(a & b)
    return inter / (len(a) + len(b) - inter)


def build_shared_answer_intersections(
    answer_sets: Sequence[Set[str]],
) -> Dict[Tuple[int, int], int]:
    """
    Efficiently count intersections only for query pairs that share >=1 row.
    Pairs absent from the returned map have Jaccard = 0.

    This avoids an O(N^2) scan just to discover positive-overlap pairs.
    """
    inverted: Dict[str, List[int]] = defaultdict(list)

    for qid, answers in enumerate(answer_sets):
        for answer in answers:
            inverted[answer].append(qid)

    intersections: Dict[Tuple[int, int], int] = defaultdict(int)

    for ids in tqdm(inverted.values(), desc="Counting shared answers"):
        if len(ids) < 2:
            continue
        ids = sorted(set(ids))
        for x in range(len(ids)):
            i = ids[x]
            for y in range(x + 1, len(ids)):
                j = ids[y]
                intersections[(i, j)] += 1

    return dict(intersections)


def jaccard_from_intersection(
    i: int,
    j: int,
    intersections: Dict[Tuple[int, int], int],
    sizes: Sequence[int],
) -> float:
    if i == j:
        return 1.0
    a, b = (i, j) if i < j else (j, i)
    inter = intersections.get((a, b), 0)
    if inter == 0:
        return 0.0
    union = sizes[a] + sizes[b] - inter
    return inter / union if union else 1.0


# ---------------------------------------------------------------------------
# DSU / leakage-free positive components
# ---------------------------------------------------------------------------

class DSU:
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1


def build_positive_edges_and_components(
    answer_sets: Sequence[Set[str]],
    intersections: Dict[Tuple[int, int], int],
    positive_min: float,
) -> Tuple[List[Tuple[int, int, float]], List[List[int]]]:
    sizes = [len(x) for x in answer_sets]
    dsu = DSU(len(answer_sets))
    positive_edges: List[Tuple[int, int, float]] = []

    # A positive Jaccard > 0 must share at least one answer, therefore we only
    # need to inspect pairs present in intersections.
    for (i, j), inter in tqdm(intersections.items(), desc="Finding positives"):
        union = sizes[i] + sizes[j] - inter
        jac = inter / union if union else 1.0
        if jac >= positive_min:
            positive_edges.append((i, j, jac))
            dsu.union(i, j)

    groups: Dict[int, List[int]] = defaultdict(list)
    for i in range(len(answer_sets)):
        groups[dsu.find(i)].append(i)

    components = list(groups.values())
    return positive_edges, components


# ---------------------------------------------------------------------------
# Component-disjoint split
# ---------------------------------------------------------------------------

def split_components(
    components: Sequence[Sequence[int]],
    train_ratio: float,
    valid_ratio: float,
    test_ratio: float,
    seed: int,
) -> Dict[str, Set[int]]:
    if not math.isclose(train_ratio + valid_ratio + test_ratio, 1.0, abs_tol=1e-9):
        raise ValueError("train + valid + test ratios must sum to 1.0")

    rng = random.Random(seed)

    comps = [list(c) for c in components]
    rng.shuffle(comps)
    # Largest first makes the greedy balancing more stable.
    comps.sort(key=len, reverse=True)

    total = sum(len(c) for c in comps)
    targets = {
        "train": total * train_ratio,
        "valid": total * valid_ratio,
        "test": total * test_ratio,
    }
    splits: Dict[str, Set[int]] = {
        "train": set(),
        "valid": set(),
        "test": set(),
    }

    for comp in comps:
        # Put this entire positive component into the split with the greatest
        # remaining relative capacity.
        def score(name: str) -> float:
            target = max(targets[name], 1.0)
            remaining = targets[name] - len(splits[name])
            return remaining / target

        destination = max(("train", "valid", "test"), key=score)
        splits[destination].update(comp)

    return splits


# ---------------------------------------------------------------------------
# Rebuild triplets inside each split
# ---------------------------------------------------------------------------

def make_positive_adjacency(
    n: int,
    positive_edges: Sequence[Tuple[int, int, float]],
) -> Dict[int, List[Tuple[int, float]]]:
    adj: Dict[int, List[Tuple[int, float]]] = defaultdict(list)
    for i, j, jac in positive_edges:
        adj[i].append((j, jac))
        adj[j].append((i, jac))
    return adj


def build_split_records(
    split_ids: Set[int],
    queries: Sequence[str],
    positive_adj: Dict[int, List[Tuple[int, float]]],
    intersections: Dict[Tuple[int, int], int],
    answer_sizes: Sequence[int],
    negative_max: float,
    max_positives: int,
    max_negatives: int,
    negative_sample_trials: int,
    seed: int,
) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    ids = sorted(split_ids)
    id_set = set(ids)

    records: List[Dict[str, Any]] = []

    for anchor_id in tqdm(ids, desc="Building split records"):
        positives = [
            (qid, jac)
            for qid, jac in positive_adj.get(anchor_id, [])
            if qid in id_set
        ]
        positives.sort(key=lambda x: (-x[1], query_hash(queries[x[0]])))

        if max_positives > 0:
            positives = positives[:max_positives]

        # Anchor records without a positive cannot train/evaluate contrastively.
        if not positives:
            continue

        positive_ids = {qid for qid, _ in positives}

        # Prefer overlapping negatives first (J > 0 but <= threshold), because
        # these are more informative than completely disjoint negatives.
        overlapping_negatives: List[Tuple[int, float]] = []

        for other_id in ids:
            if other_id == anchor_id or other_id in positive_ids:
                continue
            a, b = sorted((anchor_id, other_id))
            if (a, b) not in intersections:
                continue
            jac = jaccard_from_intersection(
                anchor_id, other_id, intersections, answer_sizes
            )
            if jac <= negative_max:
                overlapping_negatives.append((other_id, jac))

        # Higher-Jaccard legal negatives first = harder result-space negatives.
        overlapping_negatives.sort(
            key=lambda x: (-x[1], query_hash(queries[x[0]]))
        )

        negatives: List[Tuple[int, float]] = overlapping_negatives[:max_negatives]
        chosen = {qid for qid, _ in negatives}

        # Fill remaining slots with J=0 negatives, sampled only from this split.
        need = max_negatives - len(negatives)
        if need > 0:
            candidates = ids[:]
            rng.shuffle(candidates)

            trials = 0
            for other_id in candidates:
                if need <= 0 or trials >= negative_sample_trials:
                    break
                trials += 1

                if (
                    other_id == anchor_id
                    or other_id in positive_ids
                    or other_id in chosen
                ):
                    continue

                a, b = sorted((anchor_id, other_id))
                # No shared answer -> exactly J=0.
                if (a, b) not in intersections:
                    negatives.append((other_id, 0.0))
                    chosen.add(other_id)
                    need -= 1

        if not negatives:
            continue

        records.append(
            {
                "anchor_query": queries[anchor_id],
                "positive_queries": [
                    {
                        "query": queries[qid],
                        "answer_jaccard": round(float(jac), 12),
                    }
                    for qid, jac in positives
                ],
                "negative_queries": [
                    {
                        "query": queries[qid],
                        "answer_jaccard": round(float(jac), 12),
                    }
                    for qid, jac in negatives
                ],
            }
        )

    return records


# ---------------------------------------------------------------------------
# Leakage checks
# ---------------------------------------------------------------------------

def queries_in_records(records: Sequence[Dict[str, Any]]) -> Set[str]:
    result: Set[str] = set()

    for x in records:
        result.add(normalize_query(x["anchor_query"]))
        for key in ("positive_queries", "negative_queries"):
            for c in x.get(key, []):
                result.add(normalize_query(c["query"]))

    return result


def assert_no_leakage(
    splits: Dict[str, Set[int]],
    records: Dict[str, List[Dict[str, Any]]],
    positive_edges: Sequence[Tuple[int, int, float]],
    answer_sets: Sequence[Set[str]],
) -> Dict[str, Any]:
    # 1. Query-id partitions are disjoint.
    assert splits["train"].isdisjoint(splits["valid"])
    assert splits["train"].isdisjoint(splits["test"])
    assert splits["valid"].isdisjoint(splits["test"])

    # 2. Every output query string is disjoint across JSON files.
    output_sets = {k: queries_in_records(v) for k, v in records.items()}
    assert output_sets["train"].isdisjoint(output_sets["valid"])
    assert output_sets["train"].isdisjoint(output_sets["test"])
    assert output_sets["valid"].isdisjoint(output_sets["test"])

    # 3. Positive edges cannot cross split boundaries.
    owner: Dict[int, str] = {}
    for split_name, ids in splits.items():
        for qid in ids:
            assert qid not in owner
            owner[qid] = split_name

    cross_positive_edges = []
    for i, j, jac in positive_edges:
        if owner[i] != owner[j]:
            cross_positive_edges.append((i, j, jac))

    assert not cross_positive_edges, (
        f"Positive leakage detected across splits: "
        f"{cross_positive_edges[:5]}"
    )

    # 4. Exact answer sets (Jaccard=1) may not occur in different splits.
    # Fingerprint the whole result set so this check is cheap.
    fp_to_splits: Dict[str, Set[str]] = defaultdict(set)

    for split_name, ids in splits.items():
        for qid in ids:
            fp = hashlib.sha256(
                "\n".join(sorted(answer_sets[qid])).encode("utf-8")
            ).hexdigest()
            fp_to_splits[fp].add(split_name)

    exact_result_leaks = {
        fp: sorted(names)
        for fp, names in fp_to_splits.items()
        if len(names) > 1
    }
    assert not exact_result_leaks, (
        "Exact-result leakage exists across splits. "
        "This should be impossible because J=1 pairs are positive."
    )

    return {
        "normalized_output_queries": {
            k: len(v) for k, v in output_sets.items()
        },
        "train_valid_overlap": len(output_sets["train"] & output_sets["valid"]),
        "train_test_overlap": len(output_sets["train"] & output_sets["test"]),
        "valid_test_overlap": len(output_sets["valid"] & output_sets["test"]),
        "cross_split_positive_edges": len(cross_positive_edges),
        "cross_split_exact_result_fingerprints": len(exact_result_leaks),
        "passed": True,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def write_json(path: Path, obj: Any) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--input",
        default="sources/dbpedia-triplets-dataset.json",
        help="unique_queries.json (list of query strings) or the older "
        "dbpedia-triplets-dataset.json (anchor/positive/negative records).",
    )
    ap.add_argument(
        "--output-dir",
        default="dbpedia-clean-splits",
    )
    ap.add_argument(
        "--endpoint",
        default="https://dbpedia.data.dice-research.org/sparql",
    )
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--timeout", type=int, default=120)
    ap.add_argument("--retries", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)

    # Same semantic thresholds as the original miner by default.
    ap.add_argument("--positive-min", type=float, default=0.80)
    ap.add_argument("--negative-max", type=float, default=0.10)

    ap.add_argument("--train-ratio", type=float, default=0.80)
    ap.add_argument("--valid-ratio", type=float, default=0.10)
    ap.add_argument("--test-ratio", type=float, default=0.10)

    ap.add_argument(
        "--keep-scalar-aggregates",
        action="store_true",
        help="Keep single-row aggregate queries (excluded by default).",
    )
    ap.add_argument("--max-positives", type=int, default=20)
    ap.add_argument("--max-negatives", type=int, default=20)
    ap.add_argument(
        "--negative-sample-trials",
        type=int,
        default=10000,
        help="Maximum random candidates checked per anchor when filling J=0 negatives.",
    )
    ap.add_argument(
        "--user-agent",
        default="SPARQL-Embeddings-Dataset-Rebuilder/1.0",
    )
    args = ap.parse_args()

    input_path = Path(args.input)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    if (out / "unique_queries.json").resolve() == input_path.resolve():
        raise ValueError(
            "--output-dir would overwrite the input unique_queries.json; "
            "choose a different output directory."
        )

    print("=" * 80)
    print("1) Extracting UNIQUE queries from the old triplet dataset")
    print("=" * 80)

    unique_queries = extract_unique_queries(input_path)
    n_input_queries = len(unique_queries)
    print(f"Unique normalized queries: {n_input_queries}")

    excluded_scalar_aggregates: List[str] = []
    if not args.keep_scalar_aggregates:
        excluded_scalar_aggregates = [
            q for q in unique_queries if is_scalar_aggregate(q)
        ]
        unique_queries = [
            q for q in unique_queries if not is_scalar_aggregate(q)
        ]
        print(f"Excluded scalar-aggregate queries: {len(excluded_scalar_aggregates)}")
        print(f"Queries kept for execution:        {len(unique_queries)}")
    write_json(
        out / "excluded_scalar_aggregates.json",
        excluded_scalar_aggregates,
    )

    write_json(out / "unique_queries.json", unique_queries)

    print("\n" + "=" * 80)
    print("2) Re-executing unique queries on DBpedia")
    print("=" * 80)

    cache_path = out / "query_execution_cache.jsonl"
    execution_cache = execute_all(
        unique_queries,
        cache_path=cache_path,
        endpoint=args.endpoint,
        workers=args.workers,
        timeout_s=args.timeout,
        retries=args.retries,
        user_agent=args.user_agent,
    )

    # Keep only newly successful, non-empty SELECT queries.
    successful_queries: List[str] = []
    answer_sets: List[Set[str]] = []

    status_counts: Dict[str, int] = defaultdict(int)
    for q in unique_queries:
        r = execution_cache[query_hash(q)]
        status_counts[r.status] += 1
        if r.status == "success":
            successful_queries.append(q)
            answer_sets.append(set(r.answers))

    print("Execution statuses:")
    for k, v in sorted(status_counts.items()):
        print(f"  {k:24s} {v}")

    print(f"Usable non-empty queries: {len(successful_queries)}")

    if len(successful_queries) < 3:
        raise RuntimeError("Too few successful non-empty queries to build splits.")

    print("\n" + "=" * 80)
    print("3) Recomputing Jaccard relationships")
    print("=" * 80)

    intersections = build_shared_answer_intersections(answer_sets)
    answer_sizes = [len(x) for x in answer_sets]

    positive_edges, components = build_positive_edges_and_components(
        answer_sets,
        intersections,
        positive_min=args.positive_min,
    )

    print(f"Pairs with shared answers: {len(intersections)}")
    print(f"Positive edges (J >= {args.positive_min}): {len(positive_edges)}")
    print(f"Positive components: {len(components)}")
    print(f"Largest positive component: {max(map(len, components))}")
    print(f"Median positive component:  {statistics.median(map(len, components))}")

    print("\n" + "=" * 80)
    print("4) Creating component-disjoint train/valid/test splits")
    print("=" * 80)

    split_ids = split_components(
        components,
        train_ratio=args.train_ratio,
        valid_ratio=args.valid_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )

    for name in ("train", "valid", "test"):
        print(f"{name:5s}: {len(split_ids[name])} query IDs")

    positive_adj = make_positive_adjacency(
        len(successful_queries),
        positive_edges,
    )

    print("\n" + "=" * 80)
    print("5) Rebuilding anchor/positive/negative JSON records")
    print("=" * 80)

    records = {}
    seed_offsets = {"train": 0, "valid": 1, "test": 2}

    for split_name in ("train", "valid", "test"):
        print(f"\nBuilding {split_name}...")
        records[split_name] = build_split_records(
            split_ids=split_ids[split_name],
            queries=successful_queries,
            positive_adj=positive_adj,
            intersections=intersections,
            answer_sizes=answer_sizes,
            negative_max=args.negative_max,
            max_positives=args.max_positives,
            max_negatives=args.max_negatives,
            negative_sample_trials=args.negative_sample_trials,
            seed=args.seed + seed_offsets[split_name],
        )

    print("\n" + "=" * 80)
    print("6) Running leakage checks")
    print("=" * 80)

    leakage = assert_no_leakage(
        split_ids,
        records,
        positive_edges,
        answer_sets,
    )

    print(json.dumps(leakage, indent=2))
    print("LEAKAGE CHECKS: PASSED")

    train_path = out / "dbpedia-train-clean.json"
    valid_path = out / "dbpedia-valid-clean.json"
    test_path = out / "dbpedia-test-clean.json"

    write_json(train_path, records["train"])
    write_json(valid_path, records["valid"])
    write_json(test_path, records["test"])

    def split_stats(xs: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        pos = [p["answer_jaccard"] for x in xs for p in x["positive_queries"]]
        neg = [n["answer_jaccard"] for x in xs for n in x["negative_queries"]]
        return {
            "anchors": len(xs),
            "positive_relations": len(pos),
            "negative_relations": len(neg),
            "positive_jaccard": {
                "min": min(pos) if pos else None,
                "max": max(pos) if pos else None,
                "mean": sum(pos) / len(pos) if pos else None,
            },
            "negative_jaccard": {
                "min": min(neg) if neg else None,
                "max": max(neg) if neg else None,
                "mean": sum(neg) / len(neg) if neg else None,
            },
        }

    report = {
        "input": str(input_path),
        "endpoint": args.endpoint,
        "seed": args.seed,
        "thresholds": {
            "positive_min": args.positive_min,
            "negative_max": args.negative_max,
        },
        "requested_ratios": {
            "train": args.train_ratio,
            "valid": args.valid_ratio,
            "test": args.test_ratio,
        },
        "input_unique_queries_before_filter": n_input_queries,
        "excluded_scalar_aggregates": len(excluded_scalar_aggregates),
        "input_unique_queries": len(unique_queries),
        "execution_status_counts": dict(status_counts),
        "usable_nonempty_queries": len(successful_queries),
        "pairs_with_shared_answers": len(intersections),
        "positive_edges": len(positive_edges),
        "positive_components": len(components),
        "largest_positive_component": max(map(len, components)),
        "median_positive_component": statistics.median(map(len, components)),
        "non_singleton_components": sum(1 for c in components if len(c) > 1),
        "partition_query_counts_before_anchor_filter": {
            k: len(v) for k, v in split_ids.items()
        },
        "output": {
            "train": split_stats(records["train"]),
            "valid": split_stats(records["valid"]),
            "test": split_stats(records["test"]),
        },
        "leakage_checks": leakage,
        "files": {
            "train": str(train_path),
            "valid": str(valid_path),
            "test": str(test_path),
            "execution_cache": str(cache_path),
            "unique_queries": str(out / "unique_queries.json"),
        },
    }

    write_json(out / "split_report.json", report)

    print("\n" + "=" * 80)
    print("DONE")
    print("=" * 80)
    print(f"Train: {train_path}")
    print(f"Valid: {valid_path}")
    print(f"Test : {test_path}")
    print(f"Report: {out / 'split_report.json'}")


if __name__ == "__main__":
    main()
