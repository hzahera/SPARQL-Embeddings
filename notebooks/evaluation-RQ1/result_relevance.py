"""Result-wise (answer-set) relevance for RQ1 retrieval evaluation.

The paper's claim is about *result-equivalent* retrieval, so relevance should be
grounded in actual answer-set overlap, not hand annotation or lexical similarity.
This module replaces the binary, annotation-based relevance
(`evaluate_rq1.build_binary_relevance_maps`) with relevance computed from the
real result sets in `datasets/result_sets_wikidata.json`:

    Jaccard(anchor, cand) = |results(anchor) ∩ results(cand)|
                            / |results(anchor) ∪ results(cand)|

It returns drop-in replacements for the `eval_items` and `relevance_scores_per_item`
that the existing `evaluate_rq1` metrics already consume:

  * MRR / Hits use each eval item's positive SET  -> positives = {cand : J > tau}
    (result-equivalent by answer overlap, over the WHOLE corpus — this also
    recovers result-equivalent queries never hand-annotated for the anchor).
  * nDCG / meanRel read the relevance MAP          -> graded gain = J
    (avg_ndcg_binary already computes a graded DCG/IDCG from these values).

Design decisions (see prose):
  * tau = 0.8 (matches how the annotated positives were defined).
  * Anchors whose OWN result set is empty or missing are DROPPED (an empty
    result set would otherwise match every other empty query).
  * annotated_fallback=True: the original annotated positives are unioned into
    the result-wise positives and floored to `tau` gain, so a known
    result-equivalent is never lost if its result set is missing from the file.

Usage (in 06_eval, right after building the standard eval_items):

    from result_relevance import build_result_wise
    eval_items, relevance_scores_per_item, stats = build_result_wise(
        eval_items, global_corpus, RESULT_SETS_PATH, tau=0.8)
    print(stats)
    # then run the evaluate_* cells unchanged — every metric is now result-wise.

Streaming the 6.3 GB file takes a few minutes the first time; the needed subset
is cached next to it so re-runs are fast.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Set, Tuple

try:
    import ijson
except ImportError:  # pragma: no cover
    ijson = None


def _norm(s: str) -> str:
    return " ".join((s or "").split())


def _jaccard(a: frozenset, b: frozenset) -> float:
    if not a and not b:
        return 1.0                       # both empty: identical (empty) results
    if not a or not b:
        return 0.0
    inter = len(a & b)
    return inter / (len(a) + len(b) - inter)


def load_needed_result_sets(
    result_sets_path: str,
    needed_norm: Set[str],
    cache_path: str | None = None,
) -> Dict[str, frozenset]:
    """Return {normalised_query -> frozenset(result_rows)} for the needed queries.

    Streams the big JSON once (ijson) keeping only needed keys, then caches the
    subset as compact JSON so later runs skip the full scan.
    """
    if cache_path and os.path.exists(cache_path):
        with open(cache_path, encoding="utf-8") as f:
            raw = json.load(f)
        return {k: frozenset(v) for k, v in raw.items()}

    if ijson is None:
        raise ImportError("ijson is required to stream result_sets_wikidata.json "
                          "(pip install ijson) — or provide a prebuilt cache_path.")

    found: Dict[str, frozenset] = {}
    with open(result_sets_path, "rb") as f:
        for key, rows in ijson.kvitems(f, ""):
            nk = _norm(key)
            if nk in needed_norm and nk not in found:
                # rows is a list of JSON-string result rows; set-ify for Jaccard.
                found[nk] = frozenset(rows if isinstance(rows, list) else [])
                if len(found) == len(needed_norm):
                    break

    if cache_path:
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump({k: list(v) for k, v in found.items()}, f)
    return found


def build_result_wise(
    eval_items: List[Tuple[str, List[str], Set[str]]],
    global_corpus: List[str],
    result_sets_path: str,
    tau: float = 0.8,
    annotated_fallback: bool = True,
    cache_path: str | None = None,
):
    """Recompute eval_items + relevance maps from answer-set overlap.

    Returns (result_eval_items, result_rel_maps, stats). Both outputs are
    aligned and drop-in compatible with evaluate_rq1's report()/metric helpers.
    Anchors with an empty or missing own result set are dropped.
    """
    if cache_path is None:
        cache_path = str(Path(result_sets_path).with_suffix("")) + "_rq1subset.json"

    # Result sets we need: every anchor + every corpus candidate.
    needed = {_norm(q) for q in global_corpus}
    for a, _, _ in eval_items:
        needed.add(_norm(a))
    rs = load_needed_result_sets(result_sets_path, needed, cache_path)

    corpus_norm = [_norm(q) for q in global_corpus]
    new_items: List[Tuple[str, List[str], Set[str]]] = []
    new_maps: List[Dict[str, float]] = []
    n_dropped = 0
    n_recovered = 0        # result-equivalents beyond the original annotation
    cov_hit = sum(1 for q in corpus_norm if q in rs)

    for anchor, pool, ann_pos in eval_items:
        a_set = rs.get(_norm(anchor))
        if not a_set:                       # empty or missing anchor results -> drop
            n_dropped += 1
            continue

        rel_map: Dict[str, float] = {}
        positives: Set[str] = set()
        for cand, cnorm in zip(global_corpus, corpus_norm):
            if cand == anchor:
                continue
            c_set = rs.get(cnorm)
            j = _jaccard(a_set, c_set) if c_set is not None else 0.0
            if annotated_fallback and cand in ann_pos:
                j = max(j, tau)             # never lose a known positive
            rel_map[cand] = j
            if j > tau:
                positives.add(cand)

        if not positives:
            n_dropped += 1
            continue
        n_recovered += len(positives - ann_pos)
        new_items.append((anchor, pool, positives))
        new_maps.append(rel_map)

    stats = {
        "tau": tau,
        "anchors_in": len(eval_items),
        "anchors_out": len(new_items),
        "anchors_dropped_empty_or_no_positive": n_dropped,
        "corpus_result_set_coverage": f"{cov_hit}/{len(global_corpus)}",
        "extra_result_equivalents_recovered": n_recovered,
        "cache": cache_path,
    }
    return new_items, new_maps, stats
