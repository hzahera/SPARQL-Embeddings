from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from torch.utils.data import Dataset

from .utils import normalize_query, read_json


@dataclass(frozen=True)
class Candidate:
    query: str
    answer_jaccard: float


@dataclass(frozen=True)
class QueryRecord:
    anchor_query: str
    positive_queries: tuple[Candidate, ...]
    negative_queries: tuple[Candidate, ...]

    @property
    def candidates(self) -> tuple[Candidate, ...]:
        return self.positive_queries + self.negative_queries

    @property
    def positive_query_set(self) -> set[str]:
        return {normalize_query(c.query) for c in self.positive_queries}


def _parse_candidate(item: Any, default_jaccard: float) -> Candidate:
    if isinstance(item, str):
        return Candidate(query=item, answer_jaccard=default_jaccard)
    if isinstance(item, dict):
        q = item.get("query") or item.get("candidate_query")
        if q is None:
            raise ValueError(f"Candidate object has no query field: {item}")
        j = item.get("answer_jaccard", item.get("jaccard", default_jaccard))
        return Candidate(query=str(q), answer_jaccard=float(j))
    raise TypeError(f"Unsupported candidate type: {type(item)}")


def parse_record(raw: dict[str, Any]) -> QueryRecord:
    anchor = raw.get("anchor_query") or raw.get("anchor")
    if anchor is None:
        raise ValueError(f"Record missing anchor_query: {raw.keys()}")
    positives = tuple(_parse_candidate(x, 1.0) for x in raw.get("positive_queries", []))
    negatives = tuple(_parse_candidate(x, 0.0) for x in raw.get("negative_queries", []))
    return QueryRecord(str(anchor), positives, negatives)


def load_records(path: str | Path) -> list[QueryRecord]:
    raw = read_json(path)
    if isinstance(raw, dict):
        # Tolerate wrappers such as {"records": [...]}.
        for key in ("records", "data", "items"):
            if isinstance(raw.get(key), list):
                raw = raw[key]
                break
    if not isinstance(raw, list):
        raise TypeError(f"Expected a JSON list in {path}, got {type(raw)}")
    return [parse_record(x) for x in raw]


def unique_queries(records: Iterable[QueryRecord]) -> list[str]:
    # Preserve first-seen surface form. Dataset split construction should already
    # make normalized identities unique, but this keeps encoding deterministic.
    seen: set[str] = set()
    out: list[str] = []
    for rec in records:
        for q in [rec.anchor_query, *(c.query for c in rec.candidates)]:
            key = normalize_query(q)
            if key not in seen:
                seen.add(key)
                out.append(q)
    return out


def normalized_query_set(records: Iterable[QueryRecord]) -> set[str]:
    return {normalize_query(q) for q in unique_queries(records)}


def validate_records(
    records: list[QueryRecord],
    positive_min: float = 0.8,
    negative_max: float = 0.1,
) -> dict[str, Any]:
    duplicate_anchors: list[str] = []
    seen_anchors: set[str] = set()
    missing_positive = 0
    missing_negative = 0
    positive_threshold_violations = 0
    negative_threshold_violations = 0
    self_candidates = 0
    duplicate_candidates = 0

    for rec in records:
        a = normalize_query(rec.anchor_query)
        if a in seen_anchors:
            duplicate_anchors.append(rec.anchor_query)
        seen_anchors.add(a)
        if not rec.positive_queries:
            missing_positive += 1
        if not rec.negative_queries:
            missing_negative += 1

        local_seen: set[str] = set()
        for c in rec.positive_queries:
            if c.answer_jaccard < positive_min:
                positive_threshold_violations += 1
            key = normalize_query(c.query)
            if key == a:
                self_candidates += 1
            if key in local_seen:
                duplicate_candidates += 1
            local_seen.add(key)
        for c in rec.negative_queries:
            if c.answer_jaccard > negative_max:
                negative_threshold_violations += 1
            key = normalize_query(c.query)
            if key == a:
                self_candidates += 1
            if key in local_seen:
                duplicate_candidates += 1
            local_seen.add(key)

    return {
        "records": len(records),
        "unique_queries": len(normalized_query_set(records)),
        "duplicate_anchor_count": len(duplicate_anchors),
        "missing_positive_records": missing_positive,
        "missing_negative_records": missing_negative,
        "positive_threshold_violations": positive_threshold_violations,
        "negative_threshold_violations": negative_threshold_violations,
        "self_candidates": self_candidates,
        "duplicate_candidates_within_anchor": duplicate_candidates,
        "passed": not any(
            [
                duplicate_anchors,
                missing_positive,
                missing_negative,
                positive_threshold_violations,
                negative_threshold_violations,
                self_candidates,
                duplicate_candidates,
            ]
        ),
    }


def validate_split_disjointness(
    train: list[QueryRecord], valid: list[QueryRecord], test: list[QueryRecord]
) -> dict[str, Any]:
    sets = {
        "train": normalized_query_set(train),
        "valid": normalized_query_set(valid),
        "test": normalized_query_set(test),
    }
    tv = sets["train"] & sets["valid"]
    tt = sets["train"] & sets["test"]
    vt = sets["valid"] & sets["test"]
    return {
        "normalized_query_counts": {k: len(v) for k, v in sets.items()},
        "train_valid_overlap": len(tv),
        "train_test_overlap": len(tt),
        "valid_test_overlap": len(vt),
        "passed": not (tv or tt or vt),
    }


class DynamicTrainingDataset(Dataset):
    """Samples one positive and N explicit negatives per anchor.

    Epoch 1 uses random negatives. Later epochs can consume a hard-negative pool
    mined from each anchor's *explicitly labeled negative candidates only*.
    """

    def __init__(
        self,
        records: list[QueryRecord],
        seed: int,
        explicit_negatives: int = 4,
        hard_fraction: float = 0.75,
    ) -> None:
        self.records = records
        self.seed = int(seed)
        self.explicit_negatives = int(explicit_negatives)
        self.hard_fraction = float(hard_fraction)
        self.epoch = 1
        self.hard_pools: dict[int, list[str]] = {}

    def __len__(self) -> int:
        return len(self.records)

    def set_epoch(self, epoch: int, hard_pools: dict[int, list[str]] | None = None) -> None:
        self.epoch = int(epoch)
        if hard_pools is not None:
            self.hard_pools = hard_pools

    def _rng(self, index: int):
        import random

        # Stable across reruns and independent of DataLoader order.
        return random.Random(self.seed * 1_000_003 + self.epoch * 10_007 + index)

    def __getitem__(self, index: int) -> dict[str, Any]:
        rec = self.records[index]
        if not rec.positive_queries:
            raise ValueError(f"Training record {index} has no positives")
        if len(rec.negative_queries) < self.explicit_negatives:
            raise ValueError(
                f"Training record {index} has only {len(rec.negative_queries)} negatives, "
                f"needs {self.explicit_negatives}"
            )

        rng = self._rng(index)
        positive = rng.choice(rec.positive_queries).query
        all_negatives = [c.query for c in rec.negative_queries]

        hard = list(dict.fromkeys(self.hard_pools.get(index, [])))
        hard = [q for q in hard if q in all_negatives]
        n_hard = min(round(self.explicit_negatives * self.hard_fraction), len(hard))
        selected_hard = rng.sample(hard, n_hard) if n_hard else []

        remaining = [q for q in all_negatives if q not in selected_hard]
        n_random = self.explicit_negatives - len(selected_hard)
        selected_random = rng.sample(remaining, n_random)
        negatives = selected_hard + selected_random
        rng.shuffle(negatives)

        return {
            "record_index": index,
            "anchor": rec.anchor_query,
            "positive": positive,
            "negatives": negatives,
            "known_positives": [c.query for c in rec.positive_queries],
        }


def collate_training_batch(items: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "record_indices": [x["record_index"] for x in items],
        "anchors": [x["anchor"] for x in items],
        "positives": [x["positive"] for x in items],
        "negatives": [x["negatives"] for x in items],
        "known_positives": [x["known_positives"] for x in items],
    }
