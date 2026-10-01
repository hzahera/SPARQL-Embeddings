import torch

from result_aware.data import Candidate, QueryRecord
from result_aware.evaluation import EncodedSplit, rank_record_global
from result_aware.utils import normalize_query


def _encoded(vectors):
    return EncodedSplit(
        embeddings={normalize_query(q): torch.tensor(v, dtype=torch.float32) for q, v in vectors.items()},
        surface_forms={normalize_query(q): q for q in vectors},
    )


def test_global_retrieval_uses_queries_from_other_anchors():
    rec = QueryRecord(
        anchor_query="A",
        positive_queries=(Candidate("A+", 1.0),),
        negative_queries=(Candidate("A-", 0.0),),
    )
    encoded = _encoded({
        "A": [1.0, 0.0],
        "A+": [0.8, 0.6],
        "A-": [0.0, 1.0],
        # Simulates a query introduced by another test record; it must still be
        # retrievable for A and is unannotated/non-relevant for A.
        "B+": [0.99, 0.01],
    })
    ranked = rank_record_global(rec, encoded)
    assert [x["query"] for x in ranked][:2] == ["B+", "A+"]
    assert ranked[0]["answer_jaccard"] == 0.0
    assert ranked[1]["answer_jaccard"] == 1.0


def test_global_retrieval_removes_anchor_and_excluded_queries():
    rec = QueryRecord(
        anchor_query="A",
        positive_queries=(Candidate("A+", 1.0),),
        negative_queries=(),
        excluded_queries=("X",),
    )
    encoded = _encoded({
        "A": [1.0, 0.0],
        "X": [1.0, 0.0],
        "A+": [0.9, 0.1],
        "B": [0.0, 1.0],
    })
    ranked = rank_record_global(rec, encoded)
    queries = [x["query"] for x in ranked]
    assert "A" not in queries
    assert "X" not in queries
    assert queries[0] == "A+"
