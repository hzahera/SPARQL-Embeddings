from result_aware.metrics import (
    hits_at_k,
    mrr_at_k,
    ndcg_at_k,
    precision_recall_f1_at_k,
    spearman_alignment,
)


def test_binary_ranking_metrics():
    rel = [False, True, False, True]
    assert mrr_at_k(rel, 3) == 0.5
    assert hits_at_k(rel, 1) == 0.0
    assert hits_at_k(rel, 3) == 1.0
    p, r, f1 = precision_recall_f1_at_k(rel, 2)
    assert p == 0.5
    assert r == 0.5
    assert abs(f1 - 0.5) < 1e-12


def test_ndcg_prefers_ideal_order():
    ideal = [1.0, 0.9, 0.1]
    bad = [0.1, 0.9, 1.0]
    assert abs(ndcg_at_k(ideal, 3) - 1.0) < 1e-12
    assert ndcg_at_k(bad, 3) < 1.0


def test_spearman_alignment():
    rho = spearman_alignment([0.1, 0.2, 0.3, 0.4], [0.0, 0.5, 0.8, 1.0])
    assert abs(rho - 1.0) < 1e-12
