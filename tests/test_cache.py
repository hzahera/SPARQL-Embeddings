from result_aware.cache_eval import (
    CacheDecisionExample,
    cache_metrics_at_threshold,
    select_cache_threshold,
)


def test_cache_precision_and_safe_savings():
    examples = [
        CacheDecisionExample("a", 0.95, 1.0, True),
        CacheDecisionExample("b", 0.90, 0.0, True),
        CacheDecisionExample("c", 0.70, 0.9, True),
    ]
    m = cache_metrics_at_threshold(examples, threshold=0.8)
    assert m["correct_hits"] == 1
    assert m["incorrect_hits"] == 1
    assert abs(m["precision"] - 0.5) < 1e-12
    assert abs(m["recall"] - (1 / 3)) < 1e-12
    assert abs(m["safe_execution_savings"] - (1 / 3)) < 1e-12


def test_threshold_selection_respects_precision_constraint():
    examples = [
        CacheDecisionExample("a", 0.95, 1.0, True),
        CacheDecisionExample("b", 0.90, 0.0, True),
        CacheDecisionExample("c", 0.85, 0.9, True),
    ]
    cfg = {
        "target_precision": 1.0,
        "threshold_min": 0.8,
        "threshold_max": 1.0,
        "threshold_steps": 21,
        "selection_objective": "hit_rate",
    }
    tau, best, _ = select_cache_threshold(examples, cfg, 0.8, "ge")
    assert best["precision"] >= 1.0
    assert tau > 0.90
