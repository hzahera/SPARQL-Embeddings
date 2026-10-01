from result_aware.data import (
    DynamicTrainingDataset,
    parse_record,
    validate_records,
    validate_split_disjointness,
)


def rec(anchor, pos_prefix="p", neg_prefix="n"):
    return parse_record(
        {
            "anchor_query": anchor,
            "positive_queries": [
                {"query": f"{pos_prefix}{i}", "answer_jaccard": 0.8 + i * 0.01}
                for i in range(3)
            ],
            "negative_queries": [
                {"query": f"{neg_prefix}{i}", "answer_jaccard": 0.0}
                for i in range(8)
            ],
        }
    )


def test_validate_records_and_split_disjointness():
    train = [rec("A", "tp", "tn")]
    valid = [rec("B", "vp", "vn")]
    test = [rec("C", "xp", "xn")]
    assert validate_records(train)["passed"]
    report = validate_split_disjointness(train, valid, test)
    assert report["passed"]
    assert report["train_test_overlap"] == 0


def test_dynamic_sampler_uses_three_hard_one_random_for_four_negatives():
    r = rec("A", "p", "n")
    ds = DynamicTrainingDataset([r], seed=42, explicit_negatives=4, hard_fraction=0.75)
    hard = {0: ["n0", "n1", "n2", "n3"]}
    ds.set_epoch(2, hard)
    item = ds[0]
    assert len(item["negatives"]) == 4
    assert len(set(item["negatives"])) == 4
    assert len(set(item["negatives"]) & set(hard[0])) >= 3
