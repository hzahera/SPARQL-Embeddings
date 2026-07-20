# ---- Contamination check: does the anchor->positive text overlap inflate RQ1 numbers? ----
import json
_d = json.load(open("../../datasets/dbpedia-triplets-dataset.json"))
_splits = json.load(open("../../datasets/splits_dbpedia.json"))
_tv_idx = set(_splits["train_idx"]) | set(_splits["valid_idx"])
def _qtext(x): return x["query"] if isinstance(x, dict) else x
_positives_tv = set()
for _i in _tv_idx:
    _positives_tv.update(_qtext(p) for p in _d[_i].get("positive_queries", []))

overlap_ids = [i for i, item in enumerate(rq1_eval_items) if item["anchor_query"] in _positives_tv]
clean_ids   = [i for i, item in enumerate(rq1_eval_items) if item["anchor_query"] not in _positives_tv]
print(f"overlap anchors: {len(overlap_ids)}, clean anchors: {len(clean_ids)}")

for k in TOPKS:
    mrr_all     = avg_mrr_at_k_annotated(rq1_sorted_pools, rq1_eval_items, k)
    mrr_overlap = avg_mrr_at_k_annotated([rq1_sorted_pools[i] for i in overlap_ids], [rq1_eval_items[i] for i in overlap_ids], k)
    mrr_clean   = avg_mrr_at_k_annotated([rq1_sorted_pools[i] for i in clean_ids],   [rq1_eval_items[i] for i in clean_ids],   k)
    print(f"@{k:>2}  all={mrr_all:.4f}  overlap(n={len(overlap_ids)})={mrr_overlap:.4f}  clean(n={len(clean_ids)})={mrr_clean:.4f}")