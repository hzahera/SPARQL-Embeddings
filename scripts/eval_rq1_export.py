#!/usr/bin/env python3
"""RQ1 evaluation with PER-ANCHOR score export, for significance testing.

evaluate_rq1.py (notebooks/evaluation-RQ1/) prints only aggregate metrics.
Paired significance testing (see scripts/compute_significance.py) needs the
per-anchor score underneath each aggregate, so this script reuses evaluate_rq1's
data-loading and ranking functions and adds per-anchor MRR/nDCG export instead
of just the mean.

Usage:
    python eval_rq1_export.py   --kg wikidata   --model-type peft-qwen3   --model-name ./qwen_ft/qwen3_square_wikidata/best_adapter   --ablation baseline-qwen3ft   --out ../results/significance/wikidata_baseline_qwen3ft.json


    python eval_rq1_export.py   --kg dbpedia   --model-type peft-qwen3   --model-name ./qwen_ft/qwen3_square_dbpedia/best_adapter   --ablation baseline-qwen3ft   --out ../results/significance/dbpedia_baseline_qwen3ft.json

--model-type peft-qwen3 loads a fine-tuned Qwen3-Embedding LoRA adapter
correctly (base model via AutoModel + PeftModel.from_pretrained + the
last-token pooling the model was trained with). Do NOT load a bare adapter
directory with --model-type sentence-transformer: a LoRA adapter directory
only contains adapter_config.json + adapter weights, no modules.json, so
sentence-transformers' SentenceTransformer(path) falls back to
_load_auto_model(), which always builds a fresh MEAN-pooling wrapper
(see its own source/logs: "No sentence-transformers model found ... Creating
a new one with MEAN pooling.") -- silently mismatching the last-token
pooling used everywhere else for this model, without raising an error.
--model-type sentence-transformer remains correct for genuine standalone
HF/ST models (E5, zero-shot Qwen3, MiniLM, MPNet).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
EVAL_DIR = ROOT_DIR / "notebooks" / "evaluation-RQ1"
for p in (ROOT_DIR, EVAL_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from evaluate_rq1 import (  # noqa: E402
    load_triplets, build_global_corpus, build_eval_items,
    build_binary_relevance_maps, build_emb_map_sparqlce,
    build_emb_map_sentence_transformer, build_sorted_pools_global, dcg,
)
from sparqlce.embedder import SPARQLEmbedder  # noqa: E402

PEFT_QWEN3_BASE = "Qwen/Qwen3-Embedding-0.6B"
PEFT_QWEN3_MAX_LENGTH = 512


def _peft_qwen3_last_token_pool(last_hidden_states, attention_mask):
    """Last-token pooling as used to train the fine-tuned Qwen3 adapters
    (notebooks/finetuning/0{4,5}_qwen3_contrastive_finetune_square*.ipynb).
    Handles both left- and right-padded batches."""
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    seq_lens = attention_mask.sum(dim=1) - 1
    idx = torch.arange(last_hidden_states.shape[0], device=last_hidden_states.device)
    return last_hidden_states[idx, seq_lens]


def _gc_collect_and_empty_cache():
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def build_emb_map_peft_qwen3(adapter_path, queries, batch_size=8,
                              max_length=PEFT_QWEN3_MAX_LENGTH,
                              base_model=PEFT_QWEN3_BASE):
    """Encode queries with a fine-tuned Qwen3-Embedding LoRA adapter,
    reconstructing the exact training-time architecture: base model +
    PEFT adapter + last-token pooling + L2 normalization. See module
    docstring for why SentenceTransformer(adapter_path) is NOT equivalent.
    """
    from transformers import AutoModel, AutoTokenizer
    from peft import PeftModel
    import torch.nn.functional as F

    adapter_path = str(adapter_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(adapter_path, padding_side="left")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = AutoModel.from_pretrained(base_model, torch_dtype=dtype)
    model = PeftModel.from_pretrained(base, adapter_path, is_trainable=False)
    model = model.to(device)
    model.eval()

    emb_map = {}
    with torch.inference_mode():
        for i in range(0, len(queries), batch_size):
            batch = queries[i:i + batch_size]
            tok = tokenizer(batch, padding=True, truncation=True,
                             max_length=max_length, return_tensors="pt")
            tok = {k: v.to(device) for k, v in tok.items()}
            out = model(input_ids=tok["input_ids"], attention_mask=tok["attention_mask"])
            pooled = _peft_qwen3_last_token_pool(out.last_hidden_state, tok["attention_mask"])
            embs = F.normalize(pooled.float(), p=2, dim=1).cpu().numpy()
            for q, e in zip(batch, embs):
                emb_map[q] = e

    del model, base
    _gc_collect_and_empty_cache()
    return emb_map


KG_TESTSETS = {
    "dbpedia": ROOT_DIR / "datasets" / "dbpedia_testset_RQ1.json",
    "wikidata": ROOT_DIR / "datasets" / "wikidata_testset_RQ1.json",
}

TOPKS = [3, 5, 10, 20]


def per_anchor_mrr(sorted_pools, items, k):
    out = []
    for sorted_pool, (_, _, positives) in zip(sorted_pools, items):
        if not positives:
            continue
        rr = 0.0
        for rank, q in enumerate(sorted_pool[:k], start=1):
            if q in positives:
                rr = 1.0 / rank
                break
        out.append(rr)
    return out


def per_anchor_ndcg(sorted_pools, rel_maps, k):
    out = []
    for sorted_pool, rel_map in zip(sorted_pools, rel_maps):
        topk = [q for q in sorted_pool[:k] if q in rel_map]
        if not topk:
            out.append(0.0)
            continue
        rels = [rel_map[q] for q in topk]
        dcg_val = dcg(rels)
        ideal_rels = sorted(rel_map.values(), reverse=True)[:k]
        idcg = dcg(ideal_rels)
        out.append((dcg_val / idcg) if idcg > 0 else 0.0)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kg", required=True, choices=["dbpedia", "wikidata"])
    ap.add_argument("--model-type", default="sparqlce",
                     choices=["sparqlce", "sentence-transformer", "peft-qwen3"])
    ap.add_argument("--checkpoint", help="required for --model-type sparqlce")
    ap.add_argument("--model-name",
                     help="HF model id (sentence-transformer) or adapter dir (peft-qwen3)")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--ablation", default="full", help="tag for the run, e.g. full/no_gnn/baseline-e5")
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args()

    raw = load_triplets(str(KG_TESTSETS[args.kg]))
    global_corpus, seen_q = build_global_corpus(raw)
    eval_items = build_eval_items(raw, global_corpus, seen_q)
    rel_maps = build_binary_relevance_maps(eval_items, global_corpus)

    if args.model_type == "sparqlce":
        assert args.checkpoint, "--checkpoint required for --model-type sparqlce"
        model = SPARQLEmbedder(args.checkpoint)
        emb_map = build_emb_map_sparqlce(model, global_corpus, batch_size=args.batch_size)
    elif args.model_type == "peft-qwen3":
        assert args.model_name, "--model-name (adapter dir) required for --model-type peft-qwen3"
        emb_map = build_emb_map_peft_qwen3(args.model_name, global_corpus,
                                            batch_size=args.batch_size or 8)
    else:
        from sentence_transformers import SentenceTransformer
        assert args.model_name, "--model-name required for --model-type sentence-transformer"
        model_path = Path(args.model_name)
        if (model_path / "adapter_config.json").exists():
            raise ValueError(
                f"{args.model_name} is a bare PEFT adapter directory (has "
                "adapter_config.json), not a standalone sentence-transformers model. "
                "Loading it with --model-type sentence-transformer silently falls back "
                "to mean pooling (see module docstring) instead of the last-token "
                "pooling this model was trained with. Use --model-type peft-qwen3 "
                "instead."
            )
        model = SentenceTransformer(args.model_name)
        emb_map = build_emb_map_sentence_transformer(model, global_corpus, batch_size=args.batch_size)
    sorted_pools = build_sorted_pools_global(emb_map, eval_items, global_corpus)

    anchors = [a for a, _, _ in eval_items]
    per_anchor = {}
    aggregate = {}
    for k in TOPKS:
        mrr_list = per_anchor_mrr(sorted_pools, eval_items, k)
        ndcg_list = per_anchor_ndcg(sorted_pools, rel_maps, k)
        per_anchor[f"mrr@{k}"] = mrr_list
        per_anchor[f"ndcg@{k}"] = ndcg_list
        aggregate[f"mrr@{k}"] = float(np.mean(mrr_list))
        aggregate[f"ndcg@{k}"] = float(np.mean(ndcg_list))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({
            "kg": args.kg, "model_type": args.model_type,
            "model_id": args.checkpoint or args.model_name, "seed": args.seed,
            "ablation": args.ablation, "anchors": anchors,
            "aggregate": aggregate, "per_anchor": per_anchor,
        }, f, indent=2)
    print(f"Saved per-anchor scores for {len(anchors)} anchors to {out_path}")
    print(f"Aggregate: {json.dumps(aggregate, indent=2)}")


if __name__ == "__main__":
    main()
