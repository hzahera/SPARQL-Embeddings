#!/usr/bin/env python3
"""Share of queries longer than 512 tokens per backbone. Run from the repo root where HF models are reachable."""
import json
from transformers import AutoTokenizer
M = {"mpnet": "sentence-transformers/all-mpnet-base-v2", "e5": "intfloat/e5-base-v2", "bge": "BAAI/bge-base-en-v1.5",
     "gte": "thenlper/gte-base", "qwen3emb": "Qwen/Qwen3-Embedding-0.6B", "gemma": "google/gemma-3-270m",
     "codebert": "microsoft/codebert-base", "graphcodebert": "microsoft/graphcodebert-base",
     "qwen25coder": "Qwen/Qwen2.5-Coder-0.5B", "spbert": "razent/spbert-mlm-base"}
for g in ("dbpedia", "wikidata"):
    qs = json.load(open(f"datasets/{g}-splits/unique_queries.json"))
    for k, m in M.items():
        tok = AutoTokenizer.from_pretrained(m); pre = "query: " if k == "e5" else ""
        n = sum(len(tok(pre + q)["input_ids"]) > 512 for q in qs)
        print(f"{g:9s} {k:14s} {n:5d}/{len(qs)} ({100 * n / len(qs):.2f}%) over 512 tokens")
