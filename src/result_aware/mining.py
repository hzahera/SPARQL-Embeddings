from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

from .data import QueryRecord
from .model import HFEmbeddingModel
from .utils import normalize_query, write_json


@torch.inference_mode()
def mine_hard_negatives(
    model: HFEmbeddingModel,
    records: list[QueryRecord],
    top_k: int = 8,
    encode_batch_size: int = 64,
    save_path: str | Path | None = None,
) -> dict[int, list[str]]:
    """Mine hard negatives only from each record's labeled negative pool."""
    unique: dict[str, str] = {}
    for rec in records:
        for q in [rec.anchor_query, *(c.query for c in rec.negative_queries)]:
            key = normalize_query(q)
            unique.setdefault(key, q)

    keys = list(unique)
    texts = [unique[k] for k in keys]
    emb = model.encode(texts, batch_size=encode_batch_size, show_progress=True)
    index = {k: i for i, k in enumerate(keys)}

    pools: dict[int, list[str]] = {}
    audit: list[dict[str, Any]] = []
    for record_idx, rec in enumerate(tqdm(records, desc="Scoring labeled negatives")):
        a_key = normalize_query(rec.anchor_query)
        a = emb[index[a_key]]
        scored: list[tuple[float, str, float]] = []
        for cand in rec.negative_queries:
            c = emb[index[normalize_query(cand.query)]]
            score = float(torch.dot(a, c).item())
            scored.append((score, cand.query, cand.answer_jaccard))
        scored.sort(key=lambda x: x[0], reverse=True)
        chosen = scored[: min(top_k, len(scored))]
        pools[record_idx] = [x[1] for x in chosen]
        audit.append(
            {
                "record_index": record_idx,
                "anchor_query": rec.anchor_query,
                "top_negatives": [
                    {"query": q, "cosine": s, "answer_jaccard": j}
                    for s, q, j in chosen
                ],
            }
        )

    if save_path is not None:
        write_json(save_path, {"top_k": top_k, "records": audit})
    return pools
