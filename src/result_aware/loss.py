from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from .utils import normalize_query


def asymmetric_infonce_loss(
    anchor_embeddings: torch.Tensor,
    positive_embeddings: torch.Tensor,
    negative_embeddings: torch.Tensor,
    anchors: list[str],
    positives: list[str],
    negatives: list[list[str]],
    known_positives: list[list[str]],
    temperature: float = 0.05,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Asymmetric InfoNCE with explicit + cross-anchor candidates.

    Candidate pool = all designated positives in the batch + every explicit
    negative in the batch. Known positives of an anchor are masked if they
    appear as cross-anchor candidates, so they never become false negatives.
    """
    if anchor_embeddings.ndim != 2:
        raise ValueError("anchor_embeddings must be [B,D]")
    batch_size = anchor_embeddings.size(0)
    if positive_embeddings.shape != anchor_embeddings.shape:
        raise ValueError("positive_embeddings must match anchor_embeddings")
    if negative_embeddings.ndim != 3 or negative_embeddings.size(0) != batch_size:
        raise ValueError("negative_embeddings must be [B,N,D]")

    candidate_embeddings = torch.cat(
        [positive_embeddings, negative_embeddings.reshape(-1, negative_embeddings.size(-1))],
        dim=0,
    )
    candidate_texts = positives + [q for row in negatives for q in row]

    logits = anchor_embeddings @ candidate_embeddings.T
    logits = logits / float(temperature)

    # Own designated positive is candidate i for anchor i.
    targets = torch.arange(batch_size, device=logits.device)

    masked = 0
    for i in range(batch_size):
        a = normalize_query(anchors[i])
        kp = {normalize_query(q) for q in known_positives[i]}
        # Also regard the designated positive as known even if the input record
        # was malformed and omitted it from known_positives.
        kp.add(normalize_query(positives[i]))
        for j, c_text in enumerate(candidate_texts):
            if j == i:
                continue
            c = normalize_query(c_text)
            if c == a or c in kp:
                logits[i, j] = torch.finfo(logits.dtype).min
                masked += 1

    loss = F.cross_entropy(logits, targets)
    with torch.no_grad():
        pred = logits.argmax(dim=1)
        acc = (pred == targets).float().mean().item()
    return loss, {
        "batch_retrieval_accuracy": acc,
        "masked_known_positive_logits": masked,
        "candidate_pool_size": candidate_embeddings.size(0),
    }
