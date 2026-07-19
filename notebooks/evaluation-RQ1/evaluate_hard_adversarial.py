#!/usr/bin/env python3
"""Hard-adversarial ranking check for RQ1.

Reads {kg}_testset_RQ1_hard.json (built by scripts/build_hard_adversarial_rq1*.py):
one row per anchor with its single hardest positive (answer-Jaccard high,
lexical cosine low -- result-equivalent but surface-divergent) and hardest
negative (lexical cosine high, answer-Jaccard low -- surface-similar but
result-different).

This is NOT a retrieval/ranking-over-a-corpus metric like evaluate_rq1.py's
MRR/Hits/nDCG. It's the sharpest, most direct statement of the paper's
thesis: for each row, does the model score cos(anchor, positive) HIGHER than
cos(anchor, negative)? A model that only captures lexical similarity should
get this backwards (score the negative higher, since it's the lexically
close one); a result-aware model (SparqlCE) should get it right despite the
lexical gap.

Reports, per model:
  - ranking_accuracy: fraction of rows where cos(a,pos) > cos(a,neg)
  - mean_margin: mean(cos(a,pos) - cos(a,neg)) -- signed; positive = correct
    on average, negative = systematically backwards
  - n: number of rows evaluated

Recommended usage: run once on all rows, and again filtered to
`meets_strict_bar: true` only (the rows that actually clear BOTH thresholds
-- see datasets/README.md for how few that is per KG). The strict-only
number is the one to headline in the paper; the full-500 number is a noisier
extension.
"""
import json
import os
from typing import Callable, Dict, List, Optional

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize as sk_normalize
from rank_bm25 import BM25Okapi

from evaluate_rq1 import (
    build_emb_map_sentence_transformer,
    build_emb_map_sparqlce,
    build_emb_map_code_llm,
    SPARQL_ENCODER_BASELINES,
)

# ----------------------------
# Data loading
# ----------------------------

def load_hard_testset(path: str, strict_only: bool = False) -> List[Dict]:
    rows = json.load(open(path, "r", encoding="utf-8"))
    if strict_only:
        rows = [r for r in rows if r.get("meets_strict_bar")]
    return rows


def _unique_texts(rows: List[Dict]) -> List[str]:
    texts = set()
    for r in rows:
        texts.add(r["anchor_query"])
        texts.add(r["positive_query"])
        texts.add(r["negative_query"])
    return list(texts)


# ----------------------------
# Core metric
# ----------------------------

def report_ranking(name: str, rows: List[Dict], score_fn: Callable[[str, str], float]) -> Dict:
    """score_fn(a, b) -> similarity/relevance score, higher = more similar.
    Works uniformly for embedding cosine, TF-IDF cosine, or BM25 score."""
    correct = 0
    margins = []
    skipped = 0
    for r in rows:
        a, p, n = r["anchor_query"], r["positive_query"], r["negative_query"]
        try:
            sp = score_fn(a, p)
            sn = score_fn(a, n)
        except KeyError:
            skipped += 1
            continue
        margins.append(sp - sn)
        if sp > sn:
            correct += 1
    total = len(margins)
    acc = correct / total if total else 0.0
    mean_margin = float(np.mean(margins)) if margins else 0.0
    skip_note = f"  (skipped {skipped} missing embeddings)" if skipped else ""
    print(f"{name:30s}  ranking-acc={acc:.3f}  mean_margin(pos-neg)={mean_margin:+.4f}  n={total}{skip_note}")
    return {"name": name, "ranking_accuracy": acc, "mean_margin": mean_margin, "n": total}


def evaluate_embedding_model(name: str, emb_map: Dict[str, np.ndarray], rows: List[Dict]) -> Dict:
    def cos(a, b):
        return float(np.dot(emb_map[a], emb_map[b]))
    return report_ranking(name, rows, cos)


# ----------------------------
# Per-model evaluators (mirror evaluate_rq1.py's roster)
# ----------------------------

def evaluate_sparqlce_hard(checkpoint_path: str, rows: List[Dict], batch_size: int = 32) -> Optional[Dict]:
    if not os.path.isfile(checkpoint_path):
        print(f"Skip SPARQLCE: checkpoint not found at {checkpoint_path}")
        return None
    from sparqlce.embedder import SPARQLEmbedder
    model = SPARQLEmbedder(checkpoint_path)
    emb_map = build_emb_map_sparqlce(model, _unique_texts(rows), batch_size=batch_size)
    return evaluate_embedding_model("SPARQLCE", emb_map, rows)


def evaluate_sentence_transformer_hard(model_id, display_name, rows, batch_size=32, transform=None) -> Dict:
    model = SentenceTransformer(model_id)
    emb_map = build_emb_map_sentence_transformer(model, _unique_texts(rows), batch_size=batch_size, transform=transform)
    return evaluate_embedding_model(display_name, emb_map, rows)


def evaluate_sparql_encoder_baselines_hard(rows: List[Dict], batch_size: int = 32) -> List[Dict]:
    out = []
    for model_id, display_name in SPARQL_ENCODER_BASELINES:
        try:
            out.append(evaluate_sentence_transformer_hard(model_id, display_name, rows, batch_size=batch_size))
        except Exception as e:
            print(f"Skip {display_name} ({model_id}): {e}")
    return out


def evaluate_code_llm_hard(model_id, display_name, rows, batch_size=16) -> Optional[Dict]:
    try:
        emb_map = build_emb_map_code_llm(model_id, _unique_texts(rows), batch_size=batch_size)
        return evaluate_embedding_model(display_name, emb_map, rows)
    except Exception as e:
        print(f"Skip {display_name} ({model_id}): {e}")
        return None


def evaluate_e5_finetuned_hard(checkpoint_dir: str, display_name: str, rows: List[Dict], batch_size: int = 32) -> Optional[Dict]:
    """Full-finetuned E5 checkpoint (mean pooling, 'query: ' prefix), saved as a full model dir."""
    if not os.path.isdir(checkpoint_dir):
        print(f"Skip {display_name}: checkpoint dir not found at {checkpoint_dir}")
        return None
    model = SentenceTransformer(checkpoint_dir)
    emb_map = build_emb_map_sentence_transformer(model, _unique_texts(rows), batch_size=batch_size,
                                                  transform=lambda q: f"query: {q}")
    return evaluate_embedding_model(display_name, emb_map, rows)


def evaluate_tfidf_hard(rows: List[Dict]) -> Dict:
    texts = _unique_texts(rows)
    X = TfidfVectorizer(analyzer="word", ngram_range=(1, 2)).fit_transform(texts)
    X = sk_normalize(X, norm="l2", axis=1)
    idx = {t: i for i, t in enumerate(texts)}

    def cos(a, b):
        return float(X[idx[a]].multiply(X[idx[b]]).sum())
    return report_ranking("TF-IDF", rows, cos)


def evaluate_bm25_hard(rows: List[Dict]) -> Dict:
    texts = _unique_texts(rows)
    tokenized = [t.lower().split() for t in texts]
    bm25 = BM25Okapi(tokenized)
    idx = {t: i for i, t in enumerate(texts)}

    def score(a, b):
        scores = bm25.get_scores(a.lower().split())
        return float(scores[idx[b]])
    return report_ranking("BM25", rows, score)


# ----------------------------
# Full roster runner
# ----------------------------

def evaluate_all(rows: List[Dict], checkpoint_path: str, e5_ft_path: Optional[str] = None,
                  batch_size: int = 32, label: str = "") -> List[Dict]:
    print(f"\n{'=' * 70}\nHard-adversarial ranking check {label}(n={len(rows)})\n{'=' * 70}")
    results = [
        evaluate_sparqlce_hard(checkpoint_path, rows, batch_size=batch_size),
        evaluate_tfidf_hard(rows),
        evaluate_bm25_hard(rows),
        evaluate_sentence_transformer_hard("sentence-transformers/all-MiniLM-L6-v2", "SBERT/MiniLM", rows, batch_size=batch_size),
        evaluate_sentence_transformer_hard("sentence-transformers/all-mpnet-base-v2", "SBERT/MPNet", rows, batch_size=batch_size),
        evaluate_sentence_transformer_hard("intfloat/e5-base-v2", "E5-base-v2", rows, batch_size=batch_size,
                                           transform=lambda q: f"query: {q}"),
    ]
    results.extend(evaluate_sparql_encoder_baselines_hard(rows, batch_size=batch_size))
    results.append(evaluate_code_llm_hard("Qwen/Qwen2.5-Coder-0.5B", "Qwen2.5-Coder (zero-shot)", rows, batch_size=16))
    if e5_ft_path:
        results.append(evaluate_e5_finetuned_hard(e5_ft_path, "E5-base-v2 (fine-tuned)", rows, batch_size=batch_size))
    return [r for r in results if r]
