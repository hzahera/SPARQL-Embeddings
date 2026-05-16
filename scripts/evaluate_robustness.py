#!/usr/bin/env python3
"""Robustness evaluation for RQ2. Compares SPARQLECE vs. TF‑IDF, E5, Qwen.

Usage:
  python scripts/evaluate_robustness.py --robustness_file data/robustness_hard_dbpedia.json \
    --test_file data/hard_triplets_tfidf_dbpedia.json --sparqlce_ckpt checkpoints/triplet_model.pt
"""

import json
import argparse
import os
import sys
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity as sk_cosine
from scipy.stats import kendalltau
from sentence_transformers import SentenceTransformer

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))


# -------------------------------------------------------------------
# 1. Helpers – SPARQLEmbedder wrapper
# -------------------------------------------------------------------
class SparqlCEEncoder:
    """Light wrapper around the repository's `sparqlce.embedder.SPARQLEmbedder`.

    Expects a checkpoint path compatible with `sparqlce.embedder.SPARQLEmbedder`.
    """
    def __init__(self, ckpt_path, device='cpu'):
        from sparqlce.embedder import SPARQLEmbedder
        self.device = device
        self.embedder = SPARQLEmbedder(ckpt_path)

    def encode(self, queries, batch_size=32):
        # `SPARQLEmbedder.encode` returns a numpy array
        return self.embedder.encode(queries)


class E5Encoder:
    def __init__(self, model_name='intfloat/e5-base-v2', device='cpu'):
        self.model = SentenceTransformer(model_name, device=device)

    def encode(self, queries):
        prefixed = [f'query: {q}' for q in queries]
        return self.model.encode(prefixed, convert_to_numpy=True, show_progress_bar=False)


class QwenEncoder:
    def __init__(self, model_name='Qwen/Qwen2.5-0.5B', device='cpu'):
        self.model = SentenceTransformer(model_name, device=device)

    def encode(self, queries):
        return self.model.encode(queries, convert_to_numpy=True, show_progress_bar=False)


class TFIDFEncoder:
    def __init__(self, corpus):
        self.vectorizer = TfidfVectorizer(analyzer='word', ngram_range=(1,2), max_features=10000)
        self.embeddings = self.vectorizer.fit_transform(corpus).toarray()

    def encode(self, queries):
        return self.vectorizer.transform(queries).toarray()


# -------------------------------------------------------------------
# 2. Metrics
# -------------------------------------------------------------------
def compute_cosine_sim(emb_orig, emb_pert):
    sim = F.cosine_similarity(torch.tensor(emb_orig), torch.tensor(emb_pert), dim=1)
    return sim.numpy()


def retrieval_rank_metrics(emb_orig, emb_pert, corpus_emb, top_k=10):
    """Jaccard overlap of top-k neighbours and Kendall tau between score lists."""
    orig_scores = sk_cosine(emb_orig, corpus_emb)
    pert_scores = sk_cosine(emb_pert, corpus_emb)
    n = emb_orig.shape[0]
    jaccard = []
    tau_vals = []
    for i in range(n):
        idx_orig = np.argsort(-orig_scores[i])[:top_k]
        idx_pert = np.argsort(-pert_scores[i])[:top_k]
        inter = len(set(idx_orig) & set(idx_pert))
        jaccard.append(inter / top_k)
        tau, _ = kendalltau(orig_scores[i], pert_scores[i])
        tau_vals.append(0.0 if np.isnan(tau) else tau)
    return np.mean(jaccard), np.mean(tau_vals)


def mrr_retention(emb_orig, emb_pert, corpus_emb, anchor_ids, pos_map):
    """MRR of perturbed query w.r.t. the same positive set as original."""
    mrr_orig = []
    mrr_pert = []
    for i, anchor_id in enumerate(anchor_ids):
        pos_indices = pos_map[anchor_id]
        if not pos_indices:
            continue
        scores_orig = sk_cosine(emb_orig[i:i+1], corpus_emb).flatten()
        rank_orig = 1 + np.sum(scores_orig > scores_orig[pos_indices].max())
        mrr_orig.append(1.0 / rank_orig)
        scores_pert = sk_cosine(emb_pert[i:i+1], corpus_emb).flatten()
        rank_pert = 1 + np.sum(scores_pert > scores_pert[pos_indices].max())
        mrr_pert.append(1.0 / rank_pert)
    return np.mean(mrr_orig) if mrr_orig else 0.0, np.mean(mrr_pert) if mrr_pert else 0.0


# -------------------------------------------------------------------
# 3. Main
# -------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--robustness_file', required=True)
    parser.add_argument('--test_file', required=True, help='hard_triplets_tfidf_dbpedia.json')
    parser.add_argument('--sparqlce_ckpt', required=True)
    parser.add_argument('--output_dir', default='results/robustness')
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()

    # Load perturbation pairs
    with open(args.robustness_file) as f:
        pairs = json.load(f)
    print(f"Loaded {len(pairs)} robustness pairs")

    # Load test data for corpus and positives
    with open(args.test_file) as f:
        test_data = json.load(f)

    # Build corpus of unique queries and mapping for MRR
    corpus_queries = []
    query_to_idx = {}
    pos_map = defaultdict(list)

    for item in test_data:
        anchor = item.get('anchor_query') or item.get('query')
        if anchor is None:
            continue
        if anchor not in query_to_idx:
            query_to_idx[anchor] = len(corpus_queries)
            corpus_queries.append(anchor)
        for p in item.get('positive_queries', []):
            p_text = p['query'] if isinstance(p, dict) else p
            if p_text not in query_to_idx:
                query_to_idx[p_text] = len(corpus_queries)
                corpus_queries.append(p_text)
            pos_map[anchor].append(query_to_idx[p_text])
        for n in item.get('negative_queries', []):
            n_text = n['query'] if isinstance(n, dict) else n
            if n_text not in query_to_idx:
                query_to_idx[n_text] = len(corpus_queries)
                corpus_queries.append(n_text)

    print(f"Corpus size: {len(corpus_queries)}")

    # Models to evaluate
    models = {
        'sparqlce': SparqlCEEncoder(args.sparqlce_ckpt, device=args.device),
        'e5': E5Encoder(device=args.device),
        'qwen': QwenEncoder(device=args.device),
        'tfidf': TFIDFEncoder(corpus_queries),
    }

    # Group pairs by perturbation type
    by_type = defaultdict(list)
    for p in pairs:
        by_type[p['perturbation_type']].append(p)

    results = {}
    for name, encoder in models.items():
        print(f"\n=== {name} ===")
        originals = [p['original'] for p in pairs]
        perturbeds = [p['perturbed'] for p in pairs]
        emb_orig = encoder.encode(originals)
        emb_pert = encoder.encode(perturbeds)
        cos_sim = compute_cosine_sim(emb_orig, emb_pert)
        corpus_emb = encoder.encode(corpus_queries)

        anchor_ids = []
        anchor_emb_orig = []
        anchor_emb_pert = []
        for i, p in enumerate(pairs):
            if p['original'] in pos_map:
                anchor_ids.append(p['original'])
                anchor_emb_orig.append(emb_orig[i])
                anchor_emb_pert.append(emb_pert[i])
        if anchor_ids:
            mrr_orig_val, mrr_pert_val = mrr_retention(
                np.array(anchor_emb_orig), np.array(anchor_emb_pert), corpus_emb, anchor_ids, pos_map
            )
        else:
            mrr_orig_val, mrr_pert_val = 0, 0

        per_type = {}
        for ptype, items in by_type.items():
            idxs = []
            for item in items:
                try:
                    i = originals.index(item['original'])
                    if perturbeds[i] == item['perturbed']:
                        idxs.append(i)
                except ValueError:
                    pass
            if idxs:
                type_cos = cos_sim[idxs].mean()
                per_type[ptype] = {'mean_cos_sim': float(type_cos)}

        results[name] = {
            'overall_mean_cos_sim': float(cos_sim.mean()),
            'overall_std_cos_sim': float(cos_sim.std()),
            'mrr_original': float(mrr_orig_val),
            'mrr_perturbed': float(mrr_pert_val),
            'mrr_drop': float(mrr_orig_val - mrr_pert_val),
            'per_type': per_type
        }

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, 'robustness_results.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)

    print("\n========== Robustness Comparison ==========")
    print(f"{'Model':<10} {'CosSim':>8} {'MRR Orig':>8} {'MRR Pert':>8} {'MRR Drop':>8}")
    for name in models:
        r = results[name]
        print(f"{name:<10} {r['overall_mean_cos_sim']:8.4f} {r['mrr_original']:8.4f} {r['mrr_perturbed']:8.4f} {r['mrr_drop']:8.4f}")

    print("\nPer perturbation type (mean cosine similarity):")
    for name in models:
        print(f"\n{name}:")
        for ptype, stats in results[name]['per_type'].items():
            print(f"  {ptype:<25}: {stats['mean_cos_sim']:.4f}")


if __name__ == '__main__':
    main()
