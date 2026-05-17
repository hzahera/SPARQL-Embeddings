#!/usr/bin/env python3
"""Robustness evaluation for RQ2 - fixed version."""

import json, argparse, sys, os
import pathlib
from collections import defaultdict
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity as sk_cosine
from scipy.stats import kendalltau
from sentence_transformers import SentenceTransformer

# Variable canonicalization is now handled inside sparqlce.graph_builder.SPARQLGraphBuilder
# via sparqlce.sparql_utils.canonicalize_variables — see SPARQLGraphBuilder.build_graph.

# -------------------------------------------------------------------
# 1. Model wrappers – adjust SparqlCEEncoder for your main branch
# -------------------------------------------------------------------
class SparqlCEEncoder:
    def __init__(self, ckpt_path, device='cuda'):
        project_root = pathlib.Path(__file__).resolve().parent.parent
        src_path = project_root / 'src'
        if src_path.exists() and str(src_path) not in sys.path:
            sys.path.insert(0, str(src_path))
        elif str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))

        if src_path.exists():
            from hybrid_encoder import HybridSPARQLEncoder
            from config import Config
            from embedder import SPARQLEmbedder

            self.device = device
            self.model = HybridSPARQLEncoder(
                embed_dim=Config.OUTPUT_DIM,
                hidden_dim=Config.HIDDEN_DIM,
                output_dim=Config.OUTPUT_DIM
            )
            state = torch.load(ckpt_path, map_location=device)
            if isinstance(state, dict) and 'model_state_dict' in state:
                self.model.load_state_dict(state['model_state_dict'])
            else:
                self.model.load_state_dict(state)
            self.model.to(device)
            self.model.eval()
            self.embedder = SPARQLEmbedder(self.model)
        else:
            from sparqlce.embedder import SPARQLEmbedder
            self.device = device
            self.embedder = SPARQLEmbedder(ckpt_path)

    def encode(self, queries, batch_size=32):
        if hasattr(self.embedder, 'embed'):
            return self.embedder.embed(queries)
        if hasattr(self.embedder, 'encode'):
            return self.embedder.encode(queries)
        raise AttributeError("SPARQLEmbedder does not expose embed() or encode()")

class E5Encoder:
    def __init__(self, model_name='intfloat/e5-base-v2', device='cuda'):
        self.model = SentenceTransformer(model_name, device=device)
    def encode(self, queries):
        prefixed = [f'query: {q}' for q in queries]
        return self.model.encode(prefixed, convert_to_numpy=True, show_progress_bar=False)

class QwenEncoder:
    def __init__(self, model_name='Qwen/Qwen2.5-0.5B', device='cuda'):
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


def mrr_retention(anchor_emb_orig, anchor_emb_pert, corpus_emb, pos_indices_list):
    """pos_indices_list is a list of arrays of positive indices per anchor"""
    mrr_orig = []
    mrr_pert = []
    for i, pos_idx in enumerate(pos_indices_list):
        if len(pos_idx) == 0:
            continue
        # original MRR
        scores_orig = sk_cosine(anchor_emb_orig[i:i+1], corpus_emb).flatten()
        best_orig = scores_orig[pos_idx].max()
        rank_orig = 1 + np.sum(scores_orig > best_orig)
        mrr_orig.append(1.0 / rank_orig)
        # perturbed MRR
        scores_pert = sk_cosine(anchor_emb_pert[i:i+1], corpus_emb).flatten()
        best_pert = scores_pert[pos_idx].max()
        rank_pert = 1 + np.sum(scores_pert > best_pert)
        mrr_pert.append(1.0 / rank_pert)
    return np.mean(mrr_orig) if mrr_orig else 0, np.mean(mrr_pert) if mrr_pert else 0

# -------------------------------------------------------------------
# 3. Main
# -------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--robustness_file', required=True)
    parser.add_argument('--test_file', required=True)
    parser.add_argument('--sparqlce_ckpt', required=True)
    parser.add_argument('--output_dir', default='results/robustness')
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()

    # Load robustness pairs
    with open(args.robustness_file) as f:
        pairs = json.load(f)
    print(f"Loaded {len(pairs)} robustness pairs")
    # Assign unique index to each pair for later grouping
    for i, p in enumerate(pairs):
        p['_idx'] = i

    # Load test data – handle both formats
    test_data_loaded = []
    with open(args.test_file) as f:
        test_data_loaded = json.load(f)

    corpus_queries = []
    query_to_idx = {}
    anchor_pos_indices = defaultdict(list)

    # NOTE: queries are intentionally NOT canonicalized here.
    # The SPARQLCE graph builder canonicalizes internally, and TF-IDF/E5/Qwen
    # operate on the raw text — canonicalizing here would unfairly normalize
    # inputs for the text-based baselines too.
    for item in test_data_loaded:
        anchor = item.get('query', item.get('anchor_query', '')).strip()
        if not anchor:
            continue
        if anchor not in query_to_idx:
            query_to_idx[anchor] = len(corpus_queries)
            corpus_queries.append(anchor)

        # positives (could be strings or dicts with 'query')
        for p in item.get('positive_queries', []):
            p_text = p['query'].strip() if isinstance(p, dict) else p.strip()
            if p_text not in query_to_idx:
                query_to_idx[p_text] = len(corpus_queries)
                corpus_queries.append(p_text)
            anchor_pos_indices[anchor].append(query_to_idx[p_text])

        # negatives
        for n in item.get('negative_queries', []):
            n_text = n['query'].strip() if isinstance(n, dict) else n.strip()
            if n_text not in query_to_idx:
                query_to_idx[n_text] = len(corpus_queries)
                corpus_queries.append(n_text)

    print(f"Corpus size: {len(corpus_queries)}")
    print(f"Test anchors with positives: {len(anchor_pos_indices)}")

    # Build a mapping from test anchor -> its perturbed embedding (we'll collect later)
    # For now, we'll evaluate MRR using test anchors directly: we'll need the perturbed embedding for each anchor.
    # We'll collect perturbed embeddings for anchors that appear in the robustness pairs.
    anchor_to_perturbed = {}  # anchor -> list of perturbed embeddings (one per perturbation type? We'll take the first or average? Better: we'll evaluate each perturbation type separately, but for overall we can average? Let's pick one representative or compute MRR per type later.)
    # Actually, the robustness dataset contains multiple perturbations per original. We can compute MRR per perturbation type later.
    # For overall MRR drop, we can compute MRR using each perturbed version and average. Simpler: pick the first perturbed version for each anchor, or compute per-type.
    # We'll compute per-type MRR later. For overall, we'll compute the mean across all perturbations for each anchor.

    # Instead, let's create a list of (anchor, perturbed_query, perturbed_type) and group by anchor.
    pert_by_anchor = defaultdict(list)
    for p in pairs:
        orig = p['original'].strip()
        pert = p['perturbed'].strip()
        p['original'] = orig
        p['perturbed'] = pert
        pert_by_anchor[orig].append(p)

    # Models
    models = {
        'sparqlce': SparqlCEEncoder(args.sparqlce_ckpt, args.device),
        'e5': E5Encoder(device=args.device),
        'qwen': QwenEncoder(device=args.device),
        'tfidf': TFIDFEncoder(corpus_queries),
    }

    results = {}
    for name, encoder in models.items():
        print(f"\n=== {name} ===")
        # Encode all originals and perturbeds (raw text — encoders handle their
        # own normalization; SPARQLCE canonicalizes inside the graph builder)
        originals = [p['original'] for p in pairs]
        perturbeds = [p['perturbed'] for p in pairs]
        emb_orig = encoder.encode(originals)
        emb_pert = encoder.encode(perturbeds)

        # Quick debug: print first pair
        print("  Debug first pair:")
        print(f"    original: {originals[0][:100]}...")
        print(f"    perturbed: {perturbeds[0][:100]}...")
        cos0 = F.cosine_similarity(torch.tensor(emb_orig[0:1]), torch.tensor(emb_pert[0:1]))
        print(f"    cosine: {cos0.item():.4f}")

        # Overall cosine similarity
        cos_sim = compute_cosine_sim(emb_orig, emb_pert)

        # Corpus embeddings (same for all queries)
        corpus_emb = encoder.encode(corpus_queries)

        # --- MRR retention ---
        # For each test anchor, get its perturbed embedding(s) and compute MRR.
        mrr_orig_total = []
        mrr_pert_total = []
        for anchor, pos_idx in anchor_pos_indices.items():
            if anchor not in pert_by_anchor:
                continue  # no perturbation for this anchor
            # Original embedding (anchor is in corpus_queries)
            anchor_idx = query_to_idx[anchor]
            anchor_emb_orig = corpus_emb[anchor_idx:anchor_idx+1]
            # Compute original MRR
            scores_orig = sk_cosine(anchor_emb_orig, corpus_emb).flatten()
            best_orig = scores_orig[pos_idx].max()
            rank_orig = 1 + np.sum(scores_orig > best_orig)
            mrr_orig_total.append(1.0 / rank_orig)

            # Perturbed MRR: average over all perturbed versions of this anchor
            mrr_pert_versions = []
            for pert_pair in pert_by_anchor[anchor]:
                pert_idx = pert_pair['_idx']
                pert_emb = emb_pert[pert_idx:pert_idx+1]
                scores_pert = sk_cosine(pert_emb, corpus_emb).flatten()
                best_pert = scores_pert[pos_idx].max()
                rank_pert = 1 + np.sum(scores_pert > best_pert)
                mrr_pert_versions.append(1.0 / rank_pert)
            if mrr_pert_versions:
                mrr_pert_total.append(np.mean(mrr_pert_versions))

        mean_mrr_orig = np.mean(mrr_orig_total) if mrr_orig_total else 0
        mean_mrr_pert = np.mean(mrr_pert_total) if mrr_pert_total else 0
        mrr_drop = mean_mrr_orig - mean_mrr_pert

        # Per-type cosine similarity (using _idx)
        by_type = defaultdict(list)
        for p in pairs:
            by_type[p['perturbation_type']].append(p['_idx'])
        per_type = {}
        for ptype, idxs in by_type.items():
            type_cos = cos_sim[idxs].mean()
            per_type[ptype] = {'mean_cos_sim': float(type_cos)}

        results[name] = {
            'overall_mean_cos_sim': float(cos_sim.mean()),
            'overall_std_cos_sim': float(cos_sim.std()),
            'mrr_original': float(mean_mrr_orig),
            'mrr_perturbed': float(mean_mrr_pert),
            'mrr_drop': float(mrr_drop),
            'per_type': per_type
        }

    # Save
    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, 'robustness_results.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)

    # Print summary
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
