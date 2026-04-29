"""
RQ2 Evaluation: Robustness to structural perturbations (all 5 types).
For each model, computes mean cosine similarity between original and perturbed queries.
Outputs a table like Table 4 (with mean cosine instead of Spearman, because ground truth is constant).
Usage: python rq2_eval_all.py --input dbpedia_rq2.json --output results.csv
"""

import json
import argparse
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import warnings
import sys
from pathlib import Path

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.embedder import SPARQLEmbedder
warnings.filterwarnings('ignore')

# Embedder classes
class TFIDFEmbedder:
    def __init__(self, ngram_range=(1,2)):
        self.ngram_range = ngram_range
        self.vectorizer = None
    
    def fit(self, texts):
        self.vectorizer = TfidfVectorizer(ngram_range=self.ngram_range, lowercase=False)
        self.vectorizer.fit(texts)
    
    def encode(self, texts):
        return self.vectorizer.transform(texts)
    
    def name(self):
        return "TF-IDF (1-2 grams)"

class SentenceTransformerEmbedder:
    def __init__(self, model_name, prefix=""):
        self.model = SentenceTransformer(model_name)
        self.prefix = prefix
        self.model_name = model_name
    
    def encode(self, texts):
        if self.prefix:
            texts = [self.prefix + t for t in texts]
        return self.model.encode(texts, normalize_embeddings=True)
    
    def name(self):
        return self.model_name

class SparqlCEEmbedder:
    def __init__(self, checkpoint_path, batch_size=32):
        self.model = SPARQLEmbedder(checkpoint_path)
        self.batch_size = batch_size

    def encode(self, texts):
        embs = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            batch_embs = self.model.encode(batch)
            embs.append(batch_embs)

        embs = np.vstack(embs)
        embs = embs / (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-12)
        return embs

    def name(self):
        return "SPARQLCE (Ours)"
        
# For convenience, a wrapper that encodes then computes cosine
def compute_cosines(embedder, anchor_texts, pert_texts):
    """Compute cosine similarity for each (anchor, perturbed) pair."""
    if hasattr(embedder, 'fit'):   # TF-IDF needs fitting
        all_texts = anchor_texts + pert_texts
        embedder.fit(all_texts)
    # For SparqlCE dummy, we bypass encoding
    if hasattr(embedder, 'compute_cosine'):
        return embedder.compute_cosine(anchor_texts, pert_texts)
    # Encode
    anchor_embs = embedder.encode(anchor_texts)
    pert_embs = embedder.encode(pert_texts)
    cos_sims = []
    for i in range(len(anchor_texts)):
        if hasattr(anchor_embs, "toarray"):  # sparse
            sim = cosine_similarity(anchor_embs[i], pert_embs[i])[0][0]
        else:
            sim = np.dot(anchor_embs[i], pert_embs[i])
        cos_sims.append(sim)
    return np.array(cos_sims)

# Data loading
def load_pairs_by_type(json_path):
    """
    Return dictionary: perturbation_type -> list of (anchor_query, perturbed_query)
    Types: variable_renaming, triple_reordering, optional_insertion, 
           limit_offset_injection, combined
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    pairs_by_type = {}
    for entry in data:
        anchor = entry["anchor_query"]
        for pert in entry.get("valid_perturbations", []):
            ptype = pert["type"]
            perturbed = pert["query"]
            if ptype not in pairs_by_type:
                pairs_by_type[ptype] = []
            pairs_by_type[ptype].append((anchor, perturbed))
    return pairs_by_type


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="RQ2 JSON file (with valid_perturbations)")
    parser.add_argument("--output", default="rq2_results_all.csv", help="Output CSV")
    parser.add_argument("--sparqlce_checkpoint", required=True, help="Path to trained SPARQLCE checkpoint .pt file")
    args = parser.parse_args()
    
    # Load pairs for each perturbation type
    pairs_by_type = load_pairs_by_type(args.input)
    print("Loaded pairs:")
    for ptype, pairs in pairs_by_type.items():
        print(f"  {ptype}: {len(pairs)} pairs")
    
    # Define models (order as in table)
    models = [
        TFIDFEmbedder(ngram_range=(1,2)),
        SentenceTransformerEmbedder("all-MiniLM-L6-v2"),
        SentenceTransformerEmbedder("intfloat/e5-base-v2", prefix="query: "),
        SentenceTransformerEmbedder("all-mpnet-base-v2"),
        SentenceTransformerEmbedder("Qwen/Qwen3-Embedding-0.6B"),
        SparqlCEEmbedder(args.sparqlce_checkpoint)
    ]
    
    # Perturbation types in desired order (matching table columns)
    pert_types = ["variable_renaming", "triple_reordering", "optional_insertion",
                  "limit_offset_injection", "combined"]
    
    results = []
    for model in models:
        row = {"Model": model.name()}
        for ptype in pert_types:
            if ptype not in pairs_by_type:
                row[ptype] = np.nan
                continue
            pairs = pairs_by_type[ptype]
            anchor_texts = [p[0] for p in pairs]
            pert_texts = [p[1] for p in pairs]
            try:
                cos_sims = compute_cosines(model, anchor_texts, pert_texts)
                mean_cos = np.mean(cos_sims)
                row[ptype] = mean_cos
                print(f"{model.name()} | {ptype}: mean cosine = {mean_cos:.4f}")
            except Exception as e:
                print(f"Error with {model.name()} on {ptype}: {e}")
                row[ptype] = np.nan
        # Average across available perturbation types
        vals = [row[pt] for pt in pert_types if not np.isnan(row[pt])]
        row["Avg"] = np.mean(vals) if vals else np.nan
        results.append(row)
    
    # Create DataFrame and format table
    df = pd.DataFrame(results)
    df = df.set_index("Model")
    # Reorder columns
    df = df[pert_types + ["Avg"]]
    
    # Print table (similar to LaTeX)
    print("\n" + "="*80)
    print("RQ2 Robustness Results (Mean Cosine Similarity ↑, perfect = 1.0)")
    print("="*80)
    print(df.round(4).to_string())
    
    # Save CSV
    df.to_csv(args.output)
    print(f"\nResults saved to {args.output}")

if __name__ == "__main__":
    main()