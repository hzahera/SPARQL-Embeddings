#!/usr/bin/env python3
"""Generate training triplets from unique SPARQL queries using Jaccard similarity."""

import json, os, itertools, random, numpy as np
from collections import defaultdict
from SPARQLWrapper import SPARQLWrapper, JSON

# ---- Config ----
SPARQL_ENDPOINT = "https://dbpedia.org/sparql"
TIMEOUT = 30
JACCARD_THRESHOLD_POS = 0.7   # Jaccard > 0.7 → positive
JACCARD_THRESHOLD_NEG = 0.1   # Jaccard < 0.1 → negative
MAX_POSITIVES = 5
MAX_NEGATIVES = 10
MAX_QUERIES = 1000  # limit for speed; set to None for all
SEED = 42
# -----------------

random.seed(SEED)
np.random.seed(SEED)

def execute_query(sparql_query, endpoint=SPARQL_ENDPOINT):
    """Execute a SPARQL query and return a frozenset of result rows."""
    try:
        wrapper = SPARQLWrapper(endpoint)
        wrapper.setQuery(sparql_query)
        wrapper.setReturnFormat(JSON)
        wrapper.setTimeout(TIMEOUT)
        results = wrapper.query().convert()
        rows = set()
        for binding in results["results"]["bindings"]:
            row = tuple((var, binding[var]["value"]) for var in sorted(binding.keys()))
            rows.add(row)
        return frozenset(rows)
    except Exception as e:
        print(f"  Query failed: {e}")
        return None

def jaccard_similarity(set1, set2):
    """Compute Jaccard similarity between two sets."""
    if set1 is None or set2 is None or len(set1 | set2) == 0:
        return 0.0
    return len(set1 & set2) / len(set1 | set2)

def main():
    # Load unique queries
    queries_path = "data/unique_queries_dbpedia.json"
    if not os.path.exists(queries_path):
        queries_path = "data/unique_queries_dbpedia.txt"
        with open(queries_path) as f:
            queries = [line.strip() for line in f if line.strip()]
    else:
        with open(queries_path) as f:
            queries = json.load(f)
    
    print(f"Loaded {len(queries)} unique queries")
    
    if MAX_QUERIES:
        queries = random.sample(queries, min(MAX_QUERIES, len(queries)))
        print(f"Sampled {len(queries)} queries for speed")
    
    # Execute all queries and cache results
    print("\nExecuting queries against DBpedia...")
    results = {}
    for i, q in enumerate(queries):
        if i % 100 == 0:
            print(f"  Progress: {i}/{len(queries)}")
        results[q] = execute_query(q)
    
    # Filter to queries that returned results
    successful = {q: r for q, r in results.items() if r is not None and len(r) > 0}
    print(f"\nSuccessful queries: {len(successful)}/{len(queries)}")
    
    if len(successful) < 10:
        print("Too few successful queries — aborting.")
        return
    
    # Compute Jaccard similarity for all pairs (sample pairs for efficiency)
    queries_list = list(successful.keys())
    n = len(queries_list)
    
    print(f"\nComputing Jaccard similarities for {n} queries...")
    
    # For efficiency, sample anchor queries
    anchor_queries = queries_list[:min(500, n)]
    
    triplets = []
    for i, anchor in enumerate(anchor_queries):
        if i % 50 == 0:
            print(f"  Mining triplets: {i}/{len(anchor_queries)}")
        
        anchor_result = successful[anchor]
        
        # Compute Jaccard with all other queries
        candidates = []
        for other in queries_list:
            if other == anchor:
                continue
            jacc = jaccard_similarity(anchor_result, successful[other])
            candidates.append((other, jacc))
        
        # Sort by Jaccard (highest first)
        candidates.sort(key=lambda x: x[1], reverse=True)
        
        # Positives: Jaccard > threshold
        positives = [(q, j) for q, j in candidates if j > JACCARD_THRESHOLD_POS][:MAX_POSITIVES]
        
        # Negatives: Jaccard < threshold (take from low end)
        negatives = [(q, j) for q, j in candidates if j < JACCARD_THRESHOLD_NEG]
        negatives.sort(key=lambda x: x[1])  # lowest first
        negatives = negatives[:MAX_NEGATIVES]
        
        if positives:
            triplets.append({
                "anchor_query": anchor,
                "positive_queries": [{"query": q, "answer_jaccard": j} for q, j in positives],
                "negative_queries": [{"query": q, "answer_jaccard": j} for q, j in negatives]
            })
    
    print(f"\nGenerated {len(triplets)} triplets")
    
    # Save
    output_dir = "data/dataset/raw"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "jaccard_triplets_dbpedia.json")
    
    with open(output_path, "w") as f:
        json.dump(triplets, f, indent=2)
    
    print(f"Saved to {output_path}")
    
    # Also create splits for train/val/test
    random.shuffle(triplets)
    train_size = int(0.8 * len(triplets))
    val_size = int(0.1 * len(triplets))
    
    train = triplets[:train_size]
    val = triplets[train_size:train_size + val_size]
    test = triplets[train_size + val_size:]
    
    splits = {
        "train_indices": list(range(train_size)),
        "val_indices": list(range(train_size, train_size + val_size)),
        "test_indices": list(range(train_size + val_size, len(triplets)))
    }
    
    splits_path = "data/splits_dbpedia.json"
    with open(splits_path, "w") as f:
        json.dump(splits, f, indent=2)
    print(f"Splits saved to {splits_path}")

if __name__ == "__main__":
    main()