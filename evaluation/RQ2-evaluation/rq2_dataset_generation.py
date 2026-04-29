#!/usr/bin/env python3
"""
Generate RQ2 dataset for test split anchors only.
Usage: python rq2_dataset_generation.py --anchors anchors.json --splits splits.json --endpoint SPARQL_URL --output output.json
"""

import json
import re
import random
import argparse
import time
from typing import List, Dict, Any, Set, Tuple, Optional
from SPARQLWrapper import SPARQLWrapper, JSON, SPARQLExceptions

# ----------------------------------------------------------------------
# Perturbation functions (same as before)
# ----------------------------------------------------------------------

def rename_variables(query: str) -> str:
    var_pattern = re.compile(r'[?$]([a-zA-Z_][a-zA-Z0-9_]*)')
    var_matches = var_pattern.findall(query)
    seen = []
    for v in var_matches:
        if v not in seen:
            seen.append(v)
    mapping = {f"?{v}": f"?v{i}" for i, v in enumerate(seen)}
    mapping.update({f"${v}": f"?v{i}" for i, v in enumerate(seen)})
    for old, new in sorted(mapping.items(), key=lambda x: -len(x[0])):
        query = query.replace(old, new)
    return query

def reorder_triples(query: str) -> str:
    where_match = re.search(r'WHERE\s*\{', query, re.IGNORECASE)
    if not where_match:
        return query
    start = where_match.end()
    brace_count = 1
    end = start
    while brace_count > 0 and end < len(query):
        if query[end] == '{':
            brace_count += 1
        elif query[end] == '}':
            brace_count -= 1
        end += 1
    where_body = query[start:end-1]
    triples = re.split(r'\.\s+(?![^<]*>)', where_body.strip())
    triples = [t.strip() for t in triples if t.strip()]
    if len(triples) <= 1:
        return query
    random.shuffle(triples)
    new_body = ' . '.join(triples)
    if where_body.rstrip().endswith('.'):
        new_body += ' .'
    return query[:start] + new_body + query[end-1:]

def insert_dummy_optional(query: str) -> str:
    dummy = "OPTIONAL { ?s <http://dummy.org/nonexistent> ?o }"
    where_match = re.search(r'WHERE\s*\{', query, re.IGNORECASE)
    if not where_match:
        return query
    start = where_match.end()
    brace_count = 1
    end = start
    while brace_count > 0 and end < len(query):
        if query[end] == '{':
            brace_count += 1
        elif query[end] == '}':
            brace_count -= 1
        end += 1
    return query[:end-1] + "\n  " + dummy + "\n" + query[end-1:]

def inject_limit_offset(query: str, original_count: int, max_limit: int = 10000) -> str:
    if re.search(r'\bLIMIT\b', query, re.IGNORECASE) or re.search(r'\bOFFSET\b', query, re.IGNORECASE):
        return query
    if original_count < max_limit:
        stripped = query.rstrip()
        if stripped.endswith(';'):
            stripped = stripped[:-1].rstrip()
        return stripped + f" LIMIT {max_limit} OFFSET 0"
    return query

def combined_perturbation(query: str, original_count: int) -> str:
    q = rename_variables(query)
    q = reorder_triples(q)
    q = insert_dummy_optional(q)
    q = inject_limit_offset(q, original_count)
    return q

# ----------------------------------------------------------------------
# SPARQL execution and validation
# ----------------------------------------------------------------------

def execute_sparql(endpoint: str, query: str, retries: int = 3) -> Optional[List[Dict[str, str]]]:
    sparql = SPARQLWrapper(endpoint)
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    for attempt in range(retries):
        try:
            results = sparql.query().convert()
            bindings = results.get('results', {}).get('bindings', [])
            simplified = []
            for b in bindings:
                row = {var: term.get('value') for var, term in b.items()}
                simplified.append(row)
            return simplified
        except Exception as e:
            if attempt == retries - 1:
                print(f"    SPARQL error: {e}")
                return None
            time.sleep(2 ** attempt)
    return None

def result_set_to_tuple_set(bindings: List[Dict[str, str]]) -> Set[Tuple[Optional[str], ...]]:
    if not bindings:
        return set()
    all_vars = sorted({k for b in bindings for k in b.keys()})
    return {tuple(b.get(v) for v in all_vars) for b in bindings}

def jaccard_similarity(set1: Set[Any], set2: Set[Any]) -> float:
    if not set1 and not set2:
        return 1.0
    inter = len(set1 & set2)
    union = len(set1 | set2)
    return inter / union if union != 0 else 0.0

def validate_pair(endpoint: str, original_query: str, perturbed_query: str) -> bool:
    orig_bindings = execute_sparql(endpoint, original_query)
    if orig_bindings is None:
        return False
    pert_bindings = execute_sparql(endpoint, perturbed_query)
    if pert_bindings is None:
        return False
    orig_set = result_set_to_tuple_set(orig_bindings)
    pert_set = result_set_to_tuple_set(pert_bindings)
    jac = jaccard_similarity(orig_set, pert_set)
    return abs(jac - 1.0) < 1e-9

# ----------------------------------------------------------------------
# Main generation
# ----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--anchors", required=True, help="JSON file with all anchors (list)")
    parser.add_argument("--splits", required=True, help="JSON file with test_idx list")
    parser.add_argument("--endpoint", required=True, help="SPARQL endpoint URL")
    parser.add_argument("--output", required=True, help="Output JSON for test split RQ2 dataset")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    random.seed(args.seed)

    # Load splits
    with open(args.splits, 'r') as f:
        splits = json.load(f)
    test_idx = set(splits["test_idx"])
    print(f"Loaded test split with {len(test_idx)} anchor indices")

    # Load all anchors
    with open(args.anchors, 'r') as f:
        all_anchors = json.load(f)
    print(f"Total anchors in file: {len(all_anchors)}")

    # Filter test anchors
    test_anchors = []
    for i, entry in enumerate(all_anchors):
        if i in test_idx:
            # Extract only the anchor query (ignore positive/negative)
            anchor_query = entry.get("anchor_query")
            if anchor_query:
                test_anchors.append({"anchor_query": anchor_query, "original_index": i})
    print(f"Filtered {len(test_anchors)} test anchors")

    perturbations = [
        ("variable_renaming", rename_variables, False),
        ("triple_reordering", reorder_triples, False),
        ("optional_insertion", insert_dummy_optional, False),
        ("limit_offset_injection", inject_limit_offset, True),
        ("combined", combined_perturbation, True),
    ]

    output_data = []

    for idx, item in enumerate(test_anchors):
        anchor_query = item["anchor_query"]
        orig_idx = item["original_index"]
        print(f"\nProcessing test anchor {idx+1}/{len(test_anchors)} (orig index {orig_idx})")

        # Execute anchor to get result count
        orig_bindings = execute_sparql(args.endpoint, anchor_query)
        if orig_bindings is None:
            print("  Cannot execute anchor, skipping")
            continue
        orig_count = len(orig_bindings)
        print(f"  Result count: {orig_count}")

        valid_perturbations = []
        for pert_name, pert_func, needs_count in perturbations:
            try:
                if needs_count:
                    perturbed = pert_func(anchor_query, orig_count)
                else:
                    perturbed = pert_func(anchor_query)
            except Exception as e:
                print(f"  Error generating {pert_name}: {e}")
                continue

            if perturbed == anchor_query:
                print(f"  {pert_name}: no change, skipping")
                continue

            print(f"  Validating {pert_name}... ", end="", flush=True)
            if validate_pair(args.endpoint, anchor_query, perturbed):
                print("ACCEPTED")
                valid_perturbations.append({
                    "type": pert_name,
                    "query": perturbed,
                    "original_result_count": orig_count
                })
            else:
                print("REJECTED")

        output_data.append({
            "anchor_query": anchor_query,
            "anchor_result_count": orig_count,
            "valid_perturbations": valid_perturbations
        })

    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"\nSaved {len(output_data)} test anchors with perturbations to {args.output}")

if __name__ == "__main__":
    main()