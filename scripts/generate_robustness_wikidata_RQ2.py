#!/usr/bin/env python3
"""
Generate a HARD robustness dataset for SPARQL embeddings.
Optimized for Wikidata (handles PREFIXes, SERVICE blocks, and Label variables).
Reads anchor queries from a triplet dataset format.
"""

import argparse
import json
import random
import re
from typing import List, Set, Optional
from rdflib.plugins.sparql.parser import parseQuery


def get_all_variables(query: str) -> Set[str]:
    return set(re.findall(r'\?[a-zA-Z0-9_]+', query))


def get_renamable_variables(query: str) -> Set[str]:
    """
    In Wikidata, `SERVICE wikibase:label` automatically binds variables 
    like ?varLabel based on ?var. Renaming these breaks the service.
    This function excludes them to keep the queries valid.
    """
    vars_ = get_all_variables(query)
    # Identify variables associated with the label service
    label_vars = {v for v in vars_ if re.search(r'Label$|Description$|AltLabel$', v)}
    
    # Identify their corresponding base variables
    base_vars = {
        v.replace('Label', '').replace('Description', '').replace('AltLabel', '') 
        for v in label_vars
    }
    
    protected_vars = label_vars.union(base_vars)
    return vars_ - protected_vars


def is_valid_sparql(query: str) -> bool:
    try:
        parseQuery(query)
        return True
    except Exception:
        return False


def find_matching_brace(text: str, start: int) -> int:
    """Return the index of the '}' matching the '{' at position start, or -1."""
    if start >= len(text) or text[start] != '{':
        return -1
    depth = 0
    for i in range(start, len(text)):
        if text[i] == '{':
            depth += 1
        elif text[i] == '}':
            depth -= 1
            if depth == 0:
                return i
    return -1


def extract_where_clause(query: str):
    """
    Return (where_start, where_end, where_body) of the outermost WHERE clause.
    Accounts for Wikidata queries that omit the 'WHERE' keyword (e.g., SELECT * { ... }).
    """
    select_match = re.search(r'\bSELECT\b', query, re.IGNORECASE)
    if not select_match:
        return None
    
    brace_open = query.find('{', select_match.end())
    if brace_open == -1:
        return None
        
    brace_close = find_matching_brace(query, brace_open)
    if brace_close == -1:
        return None
        
    where_body = query[brace_open+1:brace_close].strip()
    return brace_open, brace_close, where_body


def get_top_level_elements(body: str) -> List[str]:
    """
    Safely tokenizes the body into top-level triples/blocks.
    Prevents breaking inside `SERVICE { }` blocks or property paths using semicolons.
    """
    elements = []
    depth = 0
    current = ""
    for char in body:
        current += char
        if char == '{':
            depth += 1
        elif char == '}':
            depth -= 1
            # If a block closes at depth 0, treat it as a cohesive element
            if depth == 0:
                elements.append(current.strip())
                current = ""
        elif char == '.' and depth == 0:
            # Safely split at dots that aren't nested
            elements.append(current.strip())
            current = ""
            
    if current.strip():
        elements.append(current.strip())
        
    return [e for e in elements if e]


# ---- Transformations ----

def apply_variable_renaming(query: str) -> str:
    vars_ = sorted(get_renamable_variables(query))
    if not vars_:
        return query
    mapping = {v: f'?v{i}' for i, v in enumerate(vars_)}
    new_query = query
    for old, new in mapping.items():
        new_query = re.sub(r'(?<!\w)' + re.escape(old) + r'(?!\w)', new, new_query)
    return new_query


def apply_triple_shuffle(query: str) -> str:
    parts = extract_where_clause(query)
    if not parts:
        return query
    brace_open, brace_close, where_body = parts

    elements = get_top_level_elements(where_body)
    if len(elements) <= 1:
        return query
        
    random.shuffle(elements)
    new_body = '\n  '.join(elements)
    
    return query[:brace_open+1] + "\n  " + new_body + "\n" + query[brace_close:]


def apply_union_duplication(query: str) -> str:
    parts = extract_where_clause(query)
    if not parts:
        return query
        
    if not re.search(r'\bDISTINCT\b', query[:parts[0]], re.IGNORECASE):
        query = re.sub(r'(SELECT)\s+', r'\1 DISTINCT ', query, count=1, flags=re.IGNORECASE)
        parts = extract_where_clause(query)
        if not parts:
            return query

    brace_open, brace_close, where_body = parts
    new_where = f'{{ {where_body} }} UNION {{ {where_body} }}'
    return query[:brace_open+1] + f" {new_where} " + query[brace_close:]


def apply_subquery_wrapping(query: str) -> str:
    select_match = re.search(r'\bSELECT\b', query, re.IGNORECASE)
    if not select_match:
        return query

    brace_open = query.find('{', select_match.end())
    if brace_open == -1:
        return query
        
    brace_close = find_matching_brace(query, brace_open)
    if brace_close == -1:
        return query

    prefix_part = query[:select_match.start()]
    select_part = query[select_match.start():brace_open].strip()
    where_body = query[brace_open+1:brace_close].strip()

    subquery = f'{select_part} WHERE {{ {where_body} }}'
    new_query = f'{prefix_part}{select_part} WHERE {{ {{ {subquery} }} }}' + query[brace_close+1:]
    return new_query


def apply_redundant_optional(query: str) -> str:
    parts = extract_where_clause(query)
    if not parts:
        return query
    brace_open, brace_close, where_body = parts

    elements = get_top_level_elements(where_body)
    triples = [e for e in elements if e.endswith('.')]
    if not triples:
        return query
        
    target = random.choice(triples)
    optional_block = f'OPTIONAL {{ {target} }}'
    new_body = where_body.rstrip() + '\n  ' + optional_block
    
    return query[:brace_open+1] + '\n  ' + new_body + '\n' + query[brace_close:]


def apply_filter_true(query: str) -> str:
    parts = extract_where_clause(query)
    if not parts:
        return query
    brace_open = parts[0]
    return query[:brace_open+1] + " FILTER(true) .\n  " + query[brace_open+1:]


def apply_bgp_split(query: str) -> str:
    parts = extract_where_clause(query)
    if not parts:
        return query
    brace_open, brace_close, where_body = parts

    elements = get_top_level_elements(where_body)
    if len(elements) <= 1:
        return query
        
    idx = random.randint(1, len(elements)-1)
    left = '\n  '.join(elements[:idx])
    right = '\n  '.join(elements[idx:])
    
    new_body = f'{{ {left} }} {{ {right} }}'
    return query[:brace_open+1] + '\n  ' + new_body + '\n' + query[brace_close:]


def apply_distinct_toggle(query: str) -> str:
    if re.search(r'\bDISTINCT\b', query, re.IGNORECASE):
        return re.sub(r'(SELECT\s+)DISTINCT\s+', r'\1', query, count=1, flags=re.IGNORECASE)
    else:
        return re.sub(r'(SELECT\s+)', r'\1DISTINCT ', query, count=1, flags=re.IGNORECASE)


TRANSFORMATIONS = {
    'variable_renaming': apply_variable_renaming,
    'triple_shuffle': apply_triple_shuffle,
    'union_duplication': apply_union_duplication,
    'subquery_wrapping': apply_subquery_wrapping,
    'redundant_optional': apply_redundant_optional,
    'filter_true': apply_filter_true,
    'bgp_split': apply_bgp_split,
    'distinct_toggle': apply_distinct_toggle,
}


def parse_queries(file_path):
    """
    Parses a triplet JSON/JSONL dataset and extracts only the 'anchor_query'.
    """
    queries = []
    with open(file_path, 'r', encoding='utf-8') as f:
        try:
            # 1. Try to read as a single JSON array of objects
            data = json.load(f)
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, dict) and "anchor_query" in item:
                        queries.append(item["anchor_query"])
            return queries
        except json.JSONDecodeError:
            pass

        # 2. If it's not a standard JSON array, try reading as JSON Lines (.jsonl)
        f.seek(0)
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
                if isinstance(item, dict) and "anchor_query" in item:
                    queries.append(item["anchor_query"])
            except json.JSONDecodeError:
                pass  # Skip lines that aren't valid JSON

    return queries


def main():
    parser_arg = argparse.ArgumentParser()
    parser_arg.add_argument('--input', required=True, help="Input Triplet JSON file")
    parser_arg.add_argument('--output', required=True, help="Output JSON file for generated pairs")
    parser_arg.add_argument('--seed', type=int, default=42)
    parser_arg.add_argument('--types', nargs='+',
                            default=['variable_renaming', 'triple_shuffle', 'union_duplication',
                                     'subquery_wrapping', 'redundant_optional', 'filter_true',
                                     'bgp_split', 'distinct_toggle'])
    args = parser_arg.parse_args()
    random.seed(args.seed)

    queries = parse_queries(args.input)
    print(f"Loaded {len(queries)} anchor queries from dataset.")

    dataset = []
    for idx, q in enumerate(queries):
        q = q.strip()
        if not q:
            continue

        # Try every transformation
        for ptype in args.types:
            if ptype not in TRANSFORMATIONS:
                continue
            try:
                perturbed = TRANSFORMATIONS[ptype](q)
                if not perturbed or perturbed == q:
                    continue
                # Keep only if the perturbed query is valid SPARQL
                if is_valid_sparql(perturbed):
                    dataset.append({
                        "id": idx,
                        "original": q,
                        "perturbed": perturbed,
                        "perturbation_type": ptype
                    })
            except Exception:
                pass

        if idx > 0 and idx % 100 == 0:
            print(f"Processed {idx} queries...")

    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)
    print(f"Done. Wrote {len(dataset)} perturbation pairs to {args.output}")


if __name__ == '__main__':
    main()