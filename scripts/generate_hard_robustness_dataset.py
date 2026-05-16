#!/usr/bin/env python3
"""
Generate a HARD robustness dataset for SPARQL embeddings.

Works even when rdflib's parser rejects some input queries.
"""

import argparse
import json
import random
import re
from typing import List, Set, Optional
from rdflib.plugins.sparql.parser import parseQuery


def get_all_variables(query: str) -> Set[str]:
    return set(re.findall(r'\?[a-zA-Z_]\w*', query))


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
    """Return (where_start, where_end, where_body) of the outermost WHERE clause, or None."""
    # Find the WHERE keyword
    match = re.search(r'\bWHERE\s*\{', query, re.IGNORECASE)
    if not match:
        return None
    brace_open = match.end() - 1  # position of '{'
    brace_close = find_matching_brace(query, brace_open)
    if brace_close == -1:
        return None
    where_body = query[brace_open+1:brace_close].strip()
    return brace_open, brace_close, where_body


def extract_projection(query: str):
    """Return the part before the WHERE clause (SELECT vars) and whether DISTINCT is used."""
    # Find the first WHERE
    where_match = re.search(r'\bWHERE\s*\{', query, re.IGNORECASE)
    if not where_match:
        return '', False
    proj_part = query[:where_match.start()].strip()
    distinct = bool(re.search(r'\bDISTINCT\b', proj_part, re.IGNORECASE))
    return proj_part, distinct


# ---- Transformations (identical to previous, just copied) ----
def apply_variable_renaming(query: str) -> str:
    vars_ = sorted(get_all_variables(query))
    if not vars_:
        return query
    mapping = {v: f'?v{i}' for i, v in enumerate(vars_)}
    new_query = query
    for old, new in mapping.items():
        new_query = re.sub(r'(?<!\w)' + re.escape(old) + r'(?!\w)', new, new_query)
    return new_query


def apply_triple_shuffle(query: str) -> str:
    match = re.search(r'\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}', query, re.DOTALL)
    if not match:
        return query
    body = match.group(1)
    triples = [t.strip() for t in re.split(r'\s*\.\s*\n?', body.strip()) if t.strip()]
    if len(triples) <= 1:
        return query
    random.shuffle(triples)
    new_body = ' .\n  '.join(triples) + ' .'
    new_query = query[:match.start()] + '{ ' + new_body + ' }' + query[match.end():]
    return new_query


def apply_union_duplication(query: str) -> str:
    """Duplicate the outermost WHERE clause with UNION, forcing DISTINCT."""
    parts = extract_where_clause(query)
    if not parts:
        return query
    brace_open, brace_close, where_body = parts
    # Force DISTINCT if not present
    proj, has_distinct = extract_projection(query)
    if not has_distinct:
        # add DISTINCT after SELECT
        query = re.sub(r'(SELECT)\s+', r'\1 DISTINCT ', query, count=1, flags=re.IGNORECASE)

    # Re-extract after DISTINCT insertion (the positions may have shifted)
    parts = extract_where_clause(query)
    if not parts:
        return query
    brace_open, brace_close, where_body = parts
    new_where = f'{{ {{ {where_body} }} UNION {{ {where_body} }} }}'
    new_query = query[:brace_open] + new_where + query[brace_close+1:]
    return new_query


def apply_subquery_wrapping(query: str) -> str:
    """Wrap the WHERE clause into a subquery: SELECT vars WHERE { { SELECT vars WHERE { ... } } }."""
    parts = extract_where_clause(query)
    if not parts:
        return query
    brace_open, brace_close, where_body = parts
    proj, _ = extract_projection(query)
    # Build subquery: SELECT vars WHERE { ... }
    subquery = f'{proj} WHERE {{ {where_body} }}'
    # Outer query: same projection, but WHERE { subquery }
    new_where = f'{{ {subquery} }}'
    new_query = query[:brace_open] + new_where + query[brace_close+1:]
    return new_query


def apply_redundant_optional(query: str) -> str:
    match = re.search(r'\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}', query, re.DOTALL)
    if not match:
        return query
    body = match.group(1)
    triples = [t.strip() for t in re.split(r'\s*\.\s*\n?', body.strip()) if t.strip()]
    if not triples:
        return query
    target = random.choice(triples)
    optional_block = f'OPTIONAL {{ {target} . }}'
    new_body = body.rstrip() + '\n  ' + optional_block
    new_query = query[:match.start()] + '{ ' + new_body + ' }' + query[match.end():]
    return new_query


def apply_filter_true(query: str) -> str:
    return re.sub(r'(\{)\s*', r'\1 FILTER(true) .\n  ', query, count=1)


def apply_bgp_split(query: str) -> str:
    match = re.search(r'\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}', query, re.DOTALL)
    if not match:
        return query
    body = match.group(1)
    triples = [t.strip() for t in re.split(r'\s*\.\s*\n?', body.strip()) if t.strip()]
    if len(triples) <= 2:
        return query
    idx = random.randint(1, len(triples)-1)
    left = ' . '.join(triples[:idx]) + ' .'
    right = ' . '.join(triples[idx:]) + ' .'
    new_body = f'{{ {left} }} {{ {right} }}'
    new_query = query[:match.start()] + new_body + query[match.end():]
    return new_query


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
    with open(file_path, 'r', encoding='utf-8') as f:
        try:
            data = json.load(f)
            if isinstance(data, list):
                return data
        except json.JSONDecodeError:
            pass
        f.seek(0)
        return [line.strip() for line in f if line.strip()]


def main():
    parser_arg = argparse.ArgumentParser()
    parser_arg.add_argument('--input', required=True)
    parser_arg.add_argument('--output', required=True)
    parser_arg.add_argument('--seed', type=int, default=42)
    parser_arg.add_argument('--types', nargs='+',
                            default=['variable_renaming', 'triple_shuffle', 'union_duplication',
                                     'subquery_wrapping', 'redundant_optional', 'filter_true',
                                     'bgp_split', 'distinct_toggle'])
    args = parser_arg.parse_args()
    random.seed(args.seed)

    queries = parse_queries(args.input)
    print(f"Loaded {len(queries)} queries.")

    dataset = []
    for idx, q in enumerate(queries):
        q = q.strip()
        if not q:
            continue

        # try every transformation – only validate the result
        for ptype in args.types:
            if ptype not in TRANSFORMATIONS:
                continue
            try:
                perturbed = TRANSFORMATIONS[ptype](q)
                if not perturbed or perturbed == q:
                    continue
                # keep only if the perturbed query is valid SPARQL
                if is_valid_sparql(perturbed):
                    dataset.append({
                        "id": idx,
                        "original": q,
                        "perturbed": perturbed,
                        "perturbation_type": ptype
                    })
            except Exception:
                pass

        if idx % 100 == 0:
            print(f"Processed {idx} queries...")

    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)
    print(f"Done. Wrote {len(dataset)} perturbation pairs to {args.output}")


if __name__ == '__main__':
    main()