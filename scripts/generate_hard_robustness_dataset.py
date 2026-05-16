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
    if not re.search(r'\bDISTINCT\b', query, re.IGNORECASE):
        query = re.sub(r'(SELECT\s+)', r'\1DISTINCT ', query, count=1, flags=re.IGNORECASE)
    where_match = re.search(r'WHERE\s*\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}', query, re.DOTALL)
    if not where_match:
        return query
    where_body = where_match.group(1).strip()
    new_where = f'{{ {where_body} }} UNION {{ {where_body} }}'
    new_query = query[:where_match.start()] + 'WHERE ' + new_where + query[where_match.end():]
    return new_query


def apply_subquery_wrapping(query: str) -> str:
    proj_match = re.search(r'(SELECT\s+(DISTINCT\s+)?)(.*?)\s*WHERE\s*\{', query, re.DOTALL | re.IGNORECASE)
    if not proj_match:
        return query
    vars_str = proj_match.group(3).strip()
    where_match = re.search(r'WHERE\s*\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}', query, re.DOTALL)
    if not where_match:
        return query
    where_body = where_match.group(1).strip()
    subquery = f'SELECT {vars_str} WHERE {{ {where_body} }}'
    new_query = query[:proj_match.start()] + f'SELECT {vars_str} WHERE {{ ' + subquery + ' }}' + query[where_match.end():]
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