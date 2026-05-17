#!/usr/bin/env python3
"""Add union_duplication and subquery_wrapping pairs to the robustness dataset."""

import json, random, re

def find_matching_brace(text, start):
    depth = 0
    for i in range(start, len(text)):
        if text[i] == '{':
            depth += 1
        elif text[i] == '}':
            depth -= 1
            if depth == 0:
                return i
    return -1

def get_where_body(query):
    """Return (before_where, where_body, after_where) using brace matching."""
    # find WHERE {
    match = re.search(r'\bWHERE\s*\{', query, re.IGNORECASE)
    if not match:
        return None
    brace_open = match.end() - 1
    brace_close = find_matching_brace(query, brace_open)
    if brace_close == -1:
        return None
    before = query[:brace_open+1]   # includes the opening {
    body = query[brace_open+1:brace_close]
    after = query[brace_close:]     # includes the closing }
    return before, body, after

def add_union_duplication(q):
    parts = get_where_body(q)
    if not parts:
        return None
    before, body, after = parts
    # force DISTINCT if not present
    if not re.search(r'\bDISTINCT\b', q, re.IGNORECASE):
        q = re.sub(r'(SELECT)\s+', r'\1 DISTINCT ', q, count=1, flags=re.IGNORECASE)
        # re‑extract because positions changed
        parts = get_where_body(q)
        if not parts:
            return None
        before, body, after = parts
    new_where = f'{{ {{ {body} }} UNION {{ {body} }} }}'
    return before + new_where + after

def add_subquery_wrapping(q):
    parts = get_where_body(q)
    if not parts:
        return None
    before, body, after = parts
    # extract SELECT vars
    proj_match = re.search(r'(SELECT\s+(?:DISTINCT\s+)?.*?)\s*WHERE\s*\{', q, re.IGNORECASE | re.DOTALL)
    if not proj_match:
        return None
    proj = proj_match.group(1).strip()
    subquery = f'{proj} WHERE {{ {body} }}'
    new_query = before + f'{{ {subquery} }}' + after
    return new_query

def main():
    with open('data/robustness_hard_dbpedia.json') as f:
        pairs = json.load(f)

    # Collect unique original queries that have a simple BGP (triple_shuffle succeeded for them)
    # We'll use triple_shuffle pairs as candidates because they have clean BGPs
    candidates = set()
    for p in pairs:
        if p['perturbation_type'] == 'triple_shuffle':
            candidates.add(p['original'])
    
    print(f"Found {len(candidates)} candidate queries with working BGPs")
    
    random.seed(42)
    samples = random.sample(list(candidates), min(100, len(candidates)))
    
    new_pairs = []
    for q in samples:
        # union_duplication
        uq = add_union_duplication(q)
        if uq and uq != q:
            new_pairs.append({
                "id": len(pairs) + len(new_pairs),
                "original": q,
                "perturbed": uq,
                "perturbation_type": "union_duplication"
            })
        # subquery_wrapping
        sq = add_subquery_wrapping(q)
        if sq and sq != q:
            new_pairs.append({
                "id": len(pairs) + len(new_pairs),
                "original": q,
                "perturbed": sq,
                "perturbation_type": "subquery_wrapping"
            })
    
    print(f"Added {len(new_pairs)} new pairs")
    pairs.extend(new_pairs)
    
    with open('data/robustness_hard_dbpedia.json', 'w') as f:
        json.dump(pairs, f, indent=2, ensure_ascii=False)
    
    from collections import Counter
    print("New counts:", Counter(p['perturbation_type'] for p in pairs))

if __name__ == '__main__':
    main()