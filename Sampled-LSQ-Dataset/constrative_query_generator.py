# Usage: python constrative_query_generator.py input.json output.csv

import json
import csv
import sys
import os
import random


def safe_float(x, default):
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def build_global_query_pool(data):
    pool = []
    seen = set()

    for item in data:
        base_query = item.get("base_query")
        if base_query:
            s = base_query.strip()
            if s and s not in seen:
                seen.add(s)
                pool.append(base_query)

        sims = item.get("similar_queries", [])
        for s_item in sims:
            sq = s_item.get("similar_query")
            if not sq:
                continue
            s2 = sq.strip()
            if s2 and s2 not in seen:
                seen.add(s2)
                pool.append(sq)

    return pool


def tokenize_query(q: str):
    return set(q.lower().split())


def jaccard_similarity(q1: str, q2: str) -> float:
    s1 = tokenize_query(q1)
    s2 = tokenize_query(q2)
    if not s1 and not s2:
        return 0.0
    inter = s1.intersection(s2)
    uni = s1.union(s2)
    return len(inter) / len(uni)


def sample_random_negative_with_jaccard(
    global_pool,
    base_query_stripped,
    neg_jaccard_max,
    forbidden_stripped,
    max_tries=100,
):
    if not global_pool:
        return None

    # random attempts first
    for _ in range(max_tries):
        q = random.choice(global_pool)
        if not q:
            continue

        qs = q.strip()
        if qs in forbidden_stripped:
            continue

        jacc = jaccard_similarity(base_query_stripped, qs)
        if jacc < neg_jaccard_max:
            return q

    # fallback: linear scan (in case random sampling missed some)
    for q in global_pool:
        if not q:
            continue
        qs = q.strip()
        if qs in forbidden_stripped:
            continue

        jacc = jaccard_similarity(base_query_stripped, qs)
        if jacc < neg_jaccard_max:
            return q

    return None


def build_contrastive_csv(
    json_path: str,
    csv_path: str,
    # POSITIVE selection
    pos_jaccard_min: float = 0.9,
    pos_cosine_min: float = 0.7,
    max_positives: int = 5,
    # NEGATIVE selection
    neg_jaccard_max: float = 0.4,
    max_negatives: int = 5,
):

    random.seed(42)

    # 1) Load JSON
    print(f"Loading JSON from: {json_path}")
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"Loaded {len(data)} items")

    # 2) Build global pool for random negatives
    global_pool = build_global_query_pool(data)
    print(f"Global pool size: {len(global_pool)} queries")

    rows = []
    ex_id = 0

    for idx, item in enumerate(data):
        base_query = item.get("base_query")
        sims = item.get("similar_queries", [])

        if not base_query or not sims:
            continue

        base_query_stripped = base_query.strip()

        # POSITIVES
        pos_candidates = []
        for s in sims:
            jacc = safe_float(s.get("jaccard_score", 0.0), 0.0)
            cos = safe_float(s.get("cosine_score", 0.0), 0.0)

            if jacc >= pos_jaccard_min and cos >= pos_cosine_min:
                sq = s.get("similar_query")
                if not sq:
                    continue
                if sq.strip() == base_query_stripped:
                    continue
                pos_candidates.append((cos, sq))

        if not pos_candidates:
            # no suitable positives, skip this base query
            continue

        # sort positives by cosine descending and take top-k
        pos_candidates.sort(key=lambda x: x[0], reverse=True)
        positive_queries = [sq for _, sq in pos_candidates[:max_positives]]

        # NEGATIVES: random queries with low Jaccard
        negative_queries = []

        forbidden = {base_query_stripped}
        forbidden.update(p.strip() for p in positive_queries)

        # sample until we hit max_negatives or run out of options
        while len(negative_queries) < max_negatives:
            random_neg = sample_random_negative_with_jaccard(
                global_pool=global_pool,
                base_query_stripped=base_query_stripped,
                neg_jaccard_max=neg_jaccard_max,
                forbidden_stripped=forbidden,
            )
            if random_neg is None:
                break
            negative_queries.append(random_neg)
            forbidden.add(random_neg.strip())

        if not negative_queries:
            continue

        rows.append({
            "id": ex_id,
            "anchor_query": base_query,
            "positive_queries": json.dumps(positive_queries, ensure_ascii=False),
            "negative_queries": json.dumps(negative_queries, ensure_ascii=False),
        })
        ex_id += 1

    print(f"Prepared {len(rows)} rows, writing CSV to: {csv_path}")

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["id", "anchor_query", "positive_queries", "negative_queries"],
        )
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    print(f"✔ Done. Wrote {len(rows)} rows to {csv_path}")


def main():
    if len(sys.argv) != 3:
        print("Usage: python constrative_query_generator.py input.json output.csv")
        sys.exit(1)

    json_path = sys.argv[1]
    csv_path = sys.argv[2]

    if not os.path.exists(json_path):
        print(f"Error: File not found: {json_path}")
        sys.exit(1)

    build_contrastive_csv(json_path, csv_path)


if __name__ == "__main__":
    main()
