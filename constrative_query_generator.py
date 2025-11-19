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


def sample_random_negative(global_pool, forbidden_stripped, max_tries=50):
    if not global_pool:
        return None

    # random attempts
    for _ in range(max_tries):
        q = random.choice(global_pool)
        if not q:
            continue
        if q.strip() not in forbidden_stripped:
            return q

    # fallback: linear scan
    for q in global_pool:
        if q and q.strip() not in forbidden_stripped:
            return q

    return None


def build_contrastive_csv(
    json_path: str,
    csv_path: str,
     
    pos_jaccard_min: float = 0.9,
    pos_cosine_min: float = 0.7,
    max_positives: int = 5,

    hard_jaccard_max: float = 0.6,
    hard_cosine_min: float = 0.2,
    hard_cosine_max: float = 0.6,

    max_negatives: int = 5,
    max_hard_negatives: int = 3,
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
            continue

        pos_candidates.sort(key=lambda x: x[0], reverse=True)
        positive_queries = [sq for _, sq in pos_candidates[:max_positives]]

        # HARD NEGATIVES
        hard_neg_candidates = []

        for s in sims:
            jacc = safe_float(s.get("jaccard_score", 0.0), 0.0)
            cos = safe_float(s.get("cosine_score", 1.0), 1.0)
            sq = s.get("similar_query")

            if not sq:
                continue

            sq_stripped = sq.strip()
            if sq_stripped == base_query_stripped:
                continue

            if any(sq_stripped == p.strip() for p in positive_queries):
                continue

            if (jacc < hard_jaccard_max and
                hard_cosine_min <= cos <= hard_cosine_max):
                hard_neg_candidates.append((cos, sq))

        negative_queries = []

        if hard_neg_candidates:
            hard_neg_candidates.sort(key=lambda x: x[0])  # lowest cosine first
            for _, sq in hard_neg_candidates[:max_hard_negatives]:
                if len(negative_queries) >= max_negatives:
                    break
                negative_queries.append(sq)

        # RANDOM NEGATIVES 
        forbidden = {base_query_stripped}
        forbidden.update(p.strip() for p in positive_queries)
        forbidden.update(n.strip() for n in negative_queries)

        while len(negative_queries) < max_negatives:
            random_neg = sample_random_negative(global_pool, forbidden)
            if random_neg is None:
                break
            negative_queries.append(random_neg)
            forbidden.add(random_neg.strip())

        # must have at least 1 negative
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

    # 3) Write CSV
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
        sys.exit(1)

    json_path = sys.argv[1]
    csv_path = sys.argv[2]

    if not os.path.exists(json_path):
        print(f" Error: File not found: {json_path}")
        sys.exit(1)

    build_contrastive_csv(json_path, csv_path)


if __name__ == "__main__":
    main()
