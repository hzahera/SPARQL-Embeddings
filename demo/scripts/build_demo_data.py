#!/usr/bin/env python3
"""Build the data file used by demo/index.html from the released SQuARE splits.

Everything this script writes comes from the dataset files themselves:
  * real test anchors and their stored candidates with their answer-set Jaccard J
  * TF-IDF cosine (the strongest non-neural baseline in the paper), fitted on the
    training split only, as in scripts/baselines/nonneural.py
  * a sample of (TF-IDF cosine, J) pairs for the "text vs answers" scatter plot
  * TF-IDF rankings over all test queries, judged with the complete pair labels
    (datasets/*-splits/*-test-rq3.json), as in the paper's complete-label table

No model checkpoint is needed. Usage (from the repository root):

    python demo/scripts/build_demo_data.py --repo . --out demo/data/demo_data.js
"""
from __future__ import annotations

import argparse
import json
import math
import random
import re
from collections import Counter
from pathlib import Path

TOK = re.compile(
    r"<[^>]*>|\"(?:[^\"\\]|\\.)*\"|'(?:[^'\\]|\\.)*'|[?$]\w+|[\w\-]*:[\w\-./#]*|\w+|[^\s\w]"
)


def ws(q: str) -> str:
    return " ".join(q.split())


def toks(q: str) -> list[str]:
    out = []
    for t in TOK.findall(q):
        if t[0] in "?$":
            out.append("?var")
        else:
            out.append(t.lower() if re.fullmatch(r"[A-Za-z_]+", t) else t)
    return out


def strip_prefixes(q: str) -> str:
    lines = [l for l in q.splitlines() if not l.strip().upper().startswith("PREFIX")]
    text = "\n".join(lines).strip()
    return re.sub(r"\n{2,}", "\n", text)


def spearman(x: list[float], y: list[float]) -> float:
    def ranks(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(v):
            j = i
            while j + 1 < len(v) and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2 + 1
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r

    rx, ry = ranks(x), ranks(y)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    cov = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    vx = math.sqrt(sum((a - mx) ** 2 for a in rx))
    vy = math.sqrt(sum((b - my) ** 2 for b in ry))
    return cov / (vx * vy)


def build(repo: Path, graph: str, n_examples: int, n_scatter: int, seed: int) -> dict:
    d = repo / "datasets" / f"{graph}-splits"
    train = json.loads((d / f"{graph}-train.json").read_text())
    test = json.loads((d / f"{graph}-test.json").read_text())

    train_q = list(
        dict.fromkeys(
            ws(c)
            for r in train
            for c in [r["anchor_query"]] + [x["query"] for x in r["positive_queries"] + r["negative_queries"]]
        )
    )
    train_toks = [toks(q) for q in train_q]
    n = len(train_toks)
    df = Counter(t for doc in train_toks for t in set(doc))

    def idf(t):
        return math.log((1 + n) / (1 + df.get(t, 0))) + 1

    cache: dict[str, dict[str, float]] = {}

    def vec(q):
        key = ws(q)
        if key not in cache:
            c = Counter(toks(key))
            v = {t: (1 + math.log(k)) * idf(t) for t, k in c.items()}
            norm = math.sqrt(sum(x * x for x in v.values())) or 1.0
            cache[key] = {t: x / norm for t, x in v.items()}
        return cache[key]

    def cos(a, b):
        va, vb = vec(a), vec(b)
        if len(va) > len(vb):
            va, vb = vb, va
        return sum(x * vb.get(t, 0.0) for t, x in va.items())

    rng = random.Random(seed)

    # ---- (1) RQ1-style scatter over the stored test pairs (same pairs as the paper's Spearman)
    pairs = []
    for r in test:
        a = r["anchor_query"]
        for x in r["positive_queries"] + r["negative_queries"]:
            pairs.append((cos(a, x["query"]), x["answer_jaccard"]))
    rho = spearman([p[0] for p in pairs], [p[1] for p in pairs])
    rng.shuffle(pairs)
    sample = pairs[:n_scatter]

    # ---- (2) Retrieval among all test queries with complete labels (paper Table 3 protocol)
    rq3 = json.loads((d / f"{graph}-test-rq3.json").read_text())
    queries = rq3["queries"]
    qtext = [x["query"] for x in queries]
    n_ans = [x.get("n_answers") for x in queries]
    idx = {ws(q): i for i, q in enumerate(qtext)}
    J: dict[tuple[int, int], float] = {}
    for a_, c_, j_ in rq3["positive_pairs"] + rq3["negative_pairs"]:
        J[(a_, c_)] = J[(c_, a_)] = j_

    def jlab(a_, c_):
        # pairs missing from both lists have intermediate overlap 0.1 < J < 0.8
        return J.get((a_, c_))

    hits = []
    per_anchor = []
    for r in test:
        a_ = idx[ws(r["anchor_query"])]
        cands = [c for c in range(len(qtext)) if c != a_]
        sc = {c: cos(qtext[a_], qtext[c]) for c in cands}
        order = sorted(cands, key=lambda c: (-sc[c], rng.random()))
        rel = [(jlab(a_, c) or 0.0) >= 0.8 for c in order]
        hits.append(1.0 if rel[0] else 0.0)
        if any(rel):
            first_rel = rel.index(True)
            per_anchor.append((a_, order, sc, first_rel))

    def shape(q):
        q = strip_prefixes(q)
        q = re.sub(r'"[^"]*"', '"x"', q)
        q = re.sub(r"\b(wde|wd|dbpr|dbr|dbo|wdt|dbpo|dbp)?:[\w\-.%()',]+", "C", q)
        return " ".join(q.split())[:90]

    def readable(i):
        return len(strip_prefixes(qtext[i])) <= 320

    fails = [x for x in per_anchor if x[3] > 0 and readable(x[0])
             and all(readable(c) for c in x[1][: x[3] + 1][:6])]
    # strongest failures first: the wrong top match looks much more similar than the right one
    fails.sort(key=lambda x: -(x[2][x[1][0]] - x[2][x[1][x[3]]]))
    wins = [x for x in per_anchor if x[3] == 0 and readable(x[0]) and all(readable(c) for c in x[1][:5])]
    rng.shuffle(wins)

    chosen, seen = [], Counter()
    for x in fails:
        s = shape(qtext[x[0]])
        if seen[s] >= 1:
            continue
        seen[s] += 1
        chosen.append(x)
        if len(chosen) >= n_examples - 2:
            break
    chosen += wins[:2]

    examples = []
    for a_, order, sc, first_rel in chosen:
        show = order[:5]
        if first_rel >= 5:
            show = order[:4] + [order[first_rel]]
        # the highest-overlap positive anywhere in the collection (what an answer-aware ranker should put first)
        best = max((c for c in order if (jlab(a_, c) or 0) >= 0.8), key=lambda c: (jlab(a_, c), sc[c]))
        examples.append({
            "anchor": strip_prefixes(qtext[a_]),
            "anchor_answers": n_ans[a_],
            "text_rank_of_first_match": first_rel + 1,
            "n_relevant": sum((jlab(a_, c) or 0) >= 0.8 for c in order),
            "candidates": [{
                "q": strip_prefixes(qtext[c]),
                "tfidf": round(sc[c], 3),
                "J": None if jlab(a_, c) is None else round(jlab(a_, c), 3),
                "rank": order.index(c) + 1,
                "answers": n_ans[c],
                "best": c == best,
            } for c in dict.fromkeys(show + [best])],
        })

    return {
        "graph": graph,
        "test_queries": len(qtext),
        "test_anchors": len(test),
        "stored_pairs": len(pairs),
        "tfidf_hits1_complete": round(sum(hits) / len(hits), 3),
        "tfidf_spearman": round(rho, 3),
        "scatter": [[round(s_, 3), round(j_, 3)] for s_, j_ in sample],
        "examples": examples,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=".")
    ap.add_argument("--out", default="demo/data/demo_data.js")
    ap.add_argument("--examples", type=int, default=12)
    ap.add_argument("--scatter", type=int, default=1600)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    repo = Path(args.repo)
    data = {g: build(repo, g, args.examples, args.scatter, args.seed) for g in ("dbpedia", "wikidata")}
    for g, v in data.items():
        print(g, "anchors", v["test_anchors"], "pairs", v["stored_pairs"],
              "TF-IDF Hits@1 (complete labels)", v["tfidf_hits1_complete"], "rho", v["tfidf_spearman"],
              "examples", len(v["examples"]))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("// Generated by demo/scripts/build_demo_data.py from the SQuARE test splits.\n"
                   "window.DEMO_DATA = " + json.dumps(data, separators=(",", ":")) + ";\n")
    print("wrote", out, out.stat().st_size, "bytes")


if __name__ == "__main__":
    main()
