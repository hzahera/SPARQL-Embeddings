#!/usr/bin/env python3
"""Build the adversarial (low-lexical-similarity) Wikidata RQ1 test set.

Every anchor is paired only with result-equivalent positives whose lexical
similarity to the anchor is below 0.30 -- so text baselines have essentially no
surface signal and only result-aware / structural matching can retrieve. This is
the regime that tests the paper's thesis.

CRITICAL (post-fairness-fix): lexical similarity is measured on the SAME
label-expanded text the text baselines actually read at eval time
(`sparqlce.sparql_utils.expand_wikidata_labels`), NOT the raw opaque-id text.
Two queries can be cos<0.3 on raw `wdt:P31 wd:Q5` yet share tokens once expanded
to `instance of (P31) human (Q5)`; filtering on raw text would leave the
baselines residual signal and overstate the adversarial gap. The expansion is
inlined here (reads datasets/wikidata_labels.json) so this builder needs no torch.

Spec
----
* Source     : held-out Wikidata TEST split only (no train/val -> no leakage).
* Positive   : answer_jaccard > 0.8 (result-equivalent) AND expanded-text
               TF-IDF cosine to the anchor < 0.30, capped at 20/anchor.
* Anchor kept if it has >= MIN_POS such positives.
* Negatives  : the anchor's shipped 20 negatives (answer_jaccard = 0), matching
               wikidata_testset_RQ1.json.

Output: datasets/wikidata_testset_RQ1_adversarial.json

    python scripts/build_adversarial_wikidata.py
"""
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize as sk_normalize

ROOT = Path(__file__).resolve().parents[1]
TRIP = ROOT / "datasets" / "wikidata-triplets-dataset.json"
SPLITS = ROOT / "datasets" / "splits_wikidata.json"
LABELS = ROOT / "datasets" / "wikidata_labels.json"
OUT = ROOT / "datasets" / "wikidata_testset_RQ1_adversarial.json"

COS_MAX = 0.30
JAC_MIN = 0.80
CAP = 20
MIN_POS = 1
SEED = 42

_norm = lambda s: " ".join(s.split())

# Exact copy of sparqlce.sparql_utils._WD_EXPAND_RE so the adversarial filter uses
# byte-identical expansion to the eval-time text baselines.
_WD_EXPAND_RE = re.compile(
    r'<[^>]*?[/#]([PQ]\d+)>'
    r'|(?:[A-Za-z_][\w.\-]*:)?([PQ]\d+)\b'
)


def load_labels():
    if not LABELS.exists():
        raise SystemExit(
            f"{LABELS} not found — the adversarial filter needs the label map.\n"
            "Generate it with: python scripts/build_wikidata_labels.py")
    return json.load(open(LABELS, encoding="utf-8"))


def expand(query, labels):
    def repl(m):
        wid = m.group(1) or m.group(2)
        lab = labels.get(wid)
        return f"{lab} ({wid})" if lab else m.group(0)
    return _WD_EXPAND_RE.sub(repl, query)


def main():
    trip = json.load(open(TRIP))
    spl = json.load(open(SPLITS))
    labels = load_labels()
    print(f"labels loaded: {len(labels)} ids")
    test = [trip[i] for i in spl["test_idx"]]

    # expanded-text TF-IDF over the test anchors + positives
    exp = {}                                          # raw query -> expanded text
    def E(q):
        if q not in exp:
            exp[q] = expand(q, labels)
        return exp[q]
    texts = set()
    for e in test:
        texts.add(E(e["anchor_query"]))
        for p in e["positive_queries"]:
            texts.add(E(p["query"]))
    texts = list(texts)
    ridx = {t: i for i, t in enumerate(texts)}
    X = sk_normalize(
        TfidfVectorizer(analyzer="word", ngram_range=(1, 2)).fit_transform(texts),
        norm="l2", axis=1)
    cos = lambda a, p: float(X[ridx[E(a)]].multiply(X[ridx[E(p)]]).sum())

    out, all_cos, npos = [], [], []
    for e in test:
        a = e["anchor_query"]
        advs = []
        for p in e["positive_queries"]:
            j = p.get("answer_jaccard")
            if j is None or j <= JAC_MIN:
                continue
            c = cos(a, p["query"])
            if c < COS_MAX:
                advs.append((p["query"], c, float(j)))
        if len(advs) < MIN_POS:
            continue
        advs = sorted(advs, key=lambda t: t[1])[:CAP]     # hardest first, capped
        out.append({
            "anchor_query": a,
            "positive_queries": [
                {"query": q, "answer_jaccard": round(j, 4), "lexical_cos": round(c, 4)}
                for (q, c, j) in advs],
            "negative_queries": [
                {"query": n["query"],
                 "answer_jaccard": round(float(n.get("answer_jaccard", 0.0)), 4)}
                for n in e["negative_queries"]],
            "lexical_bin": "low",
        })
        all_cos += [c for _, c, _ in advs]
        npos.append(len(advs))

    if not out:
        raise SystemExit("No anchors qualify at cos<0.3 on expanded text — "
                         "loosen COS_MAX or check the label map coverage.")
    out.sort(key=lambda r: np.mean([p["lexical_cos"] for p in r["positive_queries"]]))
    json.dump(out, open(OUT, "w"), ensure_ascii=False, indent=1)

    corpus = set()
    for r in out:
        corpus.add(_norm(r["anchor_query"]))
        corpus.update(_norm(p["query"]) for p in r["positive_queries"])
        corpus.update(_norm(n["query"]) for n in r["negative_queries"])
    print(f"\nwrote {len(out)} anchors to {OUT}")
    print(f"  test anchors scanned: {len(test)}  ->  qualifying (>= {MIN_POS} "
          f"positive at cos<{COS_MAX}): {len(out)}")
    print(f"  positives/anchor: min {min(npos)} mean {np.mean(npos):.2f} max {max(npos)} "
          f"(total {sum(npos)})")
    print(f"  positive expanded-cos: min {min(all_cos):.3f} mean {np.mean(all_cos):.3f} "
          f"max {max(all_cos):.3f}  (all < {COS_MAX})")
    print(f"  positive jaccard: min "
          f"{min(p['answer_jaccard'] for r in out for p in r['positive_queries']):.3f}")
    print(f"  negatives/anchor: {len(out[0]['negative_queries'])}  (shipped, jaccard 0)")
    print(f"  unique corpus queries: {len(corpus)}")


if __name__ == "__main__":
    main()
