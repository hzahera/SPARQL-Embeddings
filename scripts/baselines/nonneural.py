#!/usr/bin/env python3
"""Non-neural retrieval baselines for SQuARE under three protocols.
  A  sampled:   rank each anchor's stored candidates (<=20 pos + <=20 neg); relevance = stored J
  B  code:      rank all test-split queries (as evaluation.rank_record_global); unstored = non-relevant
  C  full:      rank all test-split queries; relevance from complete RQ3 pair labels
Methods: random, tfidf, bm25, normalized match, algebra similarity, query-form features.
Statistics (idf, feature scaling) are fitted on the training split only; no tuning.
Usage: PYTHONHASHSEED=0 python nonneural.py <repo_root> <dbpedia|wikidata> <out.json>"""
import json, math, re, sys, random
from collections import Counter
from multiprocessing import Pool
import numpy as np
sys.path.insert(0, f"{sys.argv[1]}/scripts/audit"); sys.path.insert(0, f"{sys.argv[1]}/src")
import audit_splits as au
from result_aware.metrics import hits_at_k, mrr_at_k, ndcg_at_k
from rdflib import URIRef, Literal, Variable, BNode
from rdflib.plugins.sparql.parser import parseQuery
from rdflib.plugins.sparql.algebra import translateQuery
from rdflib.plugins.sparql.parserutils import CompValue

root, g, out = sys.argv[1:4]
ws = au.ws
D = f"{root}/datasets/{g}-splits"
train = json.load(open(f"{D}/{g}-train.json")); test = json.load(open(f"{D}/{g}-test.json"))
rq3 = json.load(open(f"{D}/{g}-test-rq3.json"))

# ---------------- features
TOK = re.compile(r"<[^>]*>|\"(?:[^\"\\]|\\.)*\"|'(?:[^'\\]|\\.)*'|[?$]\w+|[\w\-]*:[\w\-./#]*|\w+|[^\s\w]")
def toks(q):
    out = []
    for t in TOK.findall(q):
        if t[0] in "?$": out.append("?var")          # variable names carry no data meaning
        else: out.append(t.lower() if re.fullmatch(r"[A-Za-z_]+", t) else t)
    return out

def struct_bag(q):
    """multiset of algebra node labels and parent>child label edges; variables abstracted"""
    try:
        alg = translateQuery(parseQuery(au.add_prefixes(q))).algebra
    except Exception:
        return None
    bag = Counter()
    def lab(x):
        if isinstance(x, CompValue): return x.name
        if isinstance(x, Variable): return "?v"
        if isinstance(x, BNode): return "_:b"
        if isinstance(x, (URIRef, Literal)): return str(x)
        return None
    def walk(node, parent):
        if isinstance(node, CompValue):
            l = node.name; bag[l] += 1
            if parent: bag[parent + ">" + l] += 1
            for k, v in node.items():
                if k.startswith("_") or k in ("PV", "datasetClause"): continue
                walk(v, l)
        elif isinstance(node, (list, tuple)):
            if len(node) == 3 and all(isinstance(t, (URIRef, Literal, Variable, BNode)) for t in node):
                tp = "TP(" + ",".join(lab(t) for t in node) + ")"; bag[tp] += 1
                if parent: bag[parent + ">TP"] += 1
                for t in node: bag["term:" + lab(t)] += 1
            else:
                for v in node: walk(v, parent)
        elif lab(node) is not None and parent:
            bag[parent + ">" + lab(node)] += 1
    walk(alg, None)
    return bag

FORM_KEYS = ["OPTIONAL", "UNION", "FILTER", "GROUP BY", "ORDER BY", "DISTINCT", "LIMIT", "OFFSET", "SERVICE",
             "VALUES", "MINUS", "BIND", "NOT EXISTS", "REGEX", "LANG"]
def form_vec(q, info):
    t = re.sub(r'"(?:[^"\\]|\\.)*"', '""', q).upper()
    v = [info.get("arity") or 0, info.get("n_tp") or 0, float(info.get("aggregate", False)), t.count("SELECT") - 1]
    v += [float(k in t) for k in FORM_KEYS]
    return np.array(v, dtype=float)

# ---------------- corpus
test_q = [ws(x["query"]) for x in rq3["queries"]]            # all test-split queries (473 / 1075)
idx = {q: i for i, q in enumerate(test_q)}
train_q = list(dict.fromkeys(ws(c) for r in train for c in [r["anchor_query"]] + [x["query"] for x in r["positive_queries"] + r["negative_queries"]]))
info = json.load(open(f"{root}/scripts/audit/audit_{g}_perquery.json"))

# tf-idf + bm25 with train statistics
train_toks = [toks(q) for q in train_q]; test_toks = [toks(q) for q in test_q]
N = len(train_toks); df = Counter(t for d in train_toks for t in set(d))
idf_tfidf = lambda t: math.log((1 + N) / (1 + df.get(t, 0))) + 1
def tfidf_vec(d):
    c = Counter(d); v = {t: (1 + math.log(n)) * idf_tfidf(t) for t, n in c.items()}
    norm = math.sqrt(sum(x * x for x in v.values())) or 1.0
    return {t: x / norm for t, x in v.items()}
TV = [tfidf_vec(d) for d in test_toks]
avgdl = sum(map(len, train_toks)) / N; k1, b = 1.2, 0.75
idf_bm25 = lambda t: math.log(1 + (N - df.get(t, 0) + 0.5) / (df.get(t, 0) + 0.5))
TC = [Counter(d) for d in test_toks]
def bm25(qi, di):
    d = TC[di]; L = len(test_toks[di]); s = 0.0
    for t in set(test_toks[qi]):
        f = d.get(t, 0)
        if f: s += idf_bm25(t) * f * (k1 + 1) / (f + k1 * (1 - b + b * L / avgdl))
    return s
with Pool(4) as p: SB = p.map(struct_bag, test_q, chunksize=20)
def wjacc(a, c):
    if a is None or c is None: return 0.0
    inter = sum((a & c).values()); union = sum((a | c).values())
    return inter / union if union else 0.0
ALG = [info[q]["algebra"] for q in test_q]
F_tr = np.array([form_vec(q, info[q]) for q in train_q]); mu, sd = F_tr.mean(0), F_tr.std(0) + 1e-9
F = np.array([(form_vec(q, info[q]) - mu) / sd for q in test_q])

def score(m, i, j):
    if m == "tfidf": return sum(x * TV[j].get(t, 0.0) for t, x in TV[i].items())
    if m == "bm25": return bm25(i, j)
    if m == "normalized_match": return float(ALG[i] == ALG[j])
    if m == "algebra_similarity": return wjacc(SB[i], SB[j])
    if m == "query_form": return -float(np.linalg.norm(F[i] - F[j]))
    if m == "random": return 0.0

# ---------------- labels
full = {}
for a, c, j in rq3["positive_pairs"] + rq3["negative_pairs"]:
    full[(a, c)] = full[(c, a)] = j
anchors = []
for r in test:
    a = idx[ws(r["anchor_query"])]
    stored = {idx[ws(x["query"])]: x["answer_jaccard"] for x in r["positive_queries"] + r["negative_queries"]}
    anchors.append((a, stored))

def evaluate(m, proto, reps):
    res = {"hits@1": [], "mrr@5": [], "ndcg@5": [], "mrr": [], "ndcg@10": []}
    for a, stored in anchors:
        if proto == "A": cands = list(stored); rel = lambda c: stored[c]
        elif proto == "B": cands = [c for c in range(len(test_q)) if c != a]; rel = lambda c: stored.get(c, 0.0)
        else:
            cands = [c for c in range(len(test_q)) if c != a]
            rel = lambda c: full.get((a, c), 0.0) if full.get((a, c), 0.0) >= 0.8 else 0.0   # intermediate/negative: non-relevant
        s = np.array([score(m, a, c) for c in cands]); R = np.array([rel(c) for c in cands])
        acc = {k: 0.0 for k in res}
        for rep in range(reps):                                   # random tie-breaking, averaged
            noise = np.random.default_rng(rep * 7919 + a).random(len(cands)) * 1e-9
            order = np.lexsort((noise, -s)) if m != "random" else np.argsort(noise)
            rels = list(R[order]); binr = [x >= 0.8 for x in rels]
            acc["hits@1"] += hits_at_k(binr, 1); acc["mrr@5"] += mrr_at_k(binr, 5); acc["ndcg@5"] += ndcg_at_k(rels, 5)
            acc["mrr"] += mrr_at_k(binr, len(binr)); acc["ndcg@10"] += ndcg_at_k(rels, 10)
        for k in res: res[k].append(acc[k] / reps)
    return {k: round(float(np.mean(v)), 4) for k, v in res.items()}

methods = ["random", "tfidf", "bm25", "normalized_match", "algebra_similarity", "query_form"]
results = {"graph": g, "anchors": len(anchors), "test_queries": len(test_q)}
for proto in ("A", "B", "C"):
    for m in methods:
        reps = 50 if m in ("random", "normalized_match") else (3 if m == "query_form" else 1)
        results[f"{proto}:{m}"] = evaluate(m, proto, reps); print(proto, m, results[f"{proto}:{m}"], flush=True)
pos = rq3["positive_pairs"]
results["normalized_match_recovers_positive_pairs_share"] = sum(ALG[a] == ALG[c] for a, c, _ in pos) / len(pos)
results["struct_parse_failures"] = sum(x is None for x in SB)
json.dump(results, open(out, "w"), indent=2); print(json.dumps(results, indent=1))
