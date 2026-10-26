"""Simulate the training sampler (B=4 anchors; 1 positive + 4 explicit negatives each) and measure
how many cross-anchor candidates could be unlabeled positives. Candidates in another positive
component have J<0.8 by construction, so only unmasked candidates from the anchor's own component
can be hidden positives. Usage: python batch_audit.py <splits_dir> <graph>"""
import json, random, re, sys
import networkx as nx
ws = lambda q: re.sub(r"\s+", " ", q.strip())
d, g = sys.argv[1], sys.argv[2]
recs = json.load(open(f"{d}/{g}-train.json"))
G = nx.Graph()
for r in recs:
    a = ws(r["anchor_query"]); G.add_node(a)
    for p in r["positive_queries"]: G.add_edge(a, ws(p["query"]))
comp = {q: i for i, c in enumerate(nx.connected_components(G)) for q in c}
anch = [ws(r["anchor_query"]) for r in recs]
pos = [{ws(p["query"]) for p in r["positive_queries"]} for r in recs]
negs = [[ws(n["query"]) for n in r["negative_queries"]] for r in recs]
rng = random.Random(0); tot = same = masked = 0
for epoch in range(5):
    idx = list(range(len(recs))); rng.shuffle(idx)
    for b in range(0, len(idx) - 3, 4):
        batch = idx[b:b + 4]; cands = []
        for j in batch:
            cands.append((j, rng.choice(sorted(pos[j]))))
            cands += [(j, n) for n in rng.sample(negs[j], 4)]
        for i in batch:
            for j, c in cands:
                if j == i: continue
                tot += 1
                if c == anch[i] or c in pos[i]: masked += 1
                elif comp.get(c, -1) == comp[anch[i]]: same += 1
print(json.dumps({"graph": g, "anchors_at_20_positive_cap": sum(len(p) == 20 for p in pos),
  "cross_candidates": tot, "masked_share": round(masked / tot, 5),
  "possible_hidden_positive_share": round(same / tot, 5)}))
