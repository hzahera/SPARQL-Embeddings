"""Feasibility of a component+template-disjoint split: size of groups after merging
positive components with template classes. Usage: python template_groups.py <splits_dir> <graph>"""
import json, re, sys
import networkx as nx
d, g = sys.argv[1], sys.argv[2]
ws = lambda q: re.sub(r"\s+", " ", q.strip())
info = json.load(open(f"audit_{g}_perquery.json"))
G = nx.Graph(); G.add_nodes_from(info)
for s in ("train", "valid", "test"):
    for r in json.load(open(f"{d}/{g}-{s}.json")):
        for x in r["positive_queries"]: G.add_edge(ws(r["anchor_query"]), ws(x["query"]))
by_t = {}
for q, i in info.items(): by_t.setdefault(i["template"], []).append(q)
tc = sorted((len(v) for v in by_t.values()), reverse=True)
for qs in by_t.values():
    for q in qs[1:]: G.add_edge(qs[0], q)
comps = sorted((len(c) for c in nx.connected_components(G)), reverse=True)
print(json.dumps({"graph": g, "templates": len(tc), "largest_template_share": round(tc[0] / len(info), 3),
  "top10_templates_share": round(sum(tc[:10]) / len(info), 3), "merged_groups": len(comps),
  "largest_merged_group_share": round(comps[0] / len(info), 3), "top3_groups": comps[:3]}))
