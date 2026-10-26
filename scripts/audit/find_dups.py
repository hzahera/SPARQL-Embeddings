"""List queries whose algebra-normalized form occurs in two splits."""
import json, re, sys
from collections import defaultdict
ws = lambda q: re.sub(r"\s+", " ", q.strip())
d, g = sys.argv[1], sys.argv[2]
info = json.load(open(f"audit_{g}_perquery.json"))
where = {}
for s in ("train", "valid", "test"):
    for r in json.load(open(f"{d}/{g}-{s}.json")):
        for q in [r["anchor_query"]] + [x["query"] for x in r["positive_queries"] + r["negative_queries"]]:
            where[ws(q)] = s
grp = defaultdict(list)
for q, i in info.items(): grp[i["algebra"]].append(q)
out = []
for k, qs in grp.items():
    if len({where[q] for q in qs}) > 1:
        out.append([{"split": where[q], "query": q} for q in qs])
json.dump(out, open(f"crosssplit_duplicates_{g}.json", "w"), indent=1)
for grp_ in out:
    print("----"); [print(x["split"], "|", x["query"][:400]) for x in grp_]
