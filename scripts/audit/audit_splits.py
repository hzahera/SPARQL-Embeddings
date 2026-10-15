#!/usr/bin/env python3
"""Run with PYTHONHASHSEED=0 (rdflib builds some projection lists from sets).
W1 audit for SQuARE splits: cross-split leakage (string / algebra-normalized / template),
positive components, template overlap, J=1 share, aggregates/slicing, projection arity.
Usage: python audit_splits.py <splits_dir> <dbpedia|wikidata> <out.json>   (needs rdflib, networkx)"""
import hashlib, json, re, sys, statistics as st
from collections import Counter, defaultdict
from multiprocessing import Pool
import networkx as nx
from rdflib import URIRef, Literal, RDF
from rdflib.plugins.sparql.parser import parseQuery
from rdflib.plugins.sparql.algebra import translateQuery, traverse
from rdflib.plugins.sparql.parserutils import CompValue

STD = {"wd":"http://www.wikidata.org/entity/","wdt":"http://www.wikidata.org/prop/direct/",
 "wikibase":"http://wikiba.se/ontology#","p":"http://www.wikidata.org/prop/","ps":"http://www.wikidata.org/prop/statement/",
 "pq":"http://www.wikidata.org/prop/qualifier/","psv":"http://www.wikidata.org/prop/statement/value/",
 "pqv":"http://www.wikidata.org/prop/qualifier/value/","wdno":"http://www.wikidata.org/prop/novalue/",
 "wds":"http://www.wikidata.org/entity/statement/","wdv":"http://www.wikidata.org/value/","wdref":"http://www.wikidata.org/reference/",
 "pr":"http://www.wikidata.org/prop/reference/","prov":"http://www.w3.org/ns/prov#","bd":"http://www.bigdata.com/rdf#",
 "schema":"http://schema.org/","rdf":"http://www.w3.org/1999/02/22-rdf-syntax-ns#","rdfs":"http://www.w3.org/2000/01/rdf-schema#",
 "xsd":"http://www.w3.org/2001/XMLSchema#","owl":"http://www.w3.org/2002/07/owl#","skos":"http://www.w3.org/2004/02/skos/core#",
 "foaf":"http://xmlns.com/foaf/0.1/","dc":"http://purl.org/dc/elements/1.1/","dct":"http://purl.org/dc/terms/",
 "dcterms":"http://purl.org/dc/terms/","geo":"http://www.w3.org/2003/01/geo/wgs84_pos#","georss":"http://www.georss.org/georss/",
 "dbo":"http://dbpedia.org/ontology/","dbp":"http://dbpedia.org/property/","dbr":"http://dbpedia.org/resource/",
 "dbpedia-owl":"http://dbpedia.org/ontology/","dbpprop":"http://dbpedia.org/property/","dbpedia":"http://dbpedia.org/resource/",
 "yago":"http://dbpedia.org/class/yago/","category":"http://dbpedia.org/resource/Category:","dbc":"http://dbpedia.org/resource/Category:",
 "bif":"http://www.openlinksw.com/schemas/bif#"}
DECL = re.compile(r"PREFIX\s+([\w\-]*):", re.I)
ws = lambda q: re.sub(r"\s+", " ", q.strip())
h = lambda s: hashlib.sha1(s.encode()).hexdigest()

def add_prefixes(q):
    declared = {m.lower() for m in DECL.findall(q)}
    used = set(re.findall(r"(?<![\w<?$])([A-Za-z][\w\-]*):[\w]", re.sub(r"<[^>]*>|\"[^\"]*\"|'[^']*'", "", q)))
    extra = [f"PREFIX {p}: <{STD[p]}>" for p in used if p.lower() not in declared and p in STD]
    return "\n".join(extra) + "\n" + q if extra else q

def var_rename(text, sort_pv=False):
    # drop rdflib's unordered '_vars' sets first: their order depends on the hash seed
    text = re.sub(r", '_vars': (?:set\(\)|\{[^{}]*\})", "", text)
    names = {}
    def sub(m):
        names.setdefault(m.group(1), f"v{len(names)}"); return f"Variable('{names[m.group(1)]}')"
    text = re.sub(r"Variable\('([^']*)'\)", sub, text)
    # rdflib derives some projection lists from sets: sort them after renaming
    if not sort_pv:
        return text
    return re.sub(r"'PV': \[([^\[\]]*)\]", lambda m: "'PV': [" + ", ".join(sorted(m.group(1).split(", "))) + "]", text)

def templ(node):  # mask literals; mask subject/object IRIs in BGPs (keep predicates and rdf:type classes)
    if isinstance(node, CompValue) and node.name == "BGP":
        C = URIRef("urn:C"); new = []
        for s, p, o in node.triples:
            new.append((C if isinstance(s, (URIRef, Literal)) else s, p,
                        o if p == RDF.type or not isinstance(o, (URIRef, Literal)) else C))
        n = CompValue("BGP", triples=new); n.update({k: v for k, v in node.items() if k != "triples"}); return n
    if isinstance(node, Literal): return Literal("C")

def bgps(node, acc):
    if isinstance(node, CompValue):
        if node.name == "BGP": acc.append(node)
        for v in node.values(): bgps(v, acc)
    elif isinstance(node, (list, tuple)):
        for v in node: bgps(v, acc)
    return acc

def analyze(q):
    t = re.sub(r'"(?:[^"\\]|\\.)*"|\'(?:[^\'\\]|\\.)*\'', '""', q).upper()
    r = {"ws": h(ws(q)),
         "aggregate": bool(re.search(r"\b(COUNT|SUM|AVG|MIN|MAX|GROUP_CONCAT|SAMPLE)\s*\(", t) or "GROUP BY" in t),
         "limit": bool(re.search(r"\bLIMIT\s+\d+", t)), "offset": bool(re.search(r"\bOFFSET\s+\d+", t)),
         "order": "ORDER BY" in t, "distinct": bool(re.search(r"SELECT\s+(DISTINCT|REDUCED)", t)), "service": "SERVICE" in t}
    m = re.search(r"\b(SELECT|ASK|CONSTRUCT|DESCRIBE)\b", t); r["form"] = m.group(1) if m else "?"
    try:
        alg = translateQuery(parseQuery(add_prefixes(q))).algebra
        r.update(parsed=True, arity=len(alg.get("PV") or []), algebra=h(var_rename(repr(alg))), algebra_colfree=h(var_rename(repr(alg), sort_pv=True)),
                 template=h(var_rename(repr(traverse(alg, visitPost=templ)))), n_tp=sum(len(b.triples) for b in bgps(alg, [])))
    except Exception as e:
        r.update(parsed=False, err=type(e).__name__, arity=None, algebra="ws:" + r["ws"], algebra_colfree="ws:" + r["ws"],
                 template=h(re.sub(r"\?\w+", "?v", re.sub(r"<[^>]*>|\b\w+:\w[\w\-]*|\"[^\"]*\"|\b\d+\b", "C", ws(q)))))
    return r

def main(d, g, out):
    splits = {s: json.load(open(f"{d}/{g}-{s}.json")) for s in ("train", "valid", "test")}
    q_split, edges, negp = {}, [], []
    for s, recs in splits.items():
        for r in recs:
            a = r["anchor_query"]
            for c in [a] + [x["query"] for x in r["positive_queries"] + r["negative_queries"]]:
                if q_split.setdefault(ws(c), s) != s: q_split[ws(c)] = "CONFLICT"
            edges += [(ws(a), ws(x["query"]), x["answer_jaccard"]) for x in r["positive_queries"]]
            negp += [(ws(a), ws(x["query"]), x["answer_jaccard"]) for x in r["negative_queries"]]
    qs = list(q_split)
    with Pool(4) as p: info = dict(zip(qs, p.map(analyze, qs, chunksize=50)))
    res = {"graph": g, "unique_queries": len(qs), "ws_identity_conflicts": sum(v == "CONFLICT" for v in q_split.values())}
    by = defaultdict(lambda: defaultdict(set))
    for q, s in q_split.items():
        for k in ("ws", "algebra", "algebra_colfree", "template"): by[k][s].add(info[q][k])
    for k in ("ws", "algebra", "algebra_colfree", "template"):
        S = by[k]; res[f"cross_split_{k}"] = {"train_valid": len(S["train"] & S["valid"]),
            "train_test": len(S["train"] & S["test"]), "valid_test": len(S["valid"] & S["test"])}
    ta = [ws(r["anchor_query"]) for r in splits["test"]]
    res["test_anchors"] = len(ta)
    res["test_anchors_with_train_template"] = sum(info[q]["template"] in by["template"]["train"] for q in ta)
    res["cross_split_positive_edges"] = sum(q_split[a] != q_split[b] for a, b, _ in edges)
    G = nx.Graph(); G.add_nodes_from(qs); G.add_edges_from((a, b) for a, b, _ in edges)
    comps = sorted((len(c) for c in nx.connected_components(G)), reverse=True); big = [c for c in comps if c > 1]
    res.update(components=len(comps), components_size_ge2=len(big), largest_component=comps[0],
               median_component_size_ge2=st.median(big),
               test_components=nx.number_connected_components(G.subgraph([q for q, s in q_split.items() if s == "test"])))
    pj = [j for *_, j in edges]; und = {tuple(sorted((a, b))): j for a, b, j in edges}
    res.update(positive_relations=len(pj), positive_J1_share=sum(j >= 1 - 1e-12 for j in pj) / len(pj),
               positive_unordered_pairs=len(und), negative_relations=len(negp),
               negative_nonzero_share=sum(j > 0 for *_, j in negp) / len(negp))
    def same_arity(pairs):
        v = [(info[a]["arity"], info[b]["arity"]) for a, b, _ in pairs if info[a]["arity"] is not None and info[b]["arity"] is not None]
        return sum(x == y for x, y in v) / len(v)
    res["negatives_same_arity_share"] = same_arity(negp); res["positives_same_arity_share"] = same_arity(edges)
    res["features_count"] = {k: sum(bool(info[q][k]) for q in qs) for k in ("aggregate", "limit", "offset", "order", "distinct", "service")}
    res["slicing_count"] = sum(info[q]["limit"] or info[q]["offset"] for q in qs)
    res["forms"] = dict(Counter(info[q]["form"] for q in qs)); res["parse_failures"] = sum(not info[q]["parsed"] for q in qs)
    res["arity_distribution"] = dict(sorted(Counter(info[q]["arity"] for q in qs if info[q]["arity"] is not None).items()))
    tp = [info[q]["n_tp"] for q in qs if info[q]["parsed"]]; res["triple_patterns_median_mean_max"] = [st.median(tp), round(st.mean(tp), 2), max(tp)]
    cf = defaultdict(set)
    for q in qs: cf[info[q]["algebra_colfree"]].add(info[q]["algebra"])
    var_groups = [k for k, v in cf.items() if len(v) > 1]
    res["column_order_variant_groups"] = len(var_groups)
    res["queries_in_column_order_variant_groups"] = sum(info[q]["algebra_colfree"] in set(var_groups) for q in qs)
    res["distinct_templates"] = len({info[q]["template"] for q in qs})
    json.dump(res, open(out, "w"), indent=2); print(json.dumps(res, indent=2))
    json.dump(info, open(out.replace(".json", "_perquery.json"), "w"))

if __name__ == "__main__": main(*sys.argv[1:4])
