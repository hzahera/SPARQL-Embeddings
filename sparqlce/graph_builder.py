"""sparql_graph_builder.py
Converts a SPARQL query string into a NetworkX graph whose nodes and
edges are labelled with human-readable tokens for downstream GATv2 encoding.
"""
import json
import os
import re
import networkx as nx

from .sparql_utils import canonicalize_variables, is_noop_filter_text

# Wikidata id -> English label map, built by scripts/build_wikidata_labels.py.
# Loaded once at import time; empty (no-op) if the file is absent, so DBpedia
# and un-resolved corpora behave exactly as before.
_LABELS_PATH = os.path.join(
    os.path.dirname(__file__), "..", "datasets", "wikidata_labels.json"
)
try:
    with open(_LABELS_PATH, encoding="utf-8") as _f:
        WIKIDATA_LABELS = json.load(_f)
except (OSError, ValueError):
    WIKIDATA_LABELS = {}

# Matches an optionally-prefixed / full-URI-reduced Wikidata id, e.g.
# "wdt:P31", "wde:Q5", or a bare "P31" left after _clean strips the URI path.
_WD_ID_RE = re.compile(r"^(?:[A-Za-z]+:)?([PQ]\d+)$")


class SPARQLGraphBuilder:
    def __init__(self):
        local_name = r'(?:[\w(),\-]|\.(?!\s))+'
        qname = r'[a-zA-Z0-9_]*:' + local_name
        path_token = r'(?:<[^>]+>|' + qname + r'|[/|*+?^()])+'
        self.triple_pattern = re.compile(
            r'(\?[a-zA-Z0-9_]+|<[^>]+>|"[^"]*"|' + qname + r')\s+'
            r'(\?[a-zA-Z0-9_]+|' + path_token + r')\s+'
            r'(\?[a-zA-Z0-9_]+|<[^>]+>|"[^"]*"|' + qname + r')'
        )
        self.filter_pattern = re.compile(r'FILTER\s*\((.*?)\)', re.IGNORECASE | re.DOTALL)
        self.optional_pattern = re.compile(r'OPTIONAL\s*\{(.*?)\}', re.IGNORECASE | re.DOTALL)
        self.union_pattern = re.compile(r'UNION', re.IGNORECASE)
        self.values_pattern = re.compile(
            r'VALUES\s+(\?[a-zA-Z_][a-zA-Z0-9_]*)\s*\{(.*?)\}',
            re.IGNORECASE | re.DOTALL,
        )
        self.bind_pattern = re.compile(
            r'BIND\s*\((.*?)\s+AS\s+(\?[a-zA-Z_][a-zA-Z0-9_]*)\s*\)',
            re.IGNORECASE | re.DOTALL,
        )
        self.var_pattern = re.compile(r'\?[a-zA-Z_][a-zA-Z0-9_]*')
        self.value_token_pattern = re.compile(
            r'<[^>]+>|"[^"]*"|\?[a-zA-Z_][a-zA-Z0-9_]*|' + qname
        )

    def _extract_where_content(self, query):
        where_match = re.search(r'WHERE\s*\{', query, re.IGNORECASE)
        if not where_match:
            return query
        start = where_match.end()
        depth = 1
        i = start
        while i < len(query) and depth > 0:
            if query[i] == '{':
                depth += 1
            elif query[i] == '}':
                depth -= 1
            i += 1
        return query[start:i - 1]

    def _clean(self, term):
        term = term.strip('<>"')
        if "#" in term:
            term = term.rsplit("#", 1)[-1]
        if "/" in term:
            term = term.rsplit("/", 1)[-1]
        # Expand opaque Wikidata ids to English labels so the E5 text stream
        # sees prose (e.g. "P31" -> "instance of (P31)"). Keeping the id in
        # parentheses preserves an exact-match anchor alongside the semantics.
        # No-op on DBpedia terms (they never match _WD_ID_RE) and when the
        # label map is empty.
        if WIKIDATA_LABELS:
            m = _WD_ID_RE.match(term)
            if m:
                label = WIKIDATA_LABELS.get(m.group(1))
                if label:  # skip ids with no / empty English label
                    return f"{label} ({m.group(1)})"
        return term

    def _node_label(self, term):
        term = term.strip()
        if term.startswith("?"):
            return "VAR:" + self._clean(term)
        if term.startswith('"'):
            return "LIT:" + self._clean(term)
        if term.upper().startswith("FILTER:"):
            return term
        if term in {"OPTIONAL", "UNION", "VALUES", "BIND"}:
            return term
        if any(ch in term for ch in "/|*+?^()") and not term.startswith("?"):
            return "PATH:" + self._clean(term)
        return "ENT:" + self._clean(term)

    def _ensure_node(self, G, node_map, label):
        if label not in node_map:
            node_map[label] = len(G)
            G.add_node(node_map[label], label=label)
        return node_map[label]

    def build_graph(self, query):
        self.last_used_fallback = False
        G = nx.DiGraph()
        query = " ".join(query.split())
        query = canonicalize_variables(query)
        content = self._extract_where_content(query)
        matches = self.triple_pattern.findall(content)
        node_map = {}

        if not matches:
            self.last_used_fallback = True
            words = query.split()
            for i, word in enumerate(words[:10]):
                tok = "TOK:" + word
                G.add_node(i, label=tok)
                if i > 0:
                    G.add_edge(i - 1, i, label="next")
            return G

        var_occurs = {}

        for s, p, o in matches:
            s_lbl = self._node_label(s)
            o_lbl = self._node_label(o)
            s_id = self._ensure_node(G, node_map, s_lbl)
            o_id = self._ensure_node(G, node_map, o_lbl)

            if s_lbl.startswith("VAR:"):
                var_occurs.setdefault(s_id, set()).add(o_id)
            if o_lbl.startswith("VAR:"):
                var_occurs.setdefault(o_id, set()).add(s_id)

            p_clean = self._clean(p)
            if any(ch in p for ch in "/|*+?^()") and not p.startswith("?"):
                p_lbl = "PATH:" + p_clean
                p_id = self._ensure_node(G, node_map, p_lbl)
                G.add_edge(s_id, p_id, label="path_in")
                G.add_edge(p_id, o_id, label="path_out")
            else:
                edge_label = "PRED:" + p_clean
                G.add_edge(s_id, o_id, label=edge_label)

        for var_id, neighbors in var_occurs.items():
            neighbors = list(neighbors)
            for i in range(len(neighbors)):
                for j in range(i + 1, len(neighbors)):
                    G.add_edge(neighbors[i], neighbors[j], label="join")
                    G.add_edge(neighbors[j], neighbors[i], label="join")

        for f_match in self.filter_pattern.finditer(content):
            f_text = f_match.group(1).strip()
            if not f_text or is_noop_filter_text(f_text):
                continue
            clipped = f_text[:80] + "..." if len(f_text) > 80 else f_text
            f_label = "FILTER:" + clipped
            f_id = self._ensure_node(G, node_map, f_label)
            vars_in_filter = set(self.var_pattern.findall(f_text))
            for var in vars_in_filter:
                v_lbl = self._node_label(var)
                v_id = self._ensure_node(G, node_map, v_lbl)
                G.add_edge(v_id, f_id, label="uses_filter")

        for o_match in self.optional_pattern.finditer(content):
            o_text = o_match.group(1).strip()
            if not o_text:
                continue
            opt_id = self._ensure_node(G, node_map, "OPTIONAL")
            vars_in_opt = set(self.var_pattern.findall(o_text))
            for var in vars_in_opt:
                v_lbl = self._node_label(var)
                v_id = self._ensure_node(G, node_map, v_lbl)
                G.add_edge(opt_id, v_id, label="optional_scope")

        if self.union_pattern.search(content):
            union_id = self._ensure_node(G, node_map, "UNION")
            for var in set(self.var_pattern.findall(content)):
                v_lbl = self._node_label(var)
                v_id = self._ensure_node(G, node_map, v_lbl)
                G.add_edge(union_id, v_id, label="union_scope")

        for v_match in self.values_pattern.finditer(content):
            var, vals_text = v_match.groups()
            values_id = self._ensure_node(G, node_map, "VALUES")
            v_lbl = self._node_label(var)
            v_id = self._ensure_node(G, node_map, v_lbl)
            G.add_edge(values_id, v_id, label="binds")
            for raw in self.value_token_pattern.findall(vals_text)[:10]:
                val_lbl = self._node_label(raw)
                val_id = self._ensure_node(G, node_map, val_lbl)
                G.add_edge(values_id, val_id, label="value")

        for b_match in self.bind_pattern.finditer(content):
            expr, var = b_match.groups()
            bind_id = self._ensure_node(G, node_map, "BIND")
            v_lbl = self._node_label(var)
            v_id = self._ensure_node(G, node_map, v_lbl)
            G.add_edge(bind_id, v_id, label="defines")
            for used_var in set(self.var_pattern.findall(expr)):
                u_lbl = self._node_label(used_var)
                u_id = self._ensure_node(G, node_map, u_lbl)
                G.add_edge(u_id, bind_id, label="used_in_bind")

        return G


class WholeQueryGraphBuilder:
    """Ablation baseline: represents every query as a single node holding
    the full (canonicalized) query string, with no triple/FILTER/OPTIONAL
    decomposition at all.

    Used to isolate whether decomposing a query into a graph contributes
    anything beyond encoding it as opaque text with the same backbone and
    training recipe as the full model (as opposed to comparing against the
    external baselines, which differ in both encoder and training).
    """

    def build_graph(self, query):
        query = " ".join(query.split())
        query = canonicalize_variables(query)
        G = nx.DiGraph()
        G.add_node(0, label="QUERY:" + query)
        return G