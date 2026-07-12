"""sparql_graph_builder.py
Converts a SPARQL query string into a NetworkX graph whose nodes and
edges are labelled with human-readable tokens for downstream GATv2 encoding.
"""
import re
import networkx as nx

from .sparql_utils import canonicalize_variables, is_noop_filter_text


class SPARQLGraphBuilder:
    def __init__(self):
        # supports 'prefix:suffix' pattern, including the default/empty
        # prefix (e.g. ':Bucheon' from 'PREFIX : <...>'). Local names may
        # contain non-ASCII letters (e.g. ':Kimi_no_Na_wa_Kibō') and internal
        # dots (e.g. ':Jun._K'); \w is unicode-aware, and a dot is only
        # accepted when not followed by whitespace so triple terminators
        # ('... ?o .') aren't swallowed into the preceding local name.
        local_name = r'(?:[\w(),\-]|\.(?!\s))+'
        qname = r'[a-zA-Z0-9_]*:' + local_name
        # property-path expression, e.g. 'wdt:P31/wdt:P279*' or
        # '<uri1>|(<uri2>)?': a run of IRIs/QNames glued together with the
        # path operators / | * + ? ^ ( ) and no intervening whitespace.
        # ('?' is safe here: a bare variable predicate is matched by the
        # '\?[a-zA-Z0-9_]+' alternative below before this one is tried.)
        path_token = r'(?:<[^>]+>|' + qname + r'|[/|*+?^()])+'
        self.triple_pattern = re.compile(
            r'(\?[a-zA-Z0-9_]+|<[^>]+>|"[^"]*"|' + qname + r')\s+' # Subject
            r'(\?[a-zA-Z0-9_]+|' + path_token + r')\s+'            # Predicate (incl. paths)
            r'(\?[a-zA-Z0-9_]+|<[^>]+>|"[^"]*"|' + qname + r')'   # Object
        )
        # FILTER and OPTIONAL clauses, and variables inside them
        self.filter_pattern = re.compile(r'FILTER\s*\((.*?)\)', re.IGNORECASE | re.DOTALL)
        self.optional_pattern = re.compile(r'OPTIONAL\s*\{(.*?)\}', re.IGNORECASE | re.DOTALL)
        self.var_pattern = re.compile(r'\?[a-zA-Z_][a-zA-Z0-9_]*')

    def _extract_where_content(self, query):
        """Return the text inside the outermost WHERE{...} block, matching
        braces by depth so nested VALUES{...}/OPTIONAL{...} blocks don't
        truncate the extraction at their own closing brace."""
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
        # Strip enclosing angle brackets/quotes and keep the local name
        term = term.strip('<>"')
        # Prefer fragment/local name over full URI
        if "#" in term:
            term = term.rsplit("#", 1)[-1]
        if "/" in term:
            term = term.rsplit("/", 1)[-1]
        return term

    def build_graph(self, query):
        self.last_used_fallback = False
        G = nx.DiGraph()
        query = " ".join(query.split())
        # Bake invariances directly into the graph input:
        #  - rename ?x ?y ... to ?v0 ?v1 ... (variable-name invariance)
        query = canonicalize_variables(query)

        # Robust Regex Extraction
        # 1. Find the WHERE clause content (brace-depth matched, see
        #    _extract_where_content, so nested VALUES/OPTIONAL blocks don't
        #    truncate it at their own closing brace)
        content = self._extract_where_content(query)

        # 2. Find triples
        matches = self.triple_pattern.findall(content)

        node_map = {}

        # If regex fails, we must produce SOMETHING distinct so embeddings aren't identical
        if not matches:
            # Fallback: tokenize the raw query string into a simple chain graph
            # This ensures 'SELECT ?a' is different from 'SELECT ?b' even if parsing fails
            self.last_used_fallback = True
            words = query.split()
            for i, word in enumerate(words):
                if i > 9:
                    break
                G.add_node(i, label=word)
                if i > 0:
                    G.add_edge(i - 1, i, label="next")
            return G

        for s, p, o in matches:
            s_lbl, p_lbl, o_lbl = self._clean(s), self._clean(p), self._clean(o)

            if s_lbl not in node_map:
                node_map[s_lbl] = len(G)
                G.add_node(node_map[s_lbl], label=s_lbl)

            if o_lbl not in node_map:
                node_map[o_lbl] = len(G)
                G.add_node(node_map[o_lbl], label=o_lbl)

            # Predicate is an edge attribute
            G.add_edge(node_map[s_lbl], node_map[o_lbl], label=p_lbl)

        # 3. Add FILTER nodes and connect involved variables
        for f_match in self.filter_pattern.finditer(content):
            f_text = f_match.group(1).strip()
            if not f_text:
                continue
            # Skip no-op FILTER(true) — adds no semantics, would only inject noise
            if is_noop_filter_text(f_text):
                continue
            f_label = "FILTER " + (f_text[:80] + "..." if len(f_text) > 80 else f_text)
            f_id = len(G)
            G.add_node(f_id, label=f_label)

            vars_in_filter = set(self.var_pattern.findall(f_text))
            for var in vars_in_filter:
                v_lbl = self._clean(var)
                if v_lbl not in node_map:
                    node_map[v_lbl] = len(G)
                    G.add_node(node_map[v_lbl], label=v_lbl)
                G.add_edge(node_map[v_lbl], f_id, label="filter")

        # 4. Add OPTIONAL nodes and connect to variables used in OPTIONAL block
        for o_match in self.optional_pattern.finditer(content):
            o_text = o_match.group(1).strip()
            if not o_text:
                continue
            opt_id = len(G)
            G.add_node(opt_id, label="OPTIONAL")

            vars_in_opt = set(self.var_pattern.findall(o_text))
            for var in vars_in_opt:
                v_lbl = self._clean(var)
                if v_lbl not in node_map:
                    node_map[v_lbl] = len(G)
                    G.add_node(node_map[v_lbl], label=v_lbl)
                G.add_edge(opt_id, node_map[v_lbl], label="optional")

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
        G.add_node(0, label=query)
        return G

        return G
