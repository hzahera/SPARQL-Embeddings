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
# Global kill-switch (shared with sparql_utils): SPARQLCE_NO_WIKIDATA_LABELS=1
# forces raw opaque P/Q ids here too, so SparqlCE and the text baselines are
# disabled together -- the only fair way to run "without label expansion".
_LABELS_DISABLED = os.environ.get("SPARQLCE_NO_WIKIDATA_LABELS", "").strip().lower() \
    in ("1", "true", "yes", "on")
if _LABELS_DISABLED:
    WIKIDATA_LABELS = {}
    print("[graph_builder] WIKIDATA_LABELS: DISABLED via SPARQLCE_NO_WIKIDATA_LABELS "
          "-- Wikidata P/Q ids kept raw (no expansion) for ALL models")
else:
    try:
        with open(_LABELS_PATH, encoding="utf-8") as _f:
            WIKIDATA_LABELS = json.load(_f)
        print(f"[graph_builder] WIKIDATA_LABELS: {len(WIKIDATA_LABELS)} ids loaded from {_LABELS_PATH}")
    except (OSError, ValueError):
        WIKIDATA_LABELS = {}
        print(f"[graph_builder] WIKIDATA_LABELS: not found at {_LABELS_PATH} -- "
              f"Wikidata P/Q ids will NOT be expanded to English labels "
              f"(run scripts/build_wikidata_labels.py first if training/evaluating on Wikidata)")

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
        # Constants (IRIs, qnames, literals) usable as BIND operands.
        self.const_token_pattern = re.compile(r'<[^>]+>|"[^"]*"|' + qname)
        # Splits a property path into IRI/qname atoms and path operators, so
        # each atom can be cleaned (and label-expanded) independently.
        self.path_atom_pattern = re.compile(r'<[^>]+>|' + qname)
        # Result-affecting constructs the DBpedia-era builder never saw.
        self.select_clause_pattern = re.compile(
            r'SELECT\s+(?:DISTINCT\s+|REDUCED\s+)?(.*?)(?=\bWHERE\b|\{)',
            re.IGNORECASE | re.DOTALL,
        )
        self.aggregate_pattern = re.compile(
            r'\b(COUNT|SUM|AVG|MIN|MAX|GROUP_CONCAT|SAMPLE)\s*\(\s*(DISTINCT\s+)?'
            r'(\*|\?[a-zA-Z_][a-zA-Z0-9_]*)',
            re.IGNORECASE,
        )
        self.group_by_pattern = re.compile(
            r'GROUP\s+BY\s+(.*?)(?=\bHAVING\b|\bORDER\b|\bLIMIT\b|\bOFFSET\b|$)',
            re.IGNORECASE | re.DOTALL,
        )
        self.limit_pattern = re.compile(r'\bLIMIT\s+(\d+)', re.IGNORECASE)
        self.offset_pattern = re.compile(r'\bOFFSET\s+(\d+)', re.IGNORECASE)
        self.minus_pattern = re.compile(r'MINUS\s*\{(.*?)\}', re.IGNORECASE | re.DOTALL)
        self.not_exists_pattern = re.compile(
            r'(NOT\s+EXISTS|EXISTS)\s*\{(.*?)\}', re.IGNORECASE | re.DOTALL
        )
        self.service_pattern = re.compile(
            r'SERVICE\s+(?:SILENT\s+)?(<[^>]+>|\?[a-zA-Z_][a-zA-Z0-9_]*|' + qname + r')',
            re.IGNORECASE,
        )

    def _split_query(self, query):
        """Split a query into (header, where_body, tail).

        ``header`` is everything before the WHERE block (the projection),
        ``tail`` everything after it (GROUP BY / LIMIT / OFFSET). Both were
        previously discarded, which made a ``SELECT (COUNT(?s) AS ?c)`` query
        indistinguishable from the plain ``SELECT ?s`` over the same BGP.
        """
        where_match = re.search(r'WHERE\s*\{', query, re.IGNORECASE)
        if not where_match:
            # No explicit WHERE keyword: fall back to the first brace group.
            brace = query.find('{')
            if brace < 0:
                return query, query, ""
            start = brace + 1
            header = query[:brace]
        else:
            start = where_match.end()
            header = query[:where_match.start()]
        depth = 1
        i = start
        while i < len(query) and depth > 0:
            if query[i] == '{':
                depth += 1
            elif query[i] == '}':
                depth -= 1
            i += 1
        return header, query[start:i - 1], query[i:]

    def _extract_where_content(self, query):
        return self._split_query(query)[1]

    def _clean_path(self, term):
        """Clean each IRI/qname atom inside a property path, keeping operators.

        ``_clean`` alone reduces ``(<http://.../P279>)*`` to ``P279>)*``, which
        fails ``_WD_ID_RE`` and so silently skips Wikidata label expansion on
        every property-path query -- while the text baselines' shared
        ``expand_wikidata_labels`` does resolve the full-URI form. That asymmetry
        handed the baselines prose and SPARQLCE opaque ids on the ~28% of the
        Wikidata corpus that uses paths. Cleaning atom-wise removes it.
        """
        return self.path_atom_pattern.sub(lambda m: self._clean(m.group(0)), term)

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
            return "PATH:" + self._clean_path(term)
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
        header, content, tail = self._split_query(query)
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

            is_path = any(ch in p for ch in "/|*+?^()") and not p.startswith("?")
            p_clean = self._clean_path(p) if is_path else self._clean(p)
            if is_path:
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
            expr = expr.strip()
            # One node PER BIND expression, not a single shared "BIND" node.
            # The old shared node dropped the bound constant entirely, so
            # BIND(wd:Q5 AS ?s) and BIND(wd:Q42 AS ?s) produced byte-identical
            # graphs -- 216 Wikidata queries collapsed onto one signature.
            clipped = expr[:80] + "..." if len(expr) > 80 else expr
            bind_id = self._ensure_node(G, node_map, "BIND:" + clipped)
            v_lbl = self._node_label(var)
            v_id = self._ensure_node(G, node_map, v_lbl)
            G.add_edge(bind_id, v_id, label="defines")
            for used_var in set(self.var_pattern.findall(expr)):
                u_lbl = self._node_label(used_var)
                u_id = self._ensure_node(G, node_map, u_lbl)
                G.add_edge(u_id, bind_id, label="used_in_bind")
            # Constants bound by the expression become first-class nodes, so a
            # BIND-ed entity joins the graph exactly like a triple-pattern term.
            for raw in self.const_token_pattern.findall(expr)[:10]:
                c_lbl = self._node_label(raw)
                c_id = self._ensure_node(G, node_map, c_lbl)
                G.add_edge(c_id, bind_id, label="bind_value")

        self._add_projection(G, node_map, query, header)
        self._add_modifiers(G, node_map, tail)
        self._add_negation_scopes(G, node_map, content)

        return G

    # ── result-affecting constructs outside the plain BGP ────────────────────

    def _add_projection(self, G, node_map, query, header):
        """Encode SELECT projections and aggregates.

        The WHERE body alone cannot distinguish ``SELECT ?s`` from
        ``SELECT (COUNT(?s) AS ?c)`` over the same BGP, yet they return
        completely different results. Sub-SELECT projections are parsed too, so
        the semantic content of a subquery survives even though the wrapping
        itself stays invisible (subquery wrapping is answer-preserving and is an
        RQ2 invariance operator -- it must not become a distinguishing feature).

        DISTINCT and ORDER BY are deliberately NOT encoded: both preserve the
        answer *set*, and `distinct_toggle` is an RQ2 invariance operator.
        """
        seen_projection = False
        for m in self.select_clause_pattern.finditer(query):
            clause = m.group(1).strip()
            if not clause:
                continue
            for agg_m in self.aggregate_pattern.finditer(clause):
                fn = agg_m.group(1).upper()
                arg = agg_m.group(3)
                agg_id = self._ensure_node(G, node_map, "AGG:" + fn)
                if arg.startswith("?"):
                    a_id = self._ensure_node(G, node_map, self._node_label(arg))
                    G.add_edge(a_id, agg_id, label="aggregated_by")
            if seen_projection:
                continue        # arity node reflects the outermost projection
            seen_projection = True
            if "*" in clause and not self.aggregate_pattern.search(clause):
                self._ensure_node(G, node_map, "PROJECT:*")
                continue
            proj_vars = self.var_pattern.findall(clause)
            if not proj_vars:
                continue
            proj_id = self._ensure_node(G, node_map, f"PROJECT:{len(set(proj_vars))}")
            for var in dict.fromkeys(proj_vars):
                v_id = self._ensure_node(G, node_map, self._node_label(var))
                G.add_edge(proj_id, v_id, label="projects")

    def _add_modifiers(self, G, node_map, tail):
        """Encode GROUP BY / LIMIT / OFFSET -- all of which change the result set."""
        for g_match in self.group_by_pattern.finditer(tail):
            group_vars = self.var_pattern.findall(g_match.group(1))
            if not group_vars:
                continue
            grp_id = self._ensure_node(G, node_map, "GROUPBY")
            for var in dict.fromkeys(group_vars):
                v_id = self._ensure_node(G, node_map, self._node_label(var))
                G.add_edge(grp_id, v_id, label="groups_by")
        lim = self.limit_pattern.search(tail)
        if lim:
            self._ensure_node(G, node_map, f"MODIFIER:LIMIT={lim.group(1)}")
        off = self.offset_pattern.search(tail)
        if off:
            self._ensure_node(G, node_map, f"MODIFIER:OFFSET={off.group(1)}")

    def _add_negation_scopes(self, G, node_map, content):
        """Encode MINUS / (NOT) EXISTS / SERVICE.

        Triples inside a MINUS or NOT EXISTS block are matched by
        ``triple_pattern`` and were previously folded into the graph as ordinary
        conjunctive patterns -- i.e. an exclusion was encoded as an inclusion.
        Tagging the variables in scope at least makes the negation visible.
        """
        for m_match in self.minus_pattern.finditer(content):
            scope_vars = set(self.var_pattern.findall(m_match.group(1)))
            if not scope_vars:
                continue
            minus_id = self._ensure_node(G, node_map, "MINUS")
            for var in scope_vars:
                v_id = self._ensure_node(G, node_map, self._node_label(var))
                G.add_edge(minus_id, v_id, label="minus_scope")
        for e_match in self.not_exists_pattern.finditer(content):
            kind = "NOT_EXISTS" if e_match.group(1).upper().startswith("NOT") else "EXISTS"
            scope_vars = set(self.var_pattern.findall(e_match.group(2)))
            if not scope_vars:
                continue
            ex_id = self._ensure_node(G, node_map, kind)
            for var in scope_vars:
                v_id = self._ensure_node(G, node_map, self._node_label(var))
                G.add_edge(ex_id, v_id, label="exists_scope")
        for s_match in self.service_pattern.finditer(content):
            endpoint = s_match.group(1)
            self._ensure_node(G, node_map, "SERVICE:" + self._clean(endpoint))


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