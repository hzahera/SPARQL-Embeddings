import re
import networkx as nx

class SPARQLGraphBuilder:
    def __init__(self):
        # supports 'prefix:suffix' pattern
        self.triple_pattern = re.compile(
            r'(\?[a-zA-Z0-9_]+|<[^>]+>|"[^"]*"|[a-zA-Z0-9_]+:[a-zA-Z0-9_\(\)\,\-]+)\s+' # Subject
            r'(\?[a-zA-Z0-9_]+|<[^>]+>|[a-zA-Z0-9_]+:[a-zA-Z0-9_\(\)\,\-]+)\s+'          # Predicate
            r'(\?[a-zA-Z0-9_]+|<[^>]+>|"[^"]*"|[a-zA-Z0-9_]+:[a-zA-Z0-9_\(\)\,\-]+)'      # Object
        )
        # FILTER and OPTIONAL clauses, and variables inside them
        self.filter_pattern = re.compile(r'FILTER\s*\((.*?)\)', re.IGNORECASE | re.DOTALL)
        self.optional_pattern = re.compile(r'OPTIONAL\s*\{(.*?)\}', re.IGNORECASE | re.DOTALL)
        self.var_pattern = re.compile(r'\?[a-zA-Z_][a-zA-Z0-9_]*')

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
        G = nx.DiGraph()
        query = " ".join(query.split())
        
        # Robust Regex Extraction
        # 1. Try to find the WHERE clause content
        where_match = re.search(r'WHERE\s*\{(.*?)\}', query, re.IGNORECASE | re.DOTALL)
        content = where_match.group(1) if where_match else query
        
        # 2. Find triples
        matches = self.triple_pattern.findall(content)
        
        node_map = {}
        
        # If regex fails, we must produce SOMETHING distinct so embeddings aren't identical
        if not matches:
             # Fallback: tokenize the raw query string into a simple chain graph
             # This ensures 'SELECT ?a' is different from 'SELECT ?b' even if parsing fails
             words = query.split()
             for i, word in enumerate(words):
                 if i > 10: break # Limit size
                 G.add_node(i, label=word)
                 if i > 0: G.add_edge(i-1, i, label="next")
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
            # Shorten very long filters to keep node labels compact
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
