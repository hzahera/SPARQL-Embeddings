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

    def _clean(self, term):
        return term.strip('<>"')

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
            
        return G
