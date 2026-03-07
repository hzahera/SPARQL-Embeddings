"""embedder.py
SPARQLEmbedder: high-level inference wrapper that loads a trained
HybridSPARQLEncoder checkpoint and encodes SPARQL queries to numpy vectors.
"""
import torch
import numpy as np
from torch_geometric.data import Batch, Data
from transformers import AutoTokenizer
from .config import Config
from .hybrid_encoder import HybridSPARQLEncoder
from .sparql_graph_builder import SPARQLGraphBuilder

class SPARQLEmbedder:
    def __init__(self, checkpoint_path):
        self.device = Config.DEVICE
        self.model = HybridSPARQLEncoder().to(self.device)
        
        # Load the trained weights
        print(f"Loading model from {checkpoint_path}...")
        self.model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
        self.model.eval()
        
        self.builder = SPARQLGraphBuilder()
        self.tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)

    def _graph_to_data(self, G):
        nodes = sorted(G.nodes())
        node_labels = [G.nodes[n]['label'] for n in nodes]
        if not node_labels:
            node_labels = ["empty"]
        num_nodes = len(node_labels)

        old_to_new = {old: i for i, old in enumerate(nodes)}

        edge_indices = []
        edge_labels = []
        if G.number_of_edges() > 0:
            for u, v, d in G.edges(data=True):
                edge_indices.append([old_to_new[u], old_to_new[v]])
                edge_labels.append(d.get('label', 'unknown'))
            edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        else:
            edge_index = torch.tensor([[0], [0]], dtype=torch.long)
            edge_labels = ["self"]

        n_tok = self.tokenizer(node_labels, padding='max_length', max_length=16, truncation=True, return_tensors='pt')
        e_tok = self.tokenizer(edge_labels, padding='max_length', max_length=16, truncation=True, return_tensors='pt')

        return Data(
            x_ids=n_tok['input_ids'],
            x_mask=n_tok['attention_mask'],
            edge_index=edge_index,
            edge_attr_ids=e_tok['input_ids'],
            edge_attr_mask=e_tok['attention_mask'],
            num_nodes=num_nodes,
        )

    def encode(self, queries):
        if isinstance(queries, str):
            queries = [queries]

        data_list = []
        for q in queries:
            g = self.builder.build_graph(q)
            data_list.append(self._graph_to_data(g))

        try:
            batch = Batch.from_data_list(data_list).to(self.device)
            with torch.no_grad():
                embeddings = self.model(batch)
            out = embeddings.cpu().numpy()
        except RuntimeError:
            with torch.no_grad():
                out = np.vstack([
                    self.model(Batch.from_data_list([d]).to(self.device)).cpu().numpy()
                    for d in data_list
                ])

        if out is None or (isinstance(out, np.ndarray) and out.size == 0):
            raise RuntimeError("SPARQLEmbedder.encode returned no embeddings")
        return out
