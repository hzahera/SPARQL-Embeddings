import torch
import numpy as np
from torch_geometric.data import Batch, Data
from transformers import AutoTokenizer
from .config import Config
from .model import HybridSPARQLEncoder
from .graph_builder import SPARQLGraphBuilder

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
        # 1. Extract Node Labels
        node_labels = [G.nodes[n]['label'] for n in G.nodes]
        if not node_labels: node_labels = ["empty"]
        
        # 2. Extract Edge Attributes
        edge_indices = []
        edge_labels = []
        if G.number_of_edges() > 0:
            for u, v, d in G.edges(data=True):
                edge_indices.append([u, v])
                edge_labels.append(d.get('label', 'unknown'))
            edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        else:
            edge_index = torch.tensor([[0], [0]], dtype=torch.long)
            edge_labels = ["self"]

        # 3. Tokenize
        n_tok = self.tokenizer(node_labels, padding='max_length', max_length=16, truncation=True, return_tensors='pt')
        e_tok = self.tokenizer(edge_labels, padding='max_length', max_length=16, truncation=True, return_tensors='pt')

        return Data(
            x_ids=n_tok['input_ids'], 
            x_mask=n_tok['attention_mask'],
            edge_index=edge_index,
            edge_attr_ids=e_tok['input_ids'], 
            edge_attr_mask=e_tok['attention_mask']
        )

    def encode(self, queries):
        if isinstance(queries, str): 
            queries = [queries]
            
        data_list = []
        for q in queries:
            g = self.builder.build_graph(q)
            data_list.append(self._graph_to_data(g))
            
        batch = Batch.from_data_list(data_list).to(self.device)
        
        with torch.no_grad():
            embeddings = self.model(batch)
            
        return embeddings.cpu().numpy()
