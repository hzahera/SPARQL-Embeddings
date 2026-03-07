import torch
import json
import networkx as nx
from torch.utils.data import Dataset
from torch_geometric.data import Data, Batch
from transformers import AutoTokenizer
from .config import Config

class SPARQLDataset(Dataset):
    def __init__(self, data_path, graph_builder):
        self.builder = graph_builder
        self.tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)
        self.samples = []
        
        print(f"📂 Loading dataset from: {data_path}")
        
        with open(data_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
            
        count = 0
        for item in raw_data:
            anchor = item.get('base_query', '')
            
            # Extract query strings from the list of dictionaries
            positives = [p['query'] for p in item.get('positive_queries', []) if 'query' in p]
            negatives = [n['query'] for n in item.get('negative_queries', []) if 'query' in n]
            
            if anchor and positives and negatives:
                for i, pos in enumerate(positives):
                    neg = negatives[i % len(negatives)]
                    self.samples.append({
                        'anchor': anchor,
                        'positive': pos,
                        'negative': neg
                    })
                    count += 1

                for i, pos in enumerate(positives):
                # treat another positive as a hard negative – same template, different resource
                    hard_neg = positives[(i + 1) % len(positives)]
                    self.samples.append({
                        "anchor": anchor,
                        "positive": pos,
                        "negative": hard_neg,
                    })
                    
        print(f"✅ Generated {len(self.samples)} training triplets.")

    def _to_data(self, G):
        # 1. Extract Node Labels
        labels = [G.nodes[n]['label'] for n in G.nodes]
        if not labels: labels = ["empty"]
        
        # Tokenize Nodes
        n_tok = self.tokenizer(labels, padding='max_length', max_length=32, truncation=True, return_tensors='pt')
        
        # Calculate Number of Nodes
        num_nodes = len(labels)
        
        # 2. Extract Edges
        edge_index = []
        edge_attr = []
        if G.number_of_edges() > 0:
            for u, v, d in G.edges(data=True):
                edge_index.append([u, v])
                edge_attr.append(d.get('label', 'unknown'))
            edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
            e_tok = self.tokenizer(edge_attr, padding='max_length', max_length=16, truncation=True, return_tensors='pt')
        else:
            edge_index = torch.tensor([[0], [0]], dtype=torch.long)
            e_tok = self.tokenizer(["self"], padding='max_length', max_length=16, truncation=True, return_tensors='pt')

        return Data(
            x_ids=n_tok['input_ids'], 
            x_mask=n_tok['attention_mask'],
            edge_index=edge_index,
            edge_attr_ids=e_tok['input_ids'], 
            edge_attr_mask=e_tok['attention_mask'],
            num_nodes=num_nodes, 
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        return (
            self._to_data(self.builder.build_graph(item['anchor'])),
            self._to_data(self.builder.build_graph(item['positive'])),
            self._to_data(self.builder.build_graph(item['negative']))
        )

def collate_fn(batch):
    anchors, poss, negs = zip(*batch)
    return Batch.from_data_list(anchors), Batch.from_data_list(poss), Batch.from_data_list(negs)
