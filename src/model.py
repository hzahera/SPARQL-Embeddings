import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel
from torch_geometric.nn import GATv2Conv, global_mean_pool
from .config import Config

class HybridSPARQLEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        # Load CodeBERT
        self.text_enc = AutoModel.from_pretrained(Config.MODEL_NAME)
        
        # --- KEY CHANGE: UNFREEZE THE MODEL ---
        # We allow gradients to update CodeBERT weights
        for p in self.text_enc.parameters():
            p.requires_grad = True
            
        # GNN Layers
        self.gnn = GATv2Conv(Config.EMBEDDING_DIM, Config.HIDDEN_DIM, heads=Config.GNN_HEADS, edge_dim=Config.EMBEDDING_DIM)
        
        # Projection Head
        self.proj = nn.Linear(Config.HIDDEN_DIM * Config.GNN_HEADS, Config.OUTPUT_DIM)
        
    def _txt(self, ids, mask):
        return self.text_enc(ids, mask).last_hidden_state[:, 0, :]

    def forward(self, data):
        node_emb = self._txt(data.x_ids, data.x_mask)
        edge_emb = self._txt(data.edge_attr_ids, data.edge_attr_mask)
        
        x = self.gnn(node_emb, data.edge_index, edge_attr=edge_emb)
        x = global_mean_pool(x, data.batch)
        
        return F.normalize(self.proj(x), p=2, dim=1)