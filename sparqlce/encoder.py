"""hybrid_encoder.py
HybridSPARQLEncoder: combines CodeBERT token embeddings with a GATv2
graph encoder to produce L2-normalised query embeddings.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel
from torch_geometric.nn import GATv2Conv, global_mean_pool
from .config import Config


class HybridSPARQLEncoder(nn.Module):
	"""CodeBERT + GATv2 encoder for SPARQL queries.

	Each query is parsed into a graph; node and edge labels are encoded
	by CodeBERT, then refined by two GATv2 layers. Global mean pooling
	and a linear projection yield an L2-normalised 128-d embedding.
	"""

	def __init__(self):
		super().__init__()
		# Fine-tune CodeBERT jointly with the GNN
		self.text_enc = AutoModel.from_pretrained(Config.MODEL_NAME)
		for p in self.text_enc.parameters():
			p.requires_grad = True

		self.gnn = GATv2Conv(
			Config.EMBEDDING_DIM,
			Config.HIDDEN_DIM,
			heads=Config.GNN_HEADS,
			edge_dim=Config.EMBEDDING_DIM,
		)
		self.proj = nn.Linear(Config.HIDDEN_DIM * Config.GNN_HEADS, Config.OUTPUT_DIM)

	def _txt(self, ids, mask):
		"""Return the [CLS] token embedding for a batch of token sequences."""
		return self.text_enc(ids, mask).last_hidden_state[:, 0, :]

	def forward(self, data):
		node_emb = self._txt(data.x_ids, data.x_mask)
		edge_emb = self._txt(data.edge_attr_ids, data.edge_attr_mask)
		x = self.gnn(node_emb, data.edge_index, edge_attr=edge_emb)
		x = global_mean_pool(x, data.batch)
		return F.normalize(self.proj(x), p=2, dim=1)