"""hybrid_encoder.py
HybridSPARQLEncoder: combines CodeBERT token embeddings with a GATv2
graph encoder to produce L2-normalised query embeddings.
"""
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel
from torch_geometric.nn import GATv2Conv, global_mean_pool
from .config import Config


class HybridSPARQLEncoder(nn.Module):
	"""CodeBERT + GATv2 encoder for SPARQL queries.

	Each query is parsed into a graph; node and edge labels are encoded
	by CodeBERT, then refined by a single GATv2 layer. Global mean pooling
	and a linear projection yield an L2-normalised 128-d embedding.

	``ablation`` selects which architectural component is active, so the
	graph encoder's contribution can be isolated for the RQ1/RQ2 ablation
	study rather than only argued for in prose:
	  - "full"    (default): CodeBERT node/edge embeddings refined by GATv2
	    message passing over the query graph, then mean-pooled.
	  - "no_gnn": CodeBERT node embeddings mean-pooled directly, skipping
	    the GATv2 layer entirely. Isolates whether message passing over
	    the graph topology adds anything beyond an unordered bag of the
	    same node/edge text embeddings.
	  - "full" combined with ``WholeQueryGraphBuilder`` (see graph_builder.py)
	    isolates whether decomposing the query into a graph at all helps,
	    versus encoding the raw query string as a single node.
	"""

	def __init__(self, ablation="full"):
		super().__init__()
		assert ablation in ("full", "no_gnn"), f"unknown ablation mode: {ablation}"
		self.ablation = ablation

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
		gnn_out_dim = Config.HIDDEN_DIM * Config.GNN_HEADS
		# no_gnn skips the GATv2 layer, so the projection reads directly
		# from the (lower-dimensional) CodeBERT node embeddings instead.
		proj_in_dim = gnn_out_dim if ablation == "full" else Config.EMBEDDING_DIM
		self.proj = nn.Linear(proj_in_dim, Config.OUTPUT_DIM)

	def _txt(self, ids, mask):
		"""Return the [CLS] token embedding for a batch of token sequences."""
		return self.text_enc(ids, mask).last_hidden_state[:, 0, :]

	def forward(self, data):
		node_emb = self._txt(data.x_ids, data.x_mask)
		if self.ablation == "no_gnn":
			x = node_emb
		else:
			edge_emb = self._txt(data.edge_attr_ids, data.edge_attr_mask)
			x = self.gnn(node_emb, data.edge_index, edge_attr=edge_emb)
		x = global_mean_pool(x, data.batch)
		return F.normalize(self.proj(x), p=2, dim=1)