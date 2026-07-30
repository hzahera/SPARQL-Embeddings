"""embedder.py
SPARQLEmbedder: high-level inference wrapper that loads a trained
HybridSPARQLEncoder checkpoint and encodes SPARQL queries to numpy vectors.
"""
import torch
import numpy as np
from torch_geometric.data import Batch, Data
from transformers import AutoTokenizer
from .config import Config
from .encoder import HybridSPARQLEncoder
from .graph_builder import SPARQLGraphBuilder, WholeQueryGraphBuilder

# Maps the 3-way ablation tag used by scripts/train_sparqlce.py to the
# (encoder mode, graph builder) pair that reproduces it at inference time.
# "no_decomposition" uses the *full* encoder architecture (GNN included) but
# over a single-node whole-query graph -- see WholeQueryGraphBuilder.
_ABLATION_TO_ENCODER_MODE = {"full": "full", "no_gnn": "no_gnn", "no_decomposition": "full"}
_ABLATION_TO_BUILDER = {
	"full": SPARQLGraphBuilder, "no_gnn": SPARQLGraphBuilder,
	"no_decomposition": WholeQueryGraphBuilder,
}

class SPARQLEmbedder:
	def __init__(self, checkpoint_path, ablation=None):
		self.device = Config.DEVICE

		# Load the trained weights
		print(f"Loading model from {checkpoint_path}...")
		checkpoint = torch.load(checkpoint_path, map_location=self.device)
		# Accept both a raw state_dict (e.g. sparqlce/models/final_model_*.pt)
		# and the wrapped {"model_state_dict": ..., "ablation": ...} dict that
		# training checkpoints (best_model.pt) save.
		output_dim = None
		if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
			state_dict = checkpoint["model_state_dict"]
			ablation = ablation or checkpoint.get("ablation", "full")
			output_dim = checkpoint.get("output_dim")
		else:
			state_dict = checkpoint
			ablation = ablation or "full"

		# The embedding width is baked into the weights. Prefer the value the
		# checkpoint was trained with; otherwise infer it from the weights, so
		# a checkpoint trained at a different Config.OUTPUT_DIM still loads
		# instead of raising an opaque size mismatch.
		if output_dim is None:
			ln_w = state_dict.get("ln.weight")
			if ln_w is not None:
				output_dim = int(ln_w.shape[0])
		if output_dim is not None and output_dim != Config.OUTPUT_DIM:
			print(f"  note: checkpoint output_dim={output_dim} differs from "
			      f"Config.OUTPUT_DIM={Config.OUTPUT_DIM}; using the checkpoint's.")

		self.model = HybridSPARQLEncoder(
			ablation=_ABLATION_TO_ENCODER_MODE[ablation], output_dim=output_dim
		).to(self.device)
		self.model.load_state_dict(state_dict)
		self.model.eval()
        
		# Graph/text construction must match what this checkpoint was TRAINED
		# with, not what the current default is. Both settings change the input
		# the weights see, and neither changes any tensor shape -- so a mismatch
		# does not raise, it just quietly scores a different model. Checkpoints
		# written before these options existed carry neither key and correctly
		# fall back to the original behaviour.
		cfg = checkpoint.get("graph_config", {}) if isinstance(checkpoint, dict) else {}
		self.query_stream = bool(cfg.get("query_stream", False))
		self.query_max_length = int(cfg.get("query_max_length", 256))
		join_hubs = bool(cfg.get("join_hubs", False))

		builder_cls = _ABLATION_TO_BUILDER[ablation]
		self.builder = (builder_cls(join_hubs=join_hubs)
		                if builder_cls is SPARQLGraphBuilder else builder_cls())
		self.tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)
		print(f"  graph_config: query_stream={self.query_stream} "
		      f"join_hubs={join_hubs} (from checkpoint)")

	def _graph_to_data(self, G, query_text):
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

		# max_length must match sparqlce/dataset.py (nodes 32, edges 16). It read
		# 16 for nodes here, so any node label longer than 16 tokens was encoded
		# differently at inference than in training.
		n_tok = self.tokenizer(node_labels, padding='max_length', max_length=32, truncation=True, return_tensors='pt')
		e_tok = self.tokenizer(edge_labels, padding='max_length', max_length=16, truncation=True, return_tensors='pt')
		data = Data(
			x_ids=n_tok['input_ids'],
			x_mask=n_tok['attention_mask'],
			edge_index=edge_index,
			edge_attr_ids=e_tok['input_ids'],
			edge_attr_mask=e_tok['attention_mask'],
			num_nodes=num_nodes,
		)
		if self.query_stream:
			q_tok = self.tokenizer("query: " + " ".join(query_text.split()),
			                       padding='max_length', max_length=self.query_max_length,
			                       truncation=True, return_tensors='pt')
			data.q_ids = q_tok['input_ids']
			data.q_mask = q_tok['attention_mask']
		return data

	def encode(self, queries):
		if isinstance(queries, str):
			queries = [queries]

		data_list = []
		for q in queries:
			g = self.builder.build_graph(q)
			data_list.append(self._graph_to_data(g, q))

		try:
			batch = Batch.from_data_list(data_list).to(self.device)
			with torch.no_grad():
				            embeddings, _ = self.model(batch)
			out = embeddings.cpu().numpy()
		except RuntimeError:
			# forward() returns (embedding, gate_mean); indexing [0] was missing,
			# so this out-of-memory fallback raised instead of recovering.
			with torch.no_grad():
				out = np.vstack([
					self.model(Batch.from_data_list([d]).to(self.device))[0].cpu().numpy()
					for d in data_list
				])

		if out is None or (isinstance(out, np.ndarray) and out.size == 0):
			raise RuntimeError("SPARQLEmbedder.encode returned no embeddings")
		return out