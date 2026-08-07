"""triplet_dataset.py
PyTorch Dataset for anchor-level SPARQL triplets.

Expected JSON schema for each item in the dataset file:
	{
		"anchor_query": "SELECT ...",
		"positive_queries": [{"query": "...", "answer_jaccard": 0.9}, ...],
		"negative_queries": [{"query": "...", "answer_jaccard": 0.05}, ...]
	}

Classes:
	SemanticSPARQLDataset  -- loads triplets from a JSON file.

Functions:
	semantic_collate_fn          -- collate for full pos/neg mode.
	semantic_collate_fn_inbatch  -- collate for in-batch negatives mode.
"""
import json
import random
import re
import string

import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data, Batch
from transformers import AutoTokenizer

from .config import Config


_VAR_RE = re.compile(r'\?[a-zA-Z_]\w*')


def _random_rename_variables(query: str, rng: random.Random) -> str:
    """Replace each distinct SPARQL variable with a random fresh name.

    Together with the canonicalization performed inside the graph builder, this
    teaches the encoder to be invariant to *any* consistent variable renaming
    rather than just the canonical form.
    """
    distinct = []
    seen = set()
    for m in _VAR_RE.finditer(query):
        v = m.group(0)
        if v not in seen:
            seen.add(v)
            distinct.append(v)
    if not distinct:
        return query
    # Generate unique random names that don't collide with existing ones.
    new_names = []
    used = set(distinct)
    for _ in distinct:
        while True:
            length = rng.randint(3, 8)
            name = '?' + ''.join(rng.choices(string.ascii_lowercase, k=length))
            if name not in used:
                used.add(name)
                new_names.append(name)
                break
    mapping = dict(zip(distinct, new_names))
    return _VAR_RE.sub(lambda m: mapping[m.group(0)], query)


class SemanticSPARQLDataset(Dataset):
	"""Anchor-level semantic dataset.

	Each item in `datapath` has:
		- anchor_query
		- positive_queries: [{"query": ..., "answer_jaccard": ...}, ...]
		- negative_queries: [{"query": ..., "answer_jaccard": ...}, ...]

	Args:
		datapath (str): Path to JSON dataset file.
		graph_builder: Instance of SPARQLGraphBuilder.
		max_pos_per_anchor (int, optional): Cap positives per anchor (None = all).
		max_neg_per_anchor (int, optional): Cap negatives per anchor (None = all).
		in_batch_negatives (bool): If True, return only (anchor, one_pos, jaccard);
			negatives are other batch items at training time.
	"""

	def __init__(
		self,
		datapath,
		graph_builder,
		max_pos_per_anchor=None,
		max_neg_per_anchor=None,
		in_batch_negatives=False,
		augment_rename_prob=0.0,
		augment_seed=None,
		deterministic=False,
		query_stream=False,
		query_max_length=384,
		constant_stream=False,
		edge_type_fn=None,
	):
		"""
		Args (additions):
			augment_rename_prob (float): probability in [0, 1] that each query
				gets its variables randomly renamed before graph building.
				Combined with the canonicalization inside the graph builder,
				this teaches invariance to arbitrary variable names. Set to 0
				to disable (default). 0.3 is a reasonable starting point.
			augment_seed (int|None): optional RNG seed for reproducible
				augmentation. If None, uses Python's global RNG state.
			deterministic (bool): take the FIRST positive/negatives instead of
				sampling. Set for validation and test views: with random
				sampling the same checkpoint scores differently on each pass,
				so early stopping reacts to sampling noise rather than to the
				model. Never set it for training, where the resampling is the
				point.
			query_stream (bool): emit the whole query string (q_ids/q_len) so the
				encoder's text branch reads the query itself rather than a mean
				over node labels. Off by default: it changes what the model is
				given, so old runs stay reproducible.
			query_max_length (int): token budget for the whole-query text stream.
				Truncation here silently drops the tail of long queries from the
				text side while the graph still sees them, so raise it rather
				than let the two streams read different queries.
			constant_stream (bool): emit `x_is_const`, a per-node flag marking
				the ENT/LIT nodes -- the constants. `DBpediaSPARQLEncoder` pools
				exactly those into a dedicated stream, because on DBpedia the
				constant is the discriminating variable (71.9% of
				anchor-positive graph pairs are identical once constants are
				masked) and a mean over a 4-node graph dilutes it. Off by
				default: it adds a tensor old checkpoints never saw.
			edge_type_fn (callable|None): maps an edge label to an integer type
				id, emitted as `edge_type`. `HybridSPARQLEncoder` has consumed
				this field since it was written but nothing ever produced it,
				so its typed-edge embeddings never trained. Passing
				`sparqlce.dbpedia.edge_type_id` turns them on. Safe for
				existing checkpoints either way -- their `edge_type_emb` weights
				were saved as the zeros they were initialised to, so the offset
				they add is zero.
		"""
		self.builder = graph_builder
		self.query_stream = bool(query_stream)
		self.query_max_length = int(query_max_length)
		self.constant_stream = bool(constant_stream)
		self.edge_type_fn = edge_type_fn
		self.tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)
		self.max_pos = max_pos_per_anchor
		self.max_neg = max_neg_per_anchor
		self.in_batch_negatives = in_batch_negatives
		self.deterministic = bool(deterministic)
		self.augment_rename_prob = float(augment_rename_prob)
		self._aug_rng = random.Random(augment_seed) if augment_seed is not None else random
		self.groups = []

		print(f"Loading dataset from {datapath}...")
		with open(datapath, "r", encoding="utf-8") as f:
			raw = json.load(f)

		for item in raw:
			anchor = item.get("anchor_query")
			if not anchor:
				continue
			pos_list = [
				(p["query"], float(p["answer_jaccard"]))
				for p in item.get("positive_queries", [])
				if p.get("query") is not None and p.get("answer_jaccard") is not None
			]
			neg_list = [
				(n["query"], float(n["answer_jaccard"]))
				for n in item.get("negative_queries", [])
				if n.get("query") is not None and n.get("answer_jaccard") is not None
			]
			if not pos_list or not neg_list:
				continue
			self.groups.append({"anchor": anchor, "positives": pos_list, "negatives": neg_list})

		print(f"Loaded {len(self.groups)} anchor groups.")

	def _maybe_augment(self, query):
		"""Apply random variable renaming with probability ``augment_rename_prob``."""
		if self.augment_rename_prob > 0.0 and self._aug_rng.random() < self.augment_rename_prob:
			return _random_rename_variables(query, self._aug_rng)
		return query

	def _build(self, query):
		"""Optionally augment, then build the graph. Returns (graph, text).

		The text is returned alongside because the encoder's text stream reads
		the whole query string, not just the node labels -- and it must be the
		*augmented* string, so the two streams never disagree about which query
		this is.
		"""
		q = self._maybe_augment(query)
		return self.builder.build_graph(q), q

	def _to_data(self, built):
		"""Convert a (graph, query_text) pair to a PyG Data object."""
		G, query_text = built
		labels = [G.nodes[n]["label"] for n in G.nodes] or ["<empty>"]
		num_nodes = len(labels)

		edge_index, edge_labels = [], []
		if G.number_of_edges() > 0:
			for u, v, d in G.edges(data=True):
				edge_index.append((u, v))
				edge_labels.append(d.get("label", "unknown"))
		else:
			edge_index = [(0, 0)]
			edge_labels = ["self"]

		edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
		n_tok = self.tokenizer(labels, padding="max_length", max_length=32, truncation=True, return_tensors="pt")
		e_tok = self.tokenizer(edge_labels, padding="max_length", max_length=16, truncation=True, return_tensors="pt")

		data = Data(
			x_ids=n_tok["input_ids"],
			x_mask=n_tok["attention_mask"],
			edge_index=edge_index,
			edge_attr_ids=e_tok["input_ids"],
			edge_attr_mask=e_tok["attention_mask"],
			num_nodes=num_nodes,
		)

		if self.constant_stream:
			# 1-D per-node flag: PyG concatenates it along dim 0 exactly like
			# any other node attribute, so `data.batch[data.x_is_const]` gives
			# the graph id of every constant in the batch.
			from .dbpedia import is_constant_label
			data.x_is_const = torch.tensor(
				[is_constant_label(l) for l in labels], dtype=torch.bool)
		if self.edge_type_fn is not None:
			data.edge_type = torch.tensor(
				[self.edge_type_fn(l) for l in edge_labels], dtype=torch.long)

		if self.query_stream:
			# Stored UNPADDED, as a flat 1-D tensor plus its length; the encoder
			# pads each batch to its own longest query.
			#
			# Padding every query to a fixed width instead is what forced the
			# budget down to 256 tokens, and at 256 more than 5% of queries were
			# truncated -- the LONGEST ones, i.e. exactly the structurally
			# complex queries the graph branch exists for. Meanwhile the median
			# query is 73 tokens, so fixed-width padding spent ~3.5x the memory
			# it needed on a typical batch. Per-batch padding removes the
			# truncation and the waste together.
			#
			# 1-D means PyG concatenates along dim 0 (its default for any key
			# without "index" in the name), so a batch carries q_ids of total
			# length sum(q_len) and q_len of shape [B] -- no equal-width
			# requirement, and no reliance on how a PyG version collates
			# non-tensor attributes.
			q_ids = self.tokenizer(
				"query: " + query_text, truncation=True,
				max_length=self.query_max_length,
			)["input_ids"]
			data.q_ids = torch.tensor(q_ids, dtype=torch.long)
			data.q_len = torch.tensor([len(q_ids)], dtype=torch.long)
		return data

	def __len__(self):
		return len(self.groups)

	def __getitem__(self, idx):
		g = self.groups[idx]
		if self.in_batch_negatives:
			pos_q, pos_j = (g["positives"][0] if self.deterministic
			                else random.choice(g["positives"]))
			return (
				self._to_data(self._build(g["anchor"])),
				self._to_data(self._build(pos_q)),
				float(pos_j),
			)

		n_pos = min(self.max_pos or len(g["positives"]), len(g["positives"]))
		n_neg = min(self.max_neg or len(g["negatives"]), len(g["negatives"]))
		if self.deterministic:
			pos_list, neg_list = g["positives"][:n_pos], g["negatives"][:n_neg]
		else:
			pos_list = random.sample(g["positives"], n_pos)
			neg_list = random.sample(g["negatives"], n_neg)

		pos_qs, pos_js = zip(*pos_list)
		neg_qs, neg_js = zip(*neg_list)

		anchor_data   = self._to_data(self._build(g["anchor"]))
		pos_data_list = [self._to_data(self._build(q)) for q in pos_qs]
		neg_data_list = [self._to_data(self._build(q)) for q in neg_qs]

		return (
			anchor_data,
			pos_data_list,
			neg_data_list,
			torch.tensor(pos_js, dtype=torch.float32),
			torch.tensor(neg_js, dtype=torch.float32),
		)


def semantic_collate_fn(batch):
	"""Collate for full pos/neg mode.

	Flattens all positives and negatives into Batch objects and returns
	slice indices so the loss can align them with their anchors.
	"""
	anchors, pos_lists, neg_lists, pos_js_lists, neg_js_lists = zip(*batch)
	anchor_batch = Batch.from_data_list(anchors)

	flat_pos, pos_slices, flat_pos_js, offset = [], [], [], 0
	for pos_list, pos_js in zip(pos_lists, pos_js_lists):
		pos_slices.append((offset, offset + len(pos_list)))
		flat_pos.extend(pos_list)
		flat_pos_js.append(pos_js)
		offset += len(pos_list)

	flat_neg, neg_slices, flat_neg_js, offset = [], [], [], 0
	for neg_list, neg_js in zip(neg_lists, neg_js_lists):
		neg_slices.append((offset, offset + len(neg_list)))
		flat_neg.extend(neg_list)
		flat_neg_js.append(neg_js)
		offset += len(neg_list)

	return (
		anchor_batch,
		Batch.from_data_list(flat_pos),
		Batch.from_data_list(flat_neg),
		pos_slices,
		neg_slices,
		torch.cat(flat_pos_js),
		torch.cat(flat_neg_js),
	)


def semantic_collate_fn_inbatch(batch):
	"""Collate for in-batch negatives mode.

	Each item is (anchor_graph, one_pos_graph, pos_jaccard).
	"""
	anchors, poss, pos_js = zip(*batch)
	return (
		Batch.from_data_list(anchors),
		Batch.from_data_list(poss),
		torch.tensor(pos_js, dtype=torch.float32),
	)