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

import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data, Batch
from transformers import AutoTokenizer

from .config import Config


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
    ):
        self.builder = graph_builder
        self.tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)
        self.max_pos = max_pos_per_anchor
        self.max_neg = max_neg_per_anchor
        self.in_batch_negatives = in_batch_negatives
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

    def _to_data(self, G):
        """Convert a NetworkX graph to a PyG Data object."""
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

        return Data(
            x_ids=n_tok["input_ids"],
            x_mask=n_tok["attention_mask"],
            edge_index=edge_index,
            edge_attr_ids=e_tok["input_ids"],
            edge_attr_mask=e_tok["attention_mask"],
            num_nodes=num_nodes,
        )

    def __len__(self):
        return len(self.groups)

    def __getitem__(self, idx):
        g = self.groups[idx]
        if self.in_batch_negatives:
            pos_q, pos_j = random.choice(g["positives"])
            return (
                self._to_data(self.builder.build_graph(g["anchor"])),
                self._to_data(self.builder.build_graph(pos_q)),
                float(pos_j),
            )

        pos_list = random.sample(g["positives"], min(self.max_pos or len(g["positives"]), len(g["positives"])))
        neg_list = random.sample(g["negatives"], min(self.max_neg or len(g["negatives"]), len(g["negatives"])))

        pos_qs, pos_js = zip(*pos_list)
        neg_qs, neg_js = zip(*neg_list)

        anchor_data   = self._to_data(self.builder.build_graph(g["anchor"]))
        pos_data_list = [self._to_data(self.builder.build_graph(q)) for q in pos_qs]
        neg_data_list = [self._to_data(self.builder.build_graph(q)) for q in neg_qs]

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
