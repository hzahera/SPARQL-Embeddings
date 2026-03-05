# src/semantic_dataset.py
import json
import random

import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data, Batch
from transformers import AutoTokenizer

from .config import Config


class SemanticSPARQLDataset(Dataset):
    """
    Anchor-level semantic dataset.

    Each item in `datapath` has:
      - anchor_query
      - positive_queries: [{ "query": ..., "answer_jaccard": ... }, ...]
      - negative_queries: [{ "query": ..., "answer_jaccard": ... }, ...]

    For training, we use all positives and negatives per anchor. Optional
    max_pos_per_anchor / max_neg_per_anchor cap the count per anchor to avoid
    OOM (we randomly sample that many each __getitem__).

    If in_batch_negatives=True, each __getitem__ returns only (anchor, one positive, pos_j).
    Negatives are then the other positives in the batch (no explicit neg graphs). Much faster.
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
        self.max_pos = max_pos_per_anchor  # None = use all
        self.max_neg = max_neg_per_anchor
        self.in_batch_negatives = in_batch_negatives
        self.groups = []  # one entry per anchor

        print(f"Loading semantic dataset from {datapath}...")
        with open(datapath, "r", encoding="utf-8") as f:
            raw = json.load(f)

        for item in raw:
            anchor = item.get("anchor_query")
            if not anchor:
                continue

            pos_list = []
            neg_list = []

            for p in item.get("positive_queries", []):
                q = p.get("query")
                j = p.get("answer_jaccard")
                if q is None or j is None:
                    continue
                pos_list.append((q, float(j)))

            for n in item.get("negative_queries", []):
                q = n.get("query")
                j = n.get("answer_jaccard")
                if q is None or j is None:
                    continue
                neg_list.append((q, float(j)))

            if not pos_list or not neg_list:
                continue

            self.groups.append(
                {
                    "anchor": anchor,
                    "positives": pos_list,
                    "negatives": neg_list,
                }
            )

        print(f"Generated {len(self.groups)} semantic anchor groups.")

    def _to_data(self, G):
        labels = [G.nodes[n]["label"] for n in G.nodes]
        if not labels:
            labels = ["<empty>"]
        num_nodes = len(labels)

        edge_index = []
        edge_labels = []
        if G.number_of_edges() > 0:
            for u, v, d in G.edges(data=True):
                edge_index.append((u, v))
                edge_labels.append(d.get("label", "unknown"))
        else:
            edge_index = [(0, 0)]
            edge_labels = ["self"]

        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

        n_tok = self.tokenizer(
            labels,
            padding="max_length",
            max_length=32,
            truncation=True,
            return_tensors="pt",
        )
        e_tok = self.tokenizer(
            edge_labels,
            padding="max_length",
            max_length=16,
            truncation=True,
            return_tensors="pt",
        )

        return Data(
            x_ids=n_tok["input_ids"],
            x_mask=n_tok["attention_mask"],
            edge_index=edge_index,
            edge_attr_ids=e_tok["input_ids"],
            edge_attr_mask=e_tok["attention_mask"],
            num_nodes=num_nodes,
        )

    def __len__(self):
        # one item per anchor
        return len(self.groups)

    def __getitem__(self, idx):
        g = self.groups[idx]

        if self.in_batch_negatives:
            # Fast path: one positive per anchor; negatives = other batch items
            pos_q, pos_j = random.choice(g["positives"])
            aG = self.builder.build_graph(g["anchor"])
            pG = self.builder.build_graph(pos_q)
            return self._to_data(aG), self._to_data(pG), float(pos_j)

        # use all positives/negatives, or a random sample if capped for memory
        pos_list = g["positives"]
        neg_list = g["negatives"]
        if self.max_pos is not None and len(pos_list) > self.max_pos:
            pos_list = random.sample(pos_list, self.max_pos)
        if self.max_neg is not None and len(neg_list) > self.max_neg:
            neg_list = random.sample(neg_list, self.max_neg)
        pos_qs, pos_js = zip(*pos_list)
        neg_qs, neg_js = zip(*neg_list)

        aG = self.builder.build_graph(g["anchor"])
        pos_Gs = [self.builder.build_graph(q) for q in pos_qs]
        neg_Gs = [self.builder.build_graph(q) for q in neg_qs]

        anchor_data = self._to_data(aG)
        pos_data_list = [self._to_data(Gp) for Gp in pos_Gs]
        neg_data_list = [self._to_data(Gn) for Gn in neg_Gs]

        pos_js_tensor = torch.tensor(pos_js, dtype=torch.float32)
        neg_js_tensor = torch.tensor(neg_js, dtype=torch.float32)

        return anchor_data, pos_data_list, neg_data_list, pos_js_tensor, neg_js_tensor


def semantic_collate_fn(batch):
    """
    Collate function that:
      - batches anchors as usual
      - flattens all positives and negatives into separate Batches
      - returns slices so the loss knows which positives/negs belong to which anchor
      - concatenates Jaccards into flat tensors aligned with those slices
    """
    anchors, pos_lists, neg_lists, pos_js_lists, neg_js_lists = zip(*batch)

    anchor_batch = Batch.from_data_list(anchors)

    # flatten positives
    flat_pos = []
    pos_slices = []
    flat_pos_js = []
    offset_pos = 0
    for pos_list, pos_js in zip(pos_lists, pos_js_lists):
        start = offset_pos
        flat_pos.extend(pos_list)
        flat_pos_js.append(pos_js)
        offset_pos += len(pos_list)
        pos_slices.append((start, offset_pos))

    pos_batch = Batch.from_data_list(flat_pos)
    flat_pos_js = torch.cat(flat_pos_js, dim=0)

    # flatten negatives
    flat_negs = []
    neg_slices = []
    flat_neg_js = []
    offset_neg = 0
    for neg_list, neg_js in zip(neg_lists, neg_js_lists):
        start = offset_neg
        flat_negs.extend(neg_list)
        flat_neg_js.append(neg_js)
        offset_neg += len(neg_list)
        neg_slices.append((start, offset_neg))

    neg_batch = Batch.from_data_list(flat_negs)
    flat_neg_js = torch.cat(flat_neg_js, dim=0)

    return anchor_batch, pos_batch, neg_batch, pos_slices, neg_slices, flat_pos_js, flat_neg_js


def semantic_collate_fn_inbatch(batch):
    """For in_batch_negatives=True: each item is (anchor, one_pos, pos_j)."""
    anchors, poss, pos_js = zip(*batch)
    anchor_batch = Batch.from_data_list(anchors)
    pos_batch = Batch.from_data_list(poss)
    pos_js = torch.tensor(pos_js, dtype=torch.float32)
    return anchor_batch, pos_batch, pos_js
