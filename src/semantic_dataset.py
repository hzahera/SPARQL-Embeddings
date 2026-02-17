# src/semantic_dataset.py
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data, Batch
from transformers import AutoTokenizer

from .config import Config

class SemanticSPARQLDataset(Dataset):
    def __init__(self, datapath, graph_builder):
        self.builder = graph_builder
        self.tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)
        self.samples = []

        import json
        print(f"Loading semantic dataset from {datapath}...")
        with open(datapath, "r", encoding="utf-8") as f:
            raw = json.load(f)

        count = 0
        for item in raw:
            anchor = item.get("anchor_query")
            pos_list = [p["query"] for p in item.get("positive_queries", []) if "query" in p]
            neg_list = [n["query"] for n in item.get("negative_queries", []) if "query" in n]

            if not anchor or not pos_list or not neg_list:
                continue

            # simple pairing: each positive with a negative
            for i, pos in enumerate(pos_list):
                neg = neg_list[i % len(neg_list)]
                self.samples.append({
                    "anchor": anchor,
                    "positive": pos,
                    "negative": neg,
                })
                count += 1

        print(f"Generated {len(self.samples)} semantic training triplets.")

    def to_data(self, G):
        # identical to your SPARQLDataset.to_data logic
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
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        aG = self.builder.build_graph(item["anchor"])
        pG = self.builder.build_graph(item["positive"])
        nG = self.builder.build_graph(item["negative"])
        return self.to_data(aG), self.to_data(pG), self.to_data(nG)


def semantic_collate_fn(batch):
    anchors, poss, negs = zip(*batch)
    return Batch.from_data_list(anchors), Batch.from_data_list(poss), Batch.from_data_list(negs)
