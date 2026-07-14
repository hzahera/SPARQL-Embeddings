#!/usr/bin/env python3
"""Parameterized SparqlCE training script.

Consolidates the logic of notebooks/training/{train_dbpedia,train_wikidata}.ipynb
into a scriptable, seed- and ablation-parameterized CLI so multi-seed and
ablation sweeps can be launched in a loop instead of by hand-editing notebook
cells. Same recipe as the notebooks: InfoNCE(in-batch) + lambda * MSE(cos, jaccard),
AdamW with differential LRs, best-val-MRR checkpoint selection, paper's fixed
split when available.

Usage:
    python scripts/train_sparqlce.py --kg dbpedia --seed 42
    python scripts/train_sparqlce.py --kg wikidata --seed 43 --ablation no_gnn
    python scripts/train_sparqlce.py --kg dbpedia --seed 44 \
        --ablation no_decomposition --epochs 20

Writes:
    {ckpt_dir}/best_model.pt   -- best checkpoint by val MRR
    {ckpt_dir}/run_metadata.json -- seed, ablation, config, final val/test metrics
"""
import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, random_split
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from sparqlce.graph_builder import SPARQLGraphBuilder, WholeQueryGraphBuilder
from sparqlce.encoder import HybridSPARQLEncoder
from sparqlce.dataset import SemanticSPARQLDataset, semantic_collate_fn_inbatch

KG_PATHS = {
    "dbpedia": {
        "data": ROOT_DIR / "datasets" / "dbpedia-triplets-dataset.json",
        "splits": ROOT_DIR / "datasets" / "splits_dbpedia.json",
    },
    "wikidata": {
        "data": ROOT_DIR / "datasets" / "wikidata-triplets-dataset.json",
        "splits": ROOT_DIR / "datasets" / "splits_wikidata.json",
    },
}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def infonce_inbatch(a_emb, p_emb, temperature):
    a = F.normalize(a_emb, dim=1)
    p = F.normalize(p_emb, dim=1)
    logits = a @ p.T / temperature
    labels = torch.arange(a.size(0), device=a.device)
    return F.cross_entropy(logits, labels)


@torch.no_grad()
def evaluate_ranking(model, loader, device):
    model.eval()
    anchors, positives = [], []
    for batch in loader:
        a_emb = F.normalize(model(batch[0].to(device)), dim=1)
        p_emb = F.normalize(model(batch[1].to(device)), dim=1)
        anchors.append(a_emb.cpu())
        positives.append(p_emb.cpu())
    A = torch.cat(anchors)
    P = torch.cat(positives)
    sims = A @ P.T
    diag = sims.diag().unsqueeze(1)
    ranks = (sims >= diag).sum(dim=1).float()
    return {
        "mrr": (1.0 / ranks).mean().item(),
        "hits@1": (ranks == 1).float().mean().item(),
        "hits@5": (ranks <= 5).float().mean().item(),
    }


def build_loaders(data_path, splits_path, seed, batch_size, graph_builder):
    dataset = SemanticSPARQLDataset(
        str(data_path), graph_builder=graph_builder, in_batch_negatives=True,
    )
    if splits_path.exists():
        with open(splits_path) as f:
            splits = json.load(f)
        train_idx = splits.get("train_idx", splits.get("train"))
        val_idx = splits.get("val_idx", splits.get("valid_idx", splits.get("val")))
        test_idx = splits.get("test_idx", splits.get("test"))
        assert train_idx and val_idx and test_idx, f"could not parse splits from {splits_path}"
        train_ds, val_ds, test_ds = Subset(dataset, train_idx), Subset(dataset, val_idx), Subset(dataset, test_idx)
        print(f"Using fixed split from {splits_path}")
    else:
        print(f"WARNING: fixed split not found at {splits_path}; using seeded random split "
              f"(numbers will not be directly comparable to the paper's fixed-split runs)")
        total = len(dataset)
        n_train, n_val = int(0.80 * total), int(0.10 * total)
        n_test = total - n_train - n_val
        gen = torch.Generator().manual_seed(seed)
        train_ds, val_ds, test_ds = random_split(dataset, [n_train, n_val, n_test], generator=gen)

    mk = lambda ds, shuffle: DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                                         collate_fn=semantic_collate_fn_inbatch)
    return mk(train_ds, True), mk(val_ds, False), mk(test_ds, False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kg", required=True, choices=["dbpedia", "wikidata"])
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--ablation", default="full",
                     choices=["full", "no_gnn", "no_decomposition"],
                     help="full: paper model. no_gnn: skip GATv2 message passing "
                          "(bag of node embeddings). no_decomposition: whole query "
                          "as one node, no graph structure at all.")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--temperature", type=float, default=0.05)
    ap.add_argument("--lambda-reg", type=float, default=0.3)
    ap.add_argument("--weight-decay", type=float, default=1e-2)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--lr-text", type=float, default=1e-5)
    ap.add_argument("--lr-rest", type=float, default=1e-4)
    ap.add_argument("--single-lr", type=float, default=None,
                     help="if set, use one LR for all params instead of differential LRs")
    ap.add_argument("--ckpt-root", default=str(ROOT_DIR / "checkpoints"))
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_dir = Path(args.ckpt_root) / args.kg / f"seed{args.seed}_{args.ablation}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # no_decomposition swaps the graph builder (whole-query single node);
    # no_gnn keeps the real graph but the encoder skips message passing.
    graph_builder = WholeQueryGraphBuilder() if args.ablation == "no_decomposition" else SPARQLGraphBuilder()
    encoder_mode = "no_gnn" if args.ablation == "no_gnn" else "full"

    paths = KG_PATHS[args.kg]
    train_loader, val_loader, test_loader = build_loaders(
        paths["data"], paths["splits"], args.seed, args.batch_size, graph_builder
    )
    print(f"train={len(train_loader.dataset)} val={len(val_loader.dataset)} test={len(test_loader.dataset)}")

    model = HybridSPARQLEncoder(ablation=encoder_mode).to(device)

    if args.single_lr is not None:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.single_lr, weight_decay=args.weight_decay)
    else:
        text_kw = ("text", "txt", "codebert", "bert")
        text_params = [p for n, p in model.named_parameters() if any(k in n.lower() for k in text_kw) and p.requires_grad]
        rest_params = [p for n, p in model.named_parameters() if not any(k in n.lower() for k in text_kw) and p.requires_grad]
        optimizer = torch.optim.AdamW([
            {"params": text_params, "lr": args.lr_text},
            {"params": rest_params, "lr": args.lr_rest},
        ], weight_decay=args.weight_decay)

    best_mrr = -1.0
    print(f"Training {args.kg}/{args.ablation} seed={args.seed} on {device} for {args.epochs} epochs")
    for epoch in range(1, args.epochs + 1):
        model.train()
        sum_loss = sum_ce = sum_reg = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch:02d}/{args.epochs}")
        for batch in pbar:
            a_graphs, p_graphs, pos_js = batch[0].to(device), batch[1].to(device), batch[2].to(device)
            optimizer.zero_grad()
            a_emb, p_emb = model(a_graphs), model(p_graphs)
            loss_ce = infonce_inbatch(a_emb, p_emb, args.temperature)
            a_n, p_n = F.normalize(a_emb, dim=1), F.normalize(p_emb, dim=1)
            cos = (a_n * p_n).sum(dim=1)
            loss_reg = (cos - pos_js).pow(2).mean()
            loss = loss_ce + args.lambda_reg * loss_reg
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
            optimizer.step()
            sum_loss += loss.item(); sum_ce += loss_ce.item(); sum_reg += loss_reg.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}", ce=f"{loss_ce.item():.4f}", reg=f"{loss_reg.item():.4f}")

        nb = max(1, len(train_loader))
        val = evaluate_ranking(model, val_loader, device)
        print(f"Epoch {epoch:02d} | train_loss={sum_loss/nb:.4f} (ce={sum_ce/nb:.4f} reg={sum_reg/nb:.4f}) "
              f"| val MRR={val['mrr']:.4f} Hits@1={val['hits@1']:.4f} Hits@5={val['hits@5']:.4f}")

        if val["mrr"] > best_mrr:
            best_mrr = val["mrr"]
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict(),
                        "val_metrics": val, "ablation": args.ablation}, ckpt_dir / "best_model.pt")
            print(f"  best saved (val MRR={best_mrr:.4f})")

    best = torch.load(ckpt_dir / "best_model.pt", map_location=device)
    model.load_state_dict(best["model_state_dict"])
    test = evaluate_ranking(model, test_loader, device)
    print(f"TEST MRR={test['mrr']:.4f} Hits@1={test['hits@1']:.4f} Hits@5={test['hits@5']:.4f}")

    with open(ckpt_dir / "run_metadata.json", "w") as f:
        json.dump({
            "kg": args.kg, "seed": args.seed, "ablation": args.ablation,
            "epochs": args.epochs, "batch_size": args.batch_size,
            "val_metrics": best["val_metrics"], "test_metrics": test,
        }, f, indent=2)
    print(f"Saved checkpoint + metadata to {ckpt_dir}")


if __name__ == "__main__":
    main()
