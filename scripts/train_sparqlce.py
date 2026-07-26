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

from sparqlce.config import Config
from sparqlce.graph_builder import SPARQLGraphBuilder, WholeQueryGraphBuilder
from sparqlce import graph_builder as _graph_builder_module
from sparqlce.encoder import HybridSPARQLEncoder
from sparqlce.dataset import (
    SemanticSPARQLDataset,
    semantic_collate_fn,
    semantic_collate_fn_inbatch,
)

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


def infonce_hard(a_emb, p_emb, n_emb, neg_slices, temperature):
    """InfoNCE with in-batch negatives PLUS each anchor's curated hard negatives.

    For anchor i the candidate set is: its own positive (target), every other
    anchor's positive (in-batch negatives), and the mined hard negatives listed
    in neg_slices[i] (variable count, padded with -inf and masked out). Assumes
    exactly one positive per anchor, so p_emb rows align 1:1 with a_emb rows.
    """
    a = F.normalize(a_emb, dim=1)
    p = F.normalize(p_emb, dim=1)
    b = a.size(0)
    logits = a @ p.T / temperature                         # [B, B] (diag = pos)
    if n_emb is not None and n_emb.size(0) > 0:
        n = F.normalize(n_emb, dim=1)
        max_neg = max((e - s) for s, e in neg_slices)
        hard = a.new_full((b, max_neg), float("-inf"))     # padding -> masked
        for i, (s, e) in enumerate(neg_slices):
            if e > s:
                hard[i, : e - s] = a[i] @ n[s:e].T / temperature
        logits = torch.cat([logits, hard], dim=1)          # [B, B + max_neg]
    labels = torch.arange(b, device=a.device)
    return F.cross_entropy(logits, labels)


@torch.no_grad()
def evaluate_ranking(model, loader, device):
    model.eval()
    anchors, positives = [], []
    for batch in loader:
        a_emb, _ = model(batch[0].to(device))
        p_emb, _ = model(batch[1].to(device))
        a_emb = F.normalize(a_emb, dim=1)
        p_emb = F.normalize(p_emb, dim=1)
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


def build_loaders(data_path, splits_path, seed, batch_size, graph_builder,
                  hard_negatives=False, max_neg=8, augment_rename_prob=0.0):
    """Build train/val/test loaders.

    Two modes:
      * in-batch (default): one positive per anchor, negatives are other batch
        items. Collated by semantic_collate_fn_inbatch.
      * hard_negatives: one positive + `max_neg` curated hard negatives per
        anchor, collated by semantic_collate_fn so the loss can consume them.

    Augmentation (random variable renaming) is applied ONLY to the training
    view; val/test use a separate, deterministic dataset so metrics are stable.
    """
    if hard_negatives:
        collate = semantic_collate_fn
        common = dict(in_batch_negatives=False, max_pos_per_anchor=1, max_neg_per_anchor=max_neg)
    else:
        collate = semantic_collate_fn_inbatch
        common = dict(in_batch_negatives=True)

    train_dataset = SemanticSPARQLDataset(
        str(data_path), graph_builder=graph_builder,
        augment_rename_prob=augment_rename_prob, augment_seed=seed, **common,
    )
    # Deterministic view for evaluation (no augmentation).
    eval_dataset = SemanticSPARQLDataset(
        str(data_path), graph_builder=graph_builder,
        augment_rename_prob=0.0, **common,
    )

    if splits_path.exists():
        with open(splits_path) as f:
            splits = json.load(f)
        train_idx = splits.get("train_idx", splits.get("train"))
        val_idx = splits.get("val_idx", splits.get("valid_idx", splits.get("val")))
        test_idx = splits.get("test_idx", splits.get("test"))
        assert train_idx and val_idx and test_idx, f"could not parse splits from {splits_path}"
        train_ds = Subset(train_dataset, train_idx)
        val_ds, test_ds = Subset(eval_dataset, val_idx), Subset(eval_dataset, test_idx)
        print(f"Using fixed split from {splits_path}")
    else:
        print(f"WARNING: fixed split not found at {splits_path}; using seeded random split "
              f"(numbers will not be directly comparable to the paper's fixed-split runs)")
        total = len(train_dataset)
        n_train, n_val = int(0.80 * total), int(0.10 * total)
        n_test = total - n_train - n_val
        gen = torch.Generator().manual_seed(seed)
        train_split, val_split, test_split = random_split(
            range(total), [n_train, n_val, n_test], generator=gen)
        train_ds = Subset(train_dataset, list(train_split))
        val_ds, test_ds = Subset(eval_dataset, list(val_split)), Subset(eval_dataset, list(test_split))

    mk = lambda ds, shuffle: DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                                         collate_fn=collate)
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
    ap.add_argument("--epochs", type=int, default=20,
                     help="max epochs (a cap); with --patience, training stops "
                          "early when val MRR plateaus, so this need not match "
                          "across runs -- the best-val checkpoint is what is saved")
    ap.add_argument("--patience", type=int, default=0,
                     help="early-stopping patience on val MRR (epochs without "
                          "improvement > --min-delta before stopping). 0 disables "
                          "early stopping. Use the SAME value across KGs/baselines "
                          "so the epoch budget is not an unfair advantage.")
    ap.add_argument("--min-delta", type=float, default=1e-4,
                     help="minimum val-MRR gain to count as an improvement")
    ap.add_argument("--amp", action="store_true",
                     help="mixed-precision training (autocast + GradScaler); large "
                          "memory saving -- use this if you hit CUDA OOM at big "
                          "batch / many hard negatives")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--temperature", type=float, default=0.05)
    ap.add_argument("--lambda-reg", type=float, default=0.3)
    ap.add_argument("--weight-decay", type=float, default=1e-2)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--lr-text", type=float, default=1e-5)
    ap.add_argument("--lr-rest", type=float, default=1e-4)
    ap.add_argument("--hard-negatives", action="store_true",
                     help="use the corpus's curated hard negatives in the InfoNCE "
                          "denominator (in addition to in-batch negatives)")
    ap.add_argument("--max-neg", type=int, default=8,
                     help="hard negatives per anchor when --hard-negatives is set")
    ap.add_argument("--augment-rename-prob", type=float, default=0.0,
                     help="probability of randomly renaming a query's variables "
                          "(training view only); teaches variable-name invariance")
    ap.add_argument("--single-lr", type=float, default=None,
                     help="if set, use one LR for all params instead of differential LRs")
    ap.add_argument("--ckpt-root", default=str(ROOT_DIR / "checkpoints"))
    ap.add_argument("--allow-unlabeled", action="store_true",
                     help="skip the Wikidata label-map check (train on raw "
                          "opaque P/Q ids intentionally, e.g. for an ablation)")
    args = ap.parse_args()

    if args.kg == "wikidata" and not args.allow_unlabeled and not _graph_builder_module.WIKIDATA_LABELS:
        raise SystemExit(
            "datasets/wikidata_labels.json is missing or empty -- training on "
            "Wikidata would silently use raw opaque P/Q ids instead of English "
            "labels (see sparqlce/graph_builder.py). Run "
            "scripts/build_wikidata_labels.py first, or pass --allow-unlabeled "
            "if this is intentional."
        )

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
        paths["data"], paths["splits"], args.seed, args.batch_size, graph_builder,
        hard_negatives=args.hard_negatives, max_neg=args.max_neg,
        augment_rename_prob=args.augment_rename_prob,
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
    best_epoch = 0
    epochs_no_improve = 0
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)
    print(f"Training {args.kg}/{args.ablation} seed={args.seed} on {device} for up to "
          f"{args.epochs} epochs (patience={args.patience or 'off'}, amp={args.amp})")
    for epoch in range(1, args.epochs + 1):
        model.train()
        sum_loss = sum_ce = sum_reg = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch:02d}/{args.epochs}")
        for batch in pbar:
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=args.amp):
                if args.hard_negatives:
                    # semantic_collate_fn: (anchor, pos, neg, pos_slices, neg_slices,
                    #                       pos_js, neg_js); one positive per anchor.
                    a_emb, _ = model(batch[0].to(device))
                    p_emb, _ = model(batch[1].to(device))
                    n_emb, _ = model(batch[2].to(device))
                    neg_slices, pos_js = batch[4], batch[5].to(device)
                    loss_ce = infonce_hard(a_emb, p_emb, n_emb, neg_slices, args.temperature)
                else:
                    a_graphs, p_graphs, pos_js = batch[0].to(device), batch[1].to(device), batch[2].to(device)
                    a_emb, _ = model(a_graphs)
                    p_emb, _ = model(p_graphs)
                    loss_ce = infonce_inbatch(a_emb, p_emb, args.temperature)
                a_n, p_n = F.normalize(a_emb, dim=1), F.normalize(p_emb, dim=1)
                cos = (a_n * p_n).sum(dim=1)
                loss_reg = (cos - pos_js).pow(2).mean()
                loss = loss_ce + args.lambda_reg * loss_reg
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)                        # unscale before clip
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            sum_loss += loss.item(); sum_ce += loss_ce.item(); sum_reg += loss_reg.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}", ce=f"{loss_ce.item():.4f}", reg=f"{loss_reg.item():.4f}")

        nb = max(1, len(train_loader))
        val = evaluate_ranking(model, val_loader, device)
        print(f"Epoch {epoch:02d} | train_loss={sum_loss/nb:.4f} (ce={sum_ce/nb:.4f} reg={sum_reg/nb:.4f}) "
              f"| val MRR={val['mrr']:.4f} Hits@1={val['hits@1']:.4f} Hits@5={val['hits@5']:.4f}")

        if val["mrr"] > best_mrr + args.min_delta:
            best_mrr = val["mrr"]
            best_epoch = epoch
            epochs_no_improve = 0
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict(),
                        "val_metrics": val, "ablation": args.ablation,
                        # Record the embedding width: the eval-time encoder must
                        # be built at the same width or load_state_dict fails.
                        "output_dim": Config.OUTPUT_DIM}, ckpt_dir / "best_model.pt")
            print(f"  best saved (val MRR={best_mrr:.4f})")
        else:
            epochs_no_improve += 1
            if args.patience and epochs_no_improve >= args.patience:
                print(f"  early stop: no val-MRR gain > {args.min_delta} for "
                      f"{args.patience} epochs (best {best_mrr:.4f} @ epoch {best_epoch})")
                break

    best = torch.load(ckpt_dir / "best_model.pt", map_location=device)
    model.load_state_dict(best["model_state_dict"])
    test = evaluate_ranking(model, test_loader, device)
    print(f"TEST MRR={test['mrr']:.4f} Hits@1={test['hits@1']:.4f} Hits@5={test['hits@5']:.4f}")

    with open(ckpt_dir / "run_metadata.json", "w") as f:
        json.dump({
            "kg": args.kg, "seed": args.seed, "ablation": args.ablation,
            "epochs": args.epochs, "batch_size": args.batch_size,
            "patience": args.patience, "best_epoch": best_epoch, "amp": args.amp,
            "temperature": args.temperature, "lambda_reg": args.lambda_reg,
            "hard_negatives": args.hard_negatives, "max_neg": args.max_neg,
            "augment_rename_prob": args.augment_rename_prob,
            "output_dim": Config.OUTPUT_DIM,
            "val_metrics": best["val_metrics"], "test_metrics": test,
        }, f, indent=2)
    print(f"Saved checkpoint + metadata to {ckpt_dir}")


if __name__ == "__main__":
    main()
