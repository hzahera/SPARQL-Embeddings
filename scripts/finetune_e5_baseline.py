#!/usr/bin/env python3
"""Seed-parameterized E5-base-v2 contrastive fine-tuning (the E5-Finetuned baseline).

Scriptable form of
notebooks/finetuning/0{4,5}_e5_contrastive_finetune_square*.ipynb, so the
baseline can be run across multiple seeds in a loop for the significance test
instead of by hand-editing notebook cells. The recipe is byte-for-byte the
notebook's: generic in-batch-negatives InfoNCE over raw SPARQL text, full
fine-tune of a capacity-matched (~110M) backbone, E5's ``query: `` prefix,
AdamW + cosine schedule with warmup, best-val-MRR checkpoint selection.

SparqlCE-specific components (graph encoder, Jaccard calibration, rename
augmentation curriculum) remain DELIBERATELY EXCLUDED -- see BASELINE_REDESIGN.md.
This baseline must credit only a generic contrastive recipe on the same data.

On early stopping
-----------------
``--patience`` mirrors scripts/train_sparqlce.py's flag and is OFF by default,
matching the notebook. Note what it can and cannot do: both trainers already
select the **best-val-MRR checkpoint across epochs**, so a later-epoch model
that overfits is never the one saved. Early stopping therefore only truncates
the run to save compute -- it can lose a late val improvement, but it cannot
improve the selected checkpoint. It is *not* a fix for overfitting here,
because best-val selection already is one.

If you enable it, use the SAME --patience and --min-delta on BOTH sides
(SparqlCE and this baseline), or the comparison stops being like-for-like.

Usage
-----
    python scripts/finetune_e5_baseline.py --kg wikidata --seed 42
    python scripts/finetune_e5_baseline.py --kg wikidata --seed 43 --patience 3

Writes:
    {ckpt_root}/{kg}/seed{seed}/best_model/   -- HF dir, loadable by
                                                 SentenceTransformer / AutoModel
    {ckpt_root}/{kg}/seed{seed}/run_metadata.json
"""
import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset, random_split
from transformers import AutoModel, AutoTokenizer, get_cosine_schedule_with_warmup
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

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

MODEL_NAME = "intfloat/e5-base-v2"
E5_PREFIX = "query: "


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class SquareTextDataset(Dataset):
    """Anchor-level SQuARE dataset over raw SPARQL text (no graphs, no Jaccard)."""

    def __init__(self, datapath):
        self.groups = []
        with open(datapath, "r", encoding="utf-8") as f:
            raw = json.load(f)
        for item in raw:
            anchor = item.get("anchor_query")
            if not anchor:
                continue
            pos = [p["query"] for p in item.get("positive_queries", [])
                   if p.get("query") is not None]
            neg = [n["query"] for n in item.get("negative_queries", [])
                   if n.get("query") is not None]
            if not pos or not neg:
                continue
            self.groups.append({"anchor": anchor, "positives": pos, "negatives": neg})
        print(f"Loaded {len(self.groups)} anchor groups from {datapath}")

    def __len__(self):
        return len(self.groups)

    def __getitem__(self, idx):
        g = self.groups[idx]
        return g["anchor"], random.choice(g["positives"]), random.choice(g["negatives"])


class E5SparqlEmbedder(nn.Module):
    """E5-base-v2 backbone (full fine-tune) producing L2-normalised embeddings."""

    def __init__(self, model_name_or_path):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name_or_path)

    def forward(self, tok):
        out = self.backbone(input_ids=tok["input_ids"],
                            attention_mask=tok["attention_mask"])
        h = out.last_hidden_state
        m = tok["attention_mask"].unsqueeze(-1).float()
        emb = (h * m).sum(dim=1) / m.sum(dim=1).clamp(min=1e-9)
        return F.normalize(emb.float(), p=2, dim=1)


def info_nce_loss(a_emb, p_emb, n_emb, scale):
    """In-batch-negatives InfoNCE: anchor i's positive is column i of [P; N]."""
    candidates = torch.cat([p_emb, n_emb], dim=0)
    logits = scale * (a_emb @ candidates.T)
    labels = torch.arange(a_emb.size(0), device=a_emb.device)
    return F.cross_entropy(logits, labels)


@torch.no_grad()
def evaluate_ranking(model, loader, device):
    """Rank each anchor's true positive against every positive in the split."""
    model.eval()
    anchors, positives = [], []
    for a_tok, p_tok, _ in loader:
        a_tok = {k: v.to(device) for k, v in a_tok.items()}
        p_tok = {k: v.to(device) for k, v in p_tok.items()}
        anchors.append(model(a_tok).cpu())
        positives.append(model(p_tok).cpu())
    A, P = torch.cat(anchors), torch.cat(positives)
    sims = A @ P.T
    ranks = (sims >= sims.diag().unsqueeze(1)).sum(dim=1).float()
    return {"mrr": (1.0 / ranks).mean().item(),
            "hits@1": (ranks == 1).float().mean().item(),
            "hits@5": (ranks <= 5).float().mean().item()}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kg", required=True, choices=sorted(KG_PATHS))
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--epochs", type=int, default=20,
                    help="max epochs (notebook default 20)")
    ap.add_argument("--patience", type=int, default=0,
                    help="early-stopping patience on val MRR; 0 = off (notebook "
                         "behaviour). Use the SAME value as train_sparqlce.py.")
    ap.add_argument("--min-delta", type=float, default=0.0)
    ap.add_argument("--lr", type=float, default=None,
                    help="learning rate. Default: 1e-5 LINEARLY SCALED from the "
                         "notebook's effective batch of 32, i.e. "
                         "1e-5 * (batch_size*grad_accum)/32. Fixing 1e-5 while "
                         "raising the batch to 128 leaves ~8x fewer optimizer "
                         "steps at an unchanged step size and badly undertrains "
                         "the baseline -- pass an explicit value to override.")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--grad-accum", type=int, default=2, help="effective batch 32")
    ap.add_argument("--scale", type=float, default=20.0, help="1 / tau, tau=0.05")
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--warmup-frac", type=float, default=0.05)
    ap.add_argument("--weight-decay", type=float, default=1e-2)
    ap.add_argument("--train-file", default=None,
                    help="self-contained training split (scripts/build_rq1_twoaxis.py "
                         "--from-split train). Must be the SAME file SparqlCE "
                         "trains on, or the two are not comparable.")
    ap.add_argument("--val-file", default=None, help="see --train-file")
    ap.add_argument("--ckpt-root", default=str(ROOT_DIR / "checkpoints" / "e5_ft"))
    ap.add_argument("--amp", action="store_true")
    args = ap.parse_args()

    eff_batch = args.batch_size * args.grad_accum
    if args.lr is None:
        args.lr = 1e-5 * eff_batch / 32.0
        print(f"[e5-ft] --lr not given: linearly scaling the notebook's 1e-5 at "
              f"effective batch 32 to {args.lr:.2e} at effective batch {eff_batch}")

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    paths = KG_PATHS[args.kg]
    if not (args.train_file or args.val_file) and not paths["data"].exists():
        raise SystemExit(f"{paths['data']} not found (see datasets/README.md).")

    ckpt_dir = Path(args.ckpt_root) / args.kg / f"seed{args.seed}"
    best_path = ckpt_dir / "best_model"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def collate(batch):
        a, p, n = zip(*batch)
        tk = lambda xs: tokenizer([E5_PREFIX + x for x in xs], padding=True,
                                  truncation=True, max_length=args.max_len,
                                  return_tensors="pt")
        return tk(list(a)), tk(list(p)), tk(list(n))

    if args.train_file or args.val_file:
        # Per-split files, as emitted by scripts/build_rq1_twoaxis.py. The
        # baseline MUST read the same data as SparqlCE: training it on the old
        # group split while SparqlCE trains on the component split would compare
        # two models over different corpora, and the old split's train anchors
        # land inside the new test components.
        if not (args.train_file and args.val_file):
            raise SystemExit("per-split mode needs both --train-file and --val-file")
        train_ds = SquareTextDataset(args.train_file)
        val_ds = SquareTextDataset(args.val_file)
        dataset = train_ds
        print(f"per-split files: train={len(train_ds)} val={len(val_ds)}")
    elif paths["splits"].exists():
        dataset = SquareTextDataset(paths["data"])
        spl = json.load(open(paths["splits"]))
        val_idx = spl.get("valid_idx", spl.get("val_idx"))
        # Subset indexes the dataset's KEPT groups, not the file's records; those
        # coincide only while nothing is filtered out. Asserted rather than
        # assumed -- a misalignment would silently train on the wrong queries.
        with open(paths["data"]) as f:
            n_records = len(json.load(f))
        if len(dataset) != n_records:
            raise SystemExit(
                f"{paths['data'].name} has {n_records} records but {len(dataset)} "
                f"survived filtering, so the indices in {paths['splits'].name} no "
                f"longer address the intended queries. Use --train-file/--val-file.")
        train_ds = Subset(dataset, spl["train_idx"])
        val_ds = Subset(dataset, val_idx)
        print(f"fixed split: train={len(train_ds)} val={len(val_ds)}")
    else:
        dataset = SquareTextDataset(paths["data"])
        print(f"WARNING: {paths['splits']} not found; using a seeded random split")
        gen = torch.Generator().manual_seed(args.seed)
        n = len(dataset)
        n_tr, n_va = int(0.8 * n), int(0.1 * n)
        train_ds, val_ds, _ = random_split(dataset, [n_tr, n_va, n - n_tr - n_va],
                                           generator=gen)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate, num_workers=2, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            collate_fn=collate, num_workers=2)

    model = E5SparqlEmbedder(MODEL_NAME).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)
    total_steps = max((len(train_loader) // args.grad_accum) * args.epochs, 1)
    sched = get_cosine_schedule_with_warmup(
        opt, num_warmup_steps=int(args.warmup_frac * total_steps),
        num_training_steps=total_steps)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    n_par = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"E5-FT {args.kg} seed={args.seed} on {device}: {n_par / 1e6:.1f}M params, "
          f"up to {args.epochs} epochs (patience={args.patience or 'off'}, "
          f"effective batch {args.batch_size * args.grad_accum})")

    best_mrr, best_epoch, no_improve, history = -1.0, -1, 0, []
    for epoch in range(1, args.epochs + 1):
        model.train()
        opt.zero_grad()
        run_loss, n_steps = 0.0, 0
        for step, (a_tok, p_tok, n_tok) in enumerate(
                tqdm(train_loader, desc=f"epoch {epoch:02d}/{args.epochs}"), start=1):
            a_tok = {k: v.to(device) for k, v in a_tok.items()}
            p_tok = {k: v.to(device) for k, v in p_tok.items()}
            n_tok = {k: v.to(device) for k, v in n_tok.items()}
            with torch.cuda.amp.autocast(enabled=args.amp):
                loss = info_nce_loss(model(a_tok), model(p_tok), model(n_tok),
                                     args.scale) / args.grad_accum
            scaler.scale(loss).backward()
            if step % args.grad_accum == 0:
                scaler.step(opt)
                scaler.update()
                opt.zero_grad()
                sched.step()
            run_loss += loss.item() * args.grad_accum
            n_steps += 1

        val = evaluate_ranking(model, val_loader, device)
        history.append({"epoch": epoch, "loss": run_loss / max(n_steps, 1), **val})
        print(f"  epoch {epoch:02d}  loss {run_loss / max(n_steps, 1):.4f}  "
              f"val MRR {val['mrr']:.4f}")

        if val["mrr"] > best_mrr + args.min_delta:
            best_mrr, best_epoch, no_improve = val["mrr"], epoch, 0
            model.backbone.save_pretrained(best_path)
            tokenizer.save_pretrained(best_path)
            print(f"    best saved (val MRR={best_mrr:.4f}) -> {best_path}")
        else:
            no_improve += 1
            if args.patience and no_improve >= args.patience:
                print(f"    early stop: no val-MRR gain > {args.min_delta} for "
                      f"{args.patience} epochs (best {best_mrr:.4f} @ epoch {best_epoch})")
                break

    json.dump({"kg": args.kg, "seed": args.seed, "model": MODEL_NAME,
               "lr": args.lr, "effective_batch_lr_scaled_from": 32,
               "epochs_max": args.epochs, "best_epoch": best_epoch,
               "patience": args.patience, "best_val_mrr": best_mrr,
               "effective_batch": args.batch_size * args.grad_accum,
               "scale": args.scale,
               # Which corpus this run saw. run_seed_sweep.sh reads this file to
               # verify the baseline was trained comparably; without these it
               # cannot tell a component-split run from a group-split one.
               "train_file": args.train_file, "val_file": args.val_file,
               "data": str(paths["data"]), "splits": str(paths["splits"]),
               "history": history},
              open(ckpt_dir / "run_metadata.json", "w"), indent=2)
    print(f"\ndone: best val MRR {best_mrr:.4f} @ epoch {best_epoch} -> {best_path}")


if __name__ == "__main__":
    main()
