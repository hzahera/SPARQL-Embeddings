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


def _humanize_off():
    """Whether DBpedia IRI humanization is disabled, without importing torch."""
    from sparqlce.dbpedia import humanization_disabled
    return humanization_disabled()


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


def _groups_of(ds):
    """The dataset records behind a loader, through a Subset if there is one."""
    if isinstance(ds, Subset):
        return [ds.dataset.groups[i] for i in ds.indices]
    return ds.groups


@torch.no_grad()
def evaluate_ranking(model, loader, device):
    """MRR of the BEST correct answer for each anchor.

    Column j of the similarity matrix holds anchor j's positive, and the naive
    version of this treated the diagonal as the only correct cell. That is only
    valid when no two anchors share a positive. On the component splits they
    share them constantly -- the DBpedia validation file has 38 positives per
    anchor drawn from a 240-query pool, so roughly one column in six is a
    genuinely correct answer for any given anchor and was being scored as an
    error. Val MRR was pinned near 0.09 no matter how good the model was, and
    early stopping fired at epoch 1 on what was effectively its ceiling.

    An anchor is now credited for whichever of ITS positives ranks highest, the
    same "find a result-equivalent query" task the RQ1 evaluation scores. This
    requires the eval dataset to be deterministic (`deterministic=True`), so
    that column j really is groups[j]'s first positive.
    """
    model.eval()
    anchors, positives = [], []
    for batch in loader:
        a_emb, _ = model(batch[0].to(device))
        p_emb, _ = model(batch[1].to(device))
        anchors.append(F.normalize(a_emb, dim=1).cpu())
        positives.append(F.normalize(p_emb, dim=1).cpu())
    A, P = torch.cat(anchors), torch.cat(positives)
    sims = A @ P.T
    n = sims.size(0)

    groups = _groups_of(loader.dataset)
    assert len(groups) == n, (
        f"{len(groups)} groups but {n} embedded columns -- the loader must not "
        f"shuffle or drop for evaluation")
    col_text = [g["positives"][0][0] for g in groups]
    rel = torch.zeros(n, n, dtype=torch.bool)
    for i, g in enumerate(groups):
        pos_set = {q for q, _ in g["positives"]}
        for j, t in enumerate(col_text):
            if t in pos_set:
                rel[i, j] = True
    rel.fill_diagonal_(True)          # the sampled positive is correct by construction

    # Rank of the best-scoring relevant column: how many columns beat it, +1.
    best_rel = sims.masked_fill(~rel, float("-inf")).max(dim=1).values
    ranks = (sims > best_rel.unsqueeze(1)).sum(dim=1).float() + 1.0
    return {
        "mrr": (1.0 / ranks).mean().item(),
        "hits@1": (ranks == 1).float().mean().item(),
        "hits@5": (ranks <= 5).float().mean().item(),
    }


def build_loaders(data_path, splits_path, seed, batch_size, graph_builder,
                  hard_negatives=False, max_neg=8, augment_rename_prob=0.0,
                  split_files=None, query_stream=False,
                  query_max_length=256, constant_stream=False,
                  edge_type_fn=None):
    """Build train/val/test loaders.

    Two modes:
      * in-batch (default): one positive per anchor, negatives are other batch
        items. Collated by semantic_collate_fn_inbatch.
      * hard_negatives: one positive + `max_neg` curated hard negatives per
        anchor, collated by semantic_collate_fn so the loss can consume them.

    Augmentation (random variable renaming) is applied ONLY to the training
    view; val/test use a separate, deterministic dataset so metrics are stable.

    Data can arrive two ways. `split_files` is a {train,valid,test: path} mapping
    of self-contained per-split files (what build_rq1_twoaxis.py emits) and is
    preferred. The older path loads ONE file and slices it with integer indices
    from `splits_path` -- which is only correct while every record in that file
    survives SemanticSPARQLDataset's filtering, since the indices address the
    dataset's kept groups rather than the file's records. That holds for the
    shipped DBpedia triplets (all 4,713 qualify) but silently misaligns the
    moment a record is dropped, so it is asserted below rather than assumed.
    """
    if hard_negatives:
        collate = semantic_collate_fn
        common = dict(in_batch_negatives=False, max_pos_per_anchor=1, max_neg_per_anchor=max_neg)
    else:
        collate = semantic_collate_fn_inbatch
        common = dict(in_batch_negatives=True)
    # Every view -- train, val, test -- must agree, or the model is selected on
    # a different input representation than it was trained on.
    common["query_stream"] = query_stream
    common["query_max_length"] = query_max_length
    common["constant_stream"] = constant_stream
    common["edge_type_fn"] = edge_type_fn

    mk = lambda ds, shuffle: DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                                        collate_fn=collate)

    if split_files:
        # Self-contained files: no index arithmetic, so nothing can misalign.
        train_ds = SemanticSPARQLDataset(
            str(split_files["train"]), graph_builder=graph_builder,
            augment_rename_prob=augment_rename_prob, augment_seed=seed, **common)
        val_ds = SemanticSPARQLDataset(
            str(split_files["valid"]), graph_builder=graph_builder,
            augment_rename_prob=0.0, deterministic=True, **common)
        test_ds = SemanticSPARQLDataset(
            str(split_files["test"]), graph_builder=graph_builder,
            augment_rename_prob=0.0, deterministic=True, **common)
        print(f"Using per-split files: "
              f"train={len(train_ds)} val={len(val_ds)} test={len(test_ds)} groups")
        return mk(train_ds, True), mk(val_ds, False), mk(test_ds, False)

    train_dataset = SemanticSPARQLDataset(
        str(data_path), graph_builder=graph_builder,
        augment_rename_prob=augment_rename_prob, augment_seed=seed, **common,
    )
    # Deterministic view for evaluation (no augmentation).
    eval_dataset = SemanticSPARQLDataset(
        str(data_path), graph_builder=graph_builder,
        augment_rename_prob=0.0, deterministic=True, **common,
    )

    if splits_path.exists():
        with open(splits_path) as f:
            splits = json.load(f)
        train_idx = splits.get("train_idx", splits.get("train"))
        val_idx = splits.get("val_idx", splits.get("valid_idx", splits.get("val")))
        test_idx = splits.get("test_idx", splits.get("test"))
        assert train_idx and val_idx and test_idx, f"could not parse splits from {splits_path}"
        # The indices address positions in the SOURCE FILE, but Subset addresses
        # positions in the dataset's kept groups. Those coincide only if nothing
        # was filtered out; otherwise every split silently points at the wrong
        # queries, which no metric would reveal.
        with open(data_path) as f:
            n_records = len(json.load(f))
        assert len(train_dataset) == n_records, (
            f"{Path(data_path).name} has {n_records} records but "
            f"{len(train_dataset)} survived filtering, so the integer indices in "
            f"{Path(splits_path).name} no longer address the intended queries. "
            f"Use per-split files (--train-file/--val-file/--test-file, produced "
            f"by scripts/build_rq1_twoaxis.py) instead of an index split.")
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

    return mk(train_ds, True), mk(val_ds, False), mk(test_ds, False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kg", required=True, choices=["dbpedia", "wikidata"])
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--arch", default="hybrid", choices=["hybrid", "dbpedia"],
                     help="hybrid: HybridSPARQLEncoder, the paper model. "
                          "dbpedia: DBpediaSPARQLEncoder -- E5 query trunk with "
                          "a conditioned CONSTANT stream and typed GATv2 fused "
                          "residually, built for DBpedia's axis-C task (74.3%% "
                          "of its positives share the anchor's template, so the "
                          "constant, not the shape, is what separates them). "
                          "Implies --query-stream, --join-hubs, humanized IRIs, "
                          "populated edge types, and output_dim 768.")
    ap.add_argument("--ablation", default="full",
                     choices=["full", "no_gnn", "no_decomposition",
                              "no_const", "text_only"],
                     help="hybrid arch -- full: paper model. no_gnn: skip GATv2 "
                          "message passing (bag of node embeddings). "
                          "no_decomposition: whole query as one node. "
                          "dbpedia arch -- full / no_const (drop the constant "
                          "stream) / no_gnn (drop message passing) / text_only "
                          "(the trunk alone: a flat E5 encoder trained by OUR "
                          "recipe, which is the architecture-controlled "
                          "baseline).")
    ap.add_argument("--query-stream", action="store_true",
                     help="Text branch reads the whole query STRING instead of a "
                          "mean over node-label embeddings. Without it the text "
                          "side is a bag of node labels -- no token order, no "
                          "syntax -- i.e. strictly less informed than the flat E5 "
                          "control it is compared against, so the gate cannot "
                          "recover that control by saturating. With it, gate->1 "
                          "reproduces a flat E5 encoder and the graph can only add.")
    ap.add_argument("--join-hubs", action="store_true",
                     help="Wire a shared variable through one hub node per "
                          "variable instead of linking all participants pairwise. "
                          "The pairwise form makes joined nodes mutually adjacent, "
                          "collapsing graph diameter to 1 on 88%% of DBpedia "
                          "queries (6.9%% on Wikidata) -- a second GATv2 layer has "
                          "nothing to propagate there. Hubs keep participants 2 "
                          "hops apart: DBpedia diameter<=1 drops to 0%%, Wikidata "
                          "is unchanged.")
    ap.add_argument("--no-join-hubs", action="store_true",
                     help="force pairwise join wiring even under --arch dbpedia, "
                          "which otherwise turns hubs on. Measured on the 4,713 "
                          "DBpedia anchors: pairwise leaves 38.3%% of graphs at "
                          "diameter <=1, hubs 3.4%%.")
    ap.add_argument("--no-grad-checkpointing", action="store_true",
                     help="disable gradient checkpointing on the text backbone "
                          "(--arch dbpedia turns it on). It costs ~30%% step "
                          "time and is what makes --hard-negatives fit: the "
                          "DBpedia encoder runs the whole query through the "
                          "backbone for every graph, so B*(2+max_neg) query "
                          "streams are encoded per step.")
    ap.add_argument("--calibrate-inbatch", action="store_true",
                     help="regress every off-diagonal in-batch pair's cosine to "
                          "0 in the calibration term (--arch dbpedia only). Sound "
                          "ONLY on the component splits, where two anchors from "
                          "different components provably have disjoint answers.")
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
    ap.add_argument("--data", default=None,
                     help="override the KG's triplets file, keeping --splits. Use "
                          "for a corpus that differs only in its NEGATIVES (see "
                          "scripts/mine_train_hard_negatives.py): the record order "
                          "and positives must match, or the integer split indices "
                          "no longer address the intended queries.")
    ap.add_argument("--train-file", default=None,
                     help="self-contained training split (scripts/build_rq1_twoaxis.py "
                          "--from-split train). Bypasses the integer-index split "
                          "entirely; pass all three of --train/--val/--test-file. "
                          "Required for the component splits, whose files drop "
                          "unjudgeable anchors and so cannot be index-addressed.")
    ap.add_argument("--val-file", default=None, help="see --train-file")
    ap.add_argument("--test-file", default=None, help="see --train-file")
    ap.add_argument("--output-dim", type=int, default=None,
                     help="final embedding width (default: Config.OUTPUT_DIM, "
                          "currently 128). The E5-Finetuned baseline embeds at "
                          "768, so the default compares a 128-d representation "
                          "against a 6x wider one -- which is a capacity "
                          "difference, not an architectural one. Pass 768 to "
                          "remove that confound.")
    ap.add_argument("--query-max-length", type=int, default=384,
                     help="token budget for --query-stream (default 384, above the "
                          "corpus maximum of 364, so nothing truncates). Lower "
                          "it to fit memory, but it changes the input the "
                          "weights see, so it is recorded in the checkpoint and "
                          "two runs are only comparable at the same value.")
    ap.add_argument("--single-lr", type=float, default=None,
                     help="if set, use one LR for all params instead of differential LRs")
    ap.add_argument("--ckpt-root", default=str(ROOT_DIR / "checkpoints"))
    ap.add_argument("--allow-unlabeled", action="store_true",
                     help="skip the Wikidata label-map check (train on raw "
                          "opaque P/Q ids intentionally, e.g. for an ablation)")
    args = ap.parse_args()

    if args.kg == "wikidata" and not args.allow_unlabeled and not _graph_builder_module.WIKIDATA_LABELS:
        disabled = os.environ.get("SPARQLCE_NO_WIKIDATA_LABELS", "").strip().lower() \
            in ("1", "true", "yes", "on")
        why = ("SPARQLCE_NO_WIKIDATA_LABELS is set, so labels are disabled"
               if disabled else
               "datasets/wikidata_labels.json is missing or empty")
        raise SystemExit(
            f"{why} -- training on Wikidata would use raw opaque P/Q ids instead "
            "of English labels (see sparqlce/graph_builder.py). Pass "
            "--allow-unlabeled if this is intentional (e.g. the no-expansion "
            "run), or unset the env var / run scripts/build_wikidata_labels.py."
        )

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --arch dbpedia is a package of input choices as much as an architecture,
    # and every one of them is load-bearing (see sparqlce/dbpedia.py). Resolving
    # them here rather than expecting the caller to pass five flags is what
    # stops a run from being silently handicapped: without --query-stream the
    # encoder's trunk would be a bag of node labels, which is the single
    # biggest reason the hybrid model reads as weaker than the flat E5 baseline.
    if args.arch == "dbpedia":
        if args.ablation not in ("full", "no_const", "no_gnn", "text_only"):
            raise SystemExit(
                f"--ablation {args.ablation} is a hybrid-arch ablation; "
                f"--arch dbpedia takes full/no_const/no_gnn/text_only")
        if not args.query_stream:
            print("[arch=dbpedia] enabling --query-stream (required by the encoder)")
            args.query_stream = True
        if not args.join_hubs and not args.no_join_hubs:
            print("[arch=dbpedia] enabling --join-hubs (38.3% -> 3.4% of DBpedia "
                  "graphs at diameter <=1)")
            args.join_hubs = True
        if args.output_dim is None:
            args.output_dim = 768
            print("[arch=dbpedia] output_dim=768 to match the E5-Finetuned baseline")
    elif args.ablation in ("no_const", "text_only"):
        raise SystemExit(f"--ablation {args.ablation} requires --arch dbpedia")
    if args.no_join_hubs:
        args.join_hubs = False
    if args.calibrate_inbatch and args.arch != "dbpedia":
        raise SystemExit("--calibrate-inbatch is implemented for --arch dbpedia only")

    # The variant belongs in the path. --query-stream/--join-hubs change the
    # model without changing the ablation tag, so every variant of one seed
    # otherwise writes to the same directory and the last run silently replaces
    # a checkpoint an earlier result was computed from. Legacy runs (neither
    # flag) keep their original path, so nothing already on disk moves.
    variant = "".join(s for s, on in (("_qs", args.query_stream),
                                      ("_jh", args.join_hubs)) if on)
    # The TRAINING DATA belongs in the path too, not just the architecture
    # variant. A component-split run and an index-split run of the same variant
    # are different models on different corpora, and both were writing to
    # seed{N}_{ablation}_qs_jh -- so the second silently replaced the first, and
    # any score already exported from it became attributable to weights no
    # longer on disk. Index-split runs keep the bare path, so nothing moves.
    data_tag = f"_{Path(args.train_file).stem}" if args.train_file else ""
    # A different corpus is a different run, even at the same split.
    if args.data and Path(args.data).stem != KG_PATHS[args.kg]["data"].stem:
        data_tag += f"_{Path(args.data).stem}"
    # Width changes the model too, and load_state_dict fails on a mismatch, so a
    # 768-d run must not land on a 128-d run's path. Only tagged when set
    # explicitly, so existing default-width checkpoints keep their paths.
    dim_tag = f"_d{args.output_dim}" if args.output_dim else ""
    # Two encoder classes writing to one path would make a checkpoint's identity
    # depend on which run happened last, and load_state_dict would fail late
    # with a key mismatch rather than early with a name clash.
    arch_tag = "" if args.arch == "hybrid" else f"_{args.arch}"
    ckpt_dir = (Path(args.ckpt_root) / args.kg
                / f"seed{args.seed}{arch_tag}_{args.ablation}{variant}{dim_tag}{data_tag}")
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    edge_type_fn = None
    if args.arch == "dbpedia":
        from sparqlce.dbpedia import DBpediaGraphBuilder, edge_type_id
        from sparqlce.dbpedia import humanization_disabled
        graph_builder = DBpediaGraphBuilder(join_hubs=args.join_hubs)
        edge_type_fn = edge_type_id
        print(f"graph_config: arch=dbpedia query_stream={args.query_stream} "
              f"join_hubs={args.join_hubs} constant_stream=True edge_types=True "
              f"humanize={'OFF' if humanization_disabled() else 'ON'}")
        if not humanization_disabled():
            print("[fairness] IRI humanization is ON. The text baselines must read "
                  "the same vocabulary -- run finetune_e5_baseline.py with "
                  "--humanize, or set SPARQLCE_NO_DBPEDIA_HUMANIZE=1 for every "
                  "model. Crediting this preprocessing to the architecture is the "
                  "same trap the Wikidata label expansion documents.")
    else:
        # no_decomposition swaps the graph builder (whole-query single node);
        # no_gnn keeps the real graph but the encoder skips message passing.
        graph_builder = (WholeQueryGraphBuilder() if args.ablation == "no_decomposition"
                         else SPARQLGraphBuilder(join_hubs=args.join_hubs))
        print(f"graph_config: query_stream={args.query_stream} join_hubs={args.join_hubs}")
    encoder_mode = "no_gnn" if args.ablation == "no_gnn" else "full"

    paths = dict(KG_PATHS[args.kg])
    if args.data:
        paths["data"] = Path(args.data)
    split_files = None
    if args.train_file or args.val_file or args.test_file:
        missing = [n for n, v in (("--train-file", args.train_file),
                                  ("--val-file", args.val_file),
                                  ("--test-file", args.test_file)) if not v]
        if missing:
            raise SystemExit(
                f"per-split mode needs all three files; missing {', '.join(missing)}")
        split_files = {"train": args.train_file, "valid": args.val_file,
                       "test": args.test_file}
    train_loader, val_loader, test_loader = build_loaders(
        paths["data"], paths["splits"], args.seed, args.batch_size, graph_builder,
        hard_negatives=args.hard_negatives, max_neg=args.max_neg,
        augment_rename_prob=args.augment_rename_prob, split_files=split_files,
        query_stream=args.query_stream,
        query_max_length=args.query_max_length,
        constant_stream=(args.arch == "dbpedia"),
        edge_type_fn=edge_type_fn,
    )
    print(f"train={len(train_loader.dataset)} val={len(val_loader.dataset)} test={len(test_loader.dataset)}")

    if args.arch == "dbpedia":
        from sparqlce.dbpedia_encoder import DBpediaSPARQLEncoder, calibrated_infonce
        model = DBpediaSPARQLEncoder(ablation=args.ablation,
                                     output_dim=args.output_dim).to(device)
        if not args.no_grad_checkpointing:
            model.enable_gradient_checkpointing()
            print("gradient checkpointing: ON (backbone activations recomputed "
                  "in backward; ~30% slower, and what makes --hard-negatives fit)")
    else:
        model = HybridSPARQLEncoder(ablation=encoder_mode,
                                    output_dim=args.output_dim).to(device)

    if args.single_lr is not None:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.single_lr, weight_decay=args.weight_decay)
    else:
        # Select the backbone by IDENTITY, not by a substring of the parameter
        # name. The old test matched any name containing "text", which caught
        # `text_proj` as well as `text_enc` -- so the text projection head
        # trained at the backbone's 1e-5 while graph_proj and gate_fc trained at
        # 1e-4, biasing the fusion toward the graph stream before either had
        # learned anything. This changes the optimizer for hybrid runs too, and
        # is the intended fix.
        text_ids = {id(p) for p in model.text_enc.parameters()}
        text_params = [p for p in model.parameters() if id(p) in text_ids and p.requires_grad]
        rest_params = [p for p in model.parameters() if id(p) not in text_ids and p.requires_grad]
        print(f"param groups: backbone={sum(p.numel() for p in text_params):,} @ "
              f"{args.lr_text:g}, rest={sum(p.numel() for p in rest_params):,} @ "
              f"{args.lr_rest:g}")
        optimizer = torch.optim.AdamW([
            {"params": text_params, "lr": args.lr_text},
            {"params": rest_params, "lr": args.lr_rest},
        ], weight_decay=args.weight_decay)

    best_mrr = -1.0
    best_epoch = 0
    history = []
    epochs_no_improve = 0
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)
    print(f"Training {args.kg}/{args.ablation} seed={args.seed} on {device} for up to "
          f"{args.epochs} epochs (patience={args.patience or 'off'}, amp={args.amp})")
    for epoch in range(1, args.epochs + 1):
        model.train()
        sum_loss = sum_ce = sum_reg = sum_share = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch:02d}/{args.epochs}")
        for batch in pbar:
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=args.amp):
                if args.arch == "dbpedia":
                    # Two-sided calibration: the negatives carry their true
                    # Jaccard too. The positives-only form is MSE(cos, 1) on
                    # 78.9% of DBpedia's positives -- a pull-together with no
                    # push-apart, which is how the embedding space ends up at
                    # mean cosine 0.99 everywhere.
                    a_emb, share = model(batch[0].to(device))
                    sum_share += share
                    p_emb, _ = model(batch[1].to(device))
                    if args.hard_negatives:
                        n_emb, _ = model(batch[2].to(device))
                        neg_slices, pos_js, neg_js = batch[4], batch[5].to(device), batch[6]
                    else:
                        n_emb, neg_slices, neg_js = None, None, None
                        pos_js = batch[2].to(device)
                    loss, loss_ce, loss_reg = calibrated_infonce(
                        a_emb, p_emb, pos_js, args.temperature, args.lambda_reg,
                        n_emb=n_emb, neg_slices=neg_slices, neg_j=neg_js,
                        calibrate_inbatch=args.calibrate_inbatch)
                elif args.hard_negatives:
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
                if args.arch != "dbpedia":
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
        # Per-epoch history, in the same shape finetune_e5_baseline.py records.
        # Without it a run leaves no trace of whether val MRR was still climbing
        # when training stopped, or had turned over into overfitting -- which is
        # exactly what has to be checked when val and test disagree.
        # share = fraction of the embedding's norm contributed by the constant
        # and graph streams (--arch dbpedia only; 0 elsewhere). If it stays near
        # 0 the extra streams are not being used and the honest report is that
        # the flat trunk does the job -- which is not visible from MRR alone.
        history.append({"epoch": epoch, "loss": sum_loss / nb,
                        "ce": sum_ce / nb, "reg": sum_reg / nb,
                        "non_text_share": sum_share / nb, **val})
        share_str = (f" share={sum_share/nb:.3f}" if args.arch == "dbpedia" else "")
        print(f"Epoch {epoch:02d} | train_loss={sum_loss/nb:.4f} (ce={sum_ce/nb:.4f} reg={sum_reg/nb:.4f}){share_str} "
              f"| val MRR={val['mrr']:.4f} Hits@1={val['hits@1']:.4f} Hits@5={val['hits@5']:.4f}")

        if val["mrr"] > best_mrr + args.min_delta:
            best_mrr = val["mrr"]
            best_epoch = epoch
            epochs_no_improve = 0
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict(),
                        "val_metrics": val, "ablation": args.ablation,
                        # Which encoder CLASS these weights belong to.
                        # SPARQLEmbedder dispatches on it; without it a dbpedia
                        # checkpoint would be rebuilt as a HybridSPARQLEncoder
                        # and fail on a key mismatch.
                        "arch": args.arch,
                        # Record the embedding width: the eval-time encoder must
                        # be built at the same width or load_state_dict fails.
                        "output_dim": model.output_dim,
                        # These two change the INPUT, not any tensor shape, so a
                        # mismatch at inference loads cleanly and silently scores
                        # a different model. SPARQLEmbedder reads them back.
                        "graph_config": {"query_stream": args.query_stream,
                                         "join_hubs": args.join_hubs,
                                         "query_max_length": args.query_max_length,
                                         "constant_stream": args.arch == "dbpedia",
                                         "edge_types": edge_type_fn is not None,
                                         "humanize": args.arch == "dbpedia"
                                         and not _humanize_off()}},
                       ckpt_dir / "best_model.pt")
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
            "kg": args.kg, "seed": args.seed, "arch": args.arch,
            "ablation": args.ablation,
            "calibrate_inbatch": args.calibrate_inbatch,
            "humanize": args.arch == "dbpedia" and not _humanize_off(),
            "epochs": args.epochs, "epochs_ran": epoch, "batch_size": args.batch_size,
            "lr_text": args.lr_text, "lr_rest": args.lr_rest,
            "single_lr": args.single_lr, "min_delta": args.min_delta,
            "patience": args.patience, "best_epoch": best_epoch, "amp": args.amp,
            "temperature": args.temperature, "lambda_reg": args.lambda_reg,
            "hard_negatives": args.hard_negatives, "max_neg": args.max_neg,
            "augment_rename_prob": args.augment_rename_prob,
            "query_stream": args.query_stream, "join_hubs": args.join_hubs,
            # Which data a run saw is part of its identity: a component-split run
            # and an index-split run are not comparable, and nothing else in this
            # file distinguishes them.
            "train_file": args.train_file, "val_file": args.val_file,
            "test_file": args.test_file,
            "data": str(paths["data"]), "splits": str(paths["splits"]),
            "output_dim": model.output_dim,
            "best_val_mrr": best_mrr, "history": history,
            "val_metrics": best["val_metrics"], "test_metrics": test,
        }, f, indent=2)
    print(f"Saved checkpoint + metadata to {ckpt_dir}")


if __name__ == "__main__":
    main()
