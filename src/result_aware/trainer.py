from __future__ import annotations

import math
import shutil
import time
from pathlib import Path
from typing import Any

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data import DynamicTrainingDataset, QueryRecord, collate_training_batch
from .evaluation import evaluate_records
from .loss import asymmetric_infonce_loss
from .mining import mine_hard_negatives
from .model import HFEmbeddingModel
from .utils import ensure_dir, nested_get, set_global_seed, write_json


def _autocast_context(device: torch.device, amp: str):
    amp = str(amp).lower()
    if device.type != "cuda" or amp in {"none", "false", "off"}:
        return torch.autocast(device_type=device.type, enabled=False)
    if amp == "bf16":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    if amp == "fp16":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    raise ValueError(f"Unsupported amp mode {amp!r}")


def train_one_seed(
    model_cfg: dict[str, Any],
    train_records: list[QueryRecord],
    valid_records: list[QueryRecord],
    training_cfg: dict[str, Any],
    eval_cfg: dict[str, Any],
    seed: int,
    output_dir: str | Path,
    device: str = "cuda",
) -> dict[str, Any]:
    set_global_seed(seed)
    out = ensure_dir(output_dir)
    mining_dir = ensure_dir(out / "mining")
    checkpoint_dir = out / "best_model"

    model = HFEmbeddingModel(model_cfg, device=device)
    if bool(training_cfg.get("gradient_checkpointing", True)):
        model.enable_gradient_checkpointing()
    model.train()

    explicit_negatives = int(training_cfg.get("explicit_negatives", 4))
    hard_fraction = float(training_cfg.get("hard_fraction", 0.75))
    dataset = DynamicTrainingDataset(
        train_records,
        seed=seed,
        explicit_negatives=explicit_negatives,
        hard_fraction=hard_fraction,
    )

    batch_size = int(training_cfg.get("batch_size", 16))
    grad_accum = int(training_cfg.get("gradient_accumulation_steps", 2))
    epochs = int(training_cfg.get("epochs", 20))
    num_workers = int(training_cfg.get("num_workers", 0))
    lr = float(training_cfg.get("learning_rate", 1e-5))
    weight_decay = float(training_cfg.get("weight_decay", 0.01))
    warmup_ratio = float(training_cfg.get("warmup_ratio", 0.05))
    max_grad_norm = float(training_cfg.get("max_grad_norm", 1.0))
    temperature = float(training_cfg.get("temperature", 0.05))
    amp = str(training_cfg.get("amp", "bf16"))
    mining_start = int(training_cfg.get("mining_start_epoch", 2))
    mining_refresh = int(training_cfg.get("mining_refresh_every", 2))
    hard_top_k = int(training_cfg.get("hard_top_k", 8))
    eval_every = int(training_cfg.get("eval_every_epochs", 1))
    selection_metric = str(training_cfg.get("selection_metric", "ranking.ndcg@10"))

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    steps_per_epoch = math.ceil(len(dataset) / batch_size)
    updates_per_epoch = math.ceil(steps_per_epoch / grad_accum)
    total_updates = updates_per_epoch * epochs
    warmup_steps = int(round(total_updates * warmup_ratio))
    try:
        from transformers import get_cosine_schedule_with_warmup
    except ImportError as exc:
        raise ImportError("transformers is required for the cosine scheduler") from exc
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_updates,
    )

    scaler = torch.cuda.amp.GradScaler(enabled=(model.device.type == "cuda" and amp.lower() == "fp16"))

    best_score = -float("inf")
    best_epoch = None
    history: list[dict[str, Any]] = []
    hard_pools: dict[int, list[str]] = {}
    global_update = 0
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        should_mine = (
            epoch >= mining_start
            and (epoch - mining_start) % mining_refresh == 0
        )
        if should_mine:
            hard_pools = mine_hard_negatives(
                model,
                train_records,
                top_k=hard_top_k,
                encode_batch_size=int(eval_cfg.get("encode_batch_size", 64)),
                save_path=mining_dir / f"epoch_{epoch:03d}.json",
            )

        dataset.set_epoch(epoch, hard_pools if epoch >= mining_start else {})
        generator = torch.Generator()
        generator.manual_seed(seed * 1000 + epoch)
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=collate_training_batch,
            generator=generator,
            pin_memory=(model.device.type == "cuda"),
        )

        model.train()
        optimizer.zero_grad(set_to_none=True)
        epoch_loss = 0.0
        epoch_acc = 0.0
        epoch_batches = 0
        progress = tqdm(loader, desc=f"Seed {seed} epoch {epoch}/{epochs}")

        for batch_idx, batch in enumerate(progress, start=1):
            anchors = batch["anchors"]
            positives = batch["positives"]
            negatives = batch["negatives"]
            known_positives = batch["known_positives"]
            bsz = len(anchors)
            nneg = len(negatives[0])
            flat_negatives = [q for row in negatives for q in row]
            all_texts = anchors + positives + flat_negatives

            with _autocast_context(model.device, amp):
                embeddings = model.forward_texts(all_texts)
                anchor_emb = embeddings[:bsz]
                positive_emb = embeddings[bsz : 2 * bsz]
                negative_emb = embeddings[2 * bsz :].reshape(bsz, nneg, -1)
                loss, diagnostics = asymmetric_infonce_loss(
                    anchor_emb,
                    positive_emb,
                    negative_emb,
                    anchors=anchors,
                    positives=positives,
                    negatives=negatives,
                    known_positives=known_positives,
                    temperature=temperature,
                )
                scaled_loss = loss / grad_accum

            if scaler.is_enabled():
                scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()

            epoch_loss += float(loss.detach().item())
            epoch_acc += float(diagnostics["batch_retrieval_accuracy"])
            epoch_batches += 1

            do_update = (batch_idx % grad_accum == 0) or (batch_idx == len(loader))
            if do_update:
                if scaler.is_enabled():
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                if scaler.is_enabled():
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_update += 1

            progress.set_postfix(
                loss=f"{loss.item():.4f}",
                acc=f"{diagnostics['batch_retrieval_accuracy']:.3f}",
                lr=f"{scheduler.get_last_lr()[0]:.2e}",
            )

        epoch_entry: dict[str, Any] = {
            "epoch": epoch,
            "train_loss": epoch_loss / max(epoch_batches, 1),
            "train_batch_retrieval_accuracy": epoch_acc / max(epoch_batches, 1),
            "global_update": global_update,
            "mined_this_epoch": should_mine,
        }

        if epoch % eval_every == 0:
            valid_result = evaluate_records(
                model,
                valid_records,
                eval_cfg=eval_cfg,
                split_name="validation",
                include_per_anchor=False,
            )
            score = float(nested_get(valid_result, selection_metric))
            epoch_entry["validation"] = valid_result
            epoch_entry["selection_score"] = score
            if score > best_score:
                best_score = score
                best_epoch = epoch
                if checkpoint_dir.exists():
                    shutil.rmtree(checkpoint_dir)
                model.save_pretrained(checkpoint_dir)
                write_json(
                    out / "best_checkpoint.json",
                    {
                        "epoch": best_epoch,
                        "selection_metric": selection_metric,
                        "score": best_score,
                        "path": str(checkpoint_dir),
                    },
                )

        history.append(epoch_entry)
        write_json(out / "training_history.json", history)

    if best_epoch is None:
        # This only happens if eval_every > epochs. Save final model explicitly.
        model.save_pretrained(checkpoint_dir)
        best_epoch = epochs
        best_score = float("nan")

    summary = {
        "seed": seed,
        "model": model_cfg.get("name"),
        "best_epoch": best_epoch,
        "best_validation_score": best_score,
        "selection_metric": selection_metric,
        "epochs": epochs,
        "batch_size": batch_size,
        "gradient_accumulation_steps": grad_accum,
        "explicit_negatives": explicit_negatives,
        "hard_fraction": hard_fraction,
        "hard_top_k": hard_top_k,
        "mining_start_epoch": mining_start,
        "mining_refresh_every": mining_refresh,
        "temperature": temperature,
        "learning_rate": lr,
        "weight_decay": weight_decay,
        "warmup_ratio": warmup_ratio,
        "max_length": model_cfg.get("max_length", 512),
        "elapsed_seconds": time.time() - start_time,
        "checkpoint": str(checkpoint_dir),
    }
    write_json(out / "training_summary.json", summary)
    return summary
