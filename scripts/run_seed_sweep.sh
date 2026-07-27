#!/usr/bin/env bash
# Multi-seed training + evaluation + significance for SparqlCE vs the
# E5-Finetuned baseline, on a single named RQ1 split.
#
# BOTH sides get the same seeds, the same epoch budget, the same early-stopping
# policy and the same eval split -- otherwise the paired test is not
# like-for-like. The significance test compares SEED-AVERAGED per-anchor scores,
# so neither method is represented by one arbitrary training run.
#
# Usage:
#   ./scripts/run_seed_sweep.sh <kg> [n_seeds] [split_json] [patience] [batch] [epochs]
#
#   ./scripts/run_seed_sweep.sh wikidata 5 datasets/wikidata_testset_RQ1_clean.json 5 128 60
#   ./scripts/run_seed_sweep.sh dbpedia  5 datasets/dbpedia_testset_RQ1.json        5 128 60
#
# Prerequisite: SparqlCE must be retrained AFTER the graph_builder.py fixes --
# old checkpoints were trained on the pre-fix representation. This script
# retrains from scratch, so running it satisfies that.
set -euo pipefail

KG="${1:?usage: run_seed_sweep.sh <kg> [n_seeds] [split_json] [patience] [batch] [epochs]}"
N_SEEDS="${2:-5}"
SPLIT="${3:-datasets/${KG}_testset_RQ1_clean.json}"
PATIENCE="${4:-3}"
# Per-step batch size. BOTH models get the same value: with in-batch-negative
# InfoNCE the per-forward batch IS the negative-sampling budget, so a mismatch
# silently hands one method more negatives than the other. NOTE that gradient
# accumulation does NOT substitute -- accumulating gradients over micro-batches
# gives the same negatives per loss computation, only a smoother update. That is
# why E5-FT runs with --grad-accum 1 here instead of the notebook's 16x2.
# Default 128 matches the `checkpoints_b128/` naming of the published SparqlCE
# checkpoints (train_sparqlce.py's own default is 32 -- do not rely on it).
BATCH="${5:-128}"
# Epoch CAP, not a schedule. Both trainers select the best-val-MRR checkpoint
# and stop on --patience, so this only needs to be generous enough that neither
# model is cut off while still improving. At the old default of 20, SparqlCE on
# Wikidata peaked at epoch 19/20 with val MRR still climbing -- i.e. the cap,
# not convergence, was setting the result. train_sparqlce.py uses a constant LR
# so raising this does not change its LR trajectory; E5-FT's cosine schedule
# stretches with it, which is the normal way to train that recipe longer.
EPOCHS="${6:-60}"
SEEDS=(42 43 44 45 46 47 48 49 50 51)

cd "$(dirname "$0")/.."
mkdir -p results/significance

TAG="$(basename "$SPLIT" .json)"
echo "=== KG=$KG  seeds=$N_SEEDS  split=$SPLIT  patience=$PATIENCE  batch=$BATCH  epochs<=$EPOCHS ==="
[ -f "$SPLIT" ] || { echo "split $SPLIT not found -- build it with scripts/build_clean_rq1.py --kg $KG"; exit 1; }

# Fail fast, before burning GPU hours: Wikidata training hard-exits without the
# label map, and it is not tracked in git.
if [ "$KG" = "wikidata" ] && [ ! -s datasets/wikidata_labels.json ]; then
  echo "datasets/wikidata_labels.json missing or empty -- train_sparqlce.py will"
  echo "refuse to train on Wikidata. Run: python scripts/build_wikidata_labels.py"
  exit 1
fi
for f in datasets/${KG}-triplets-dataset.json datasets/splits_${KG}.json; do
  [ -f "$f" ] || { echo "$f not found (see datasets/README.md)"; exit 1; }
done

for i in $(seq 0 $((N_SEEDS - 1))); do
  seed="${SEEDS[$i]}"

  echo "--- [$KG seed=$seed] training SparqlCE (batch $BATCH, <=$EPOCHS epochs) ---"
  python scripts/train_sparqlce.py --kg "$KG" --seed "$seed" --ablation full \
    --patience "$PATIENCE" --epochs "$EPOCHS" --batch-size "$BATCH" --amp

  echo "--- [$KG seed=$seed] training E5-Finetuned baseline (batch $BATCH, <=$EPOCHS epochs) ---"
  python scripts/finetune_e5_baseline.py --kg "$KG" --seed "$seed" \
    --patience "$PATIENCE" --epochs "$EPOCHS" --batch-size "$BATCH" --grad-accum 1 --amp

  echo "--- [$KG seed=$seed] exporting per-anchor scores ---"
  python scripts/eval_rq1_export.py --kg "$KG" --split "$SPLIT" \
    --model-type sparqlce \
    --checkpoint "checkpoints/$KG/seed${seed}_full/best_model.pt" \
    --seed "$seed" --ablation full \
    --out "results/significance/${TAG}_sparqlce_seed${seed}.json"

  # E5-FT is a full fine-tune saved as a standalone HF/sentence-transformers
  # directory, so --model-type sentence-transformer is correct here. (The
  # peft-qwen3 path is only for bare LoRA adapter dirs and is now unused.)
  python scripts/eval_rq1_export.py --kg "$KG" --split "$SPLIT" \
    --model-type sentence-transformer \
    --model-name "checkpoints/e5_ft/$KG/seed${seed}/best_model" \
    --seed "$seed" --ablation baseline-e5ft \
    --out "results/significance/${TAG}_e5ft_seed${seed}.json"
done

echo "=== aggregating + paired significance test ==="
python scripts/compute_significance.py \
  --sparqlce-glob "results/significance/${TAG}_sparqlce_seed*.json" \
  --baseline-glob "results/significance/${TAG}_e5ft_seed*.json" \
  --out "results/significance/${TAG}_summary.json"

echo "Done -> results/significance/${TAG}_summary.json"
