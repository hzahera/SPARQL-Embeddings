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

# The shared A100s fragment badly across the two back-to-back trainings per seed
# (SparqlCE then E5-FT), leaving GiBs "reserved but unallocated" that a later
# alloc cannot reuse -> spurious OOM even when a few hundred MiB are all that is
# short. expandable_segments lets the allocator hand that memory back. Respect an
# already-set value; otherwise default it on.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

KG="${1:?usage: run_seed_sweep.sh <kg> [n_seeds] [split_json] [patience] [batch] [epochs] [e5_lr]}"
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
# E5-FT learning rate. Empty => finetune_e5_baseline.py linearly scales the
# notebook's 1e-5 (tuned at effective batch 32) to the batch used here. Holding
# 1e-5 fixed while raising the batch to 128 undertrains the baseline badly:
# in the first sweep it dropped E5-FT from 0.590 to 0.480 aggregate MRR@10 and
# from 0.758 to 0.118 in the 0.3-0.5 bin. Never compare against that run.
E5_LR="${7:-}"
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

# SKIP_E5FT=1 trains + exports ONLY SparqlCE for every seed (no E5-FT, no
# significance) -- use it to bank the expensive SparqlCE checkpoints first when
# E5-FT is flaky, then run the full sweep (or an E5-FT-only loop) later.
SKIP_E5FT="${SKIP_E5FT:-0}"
# SKIP_SPARQLCE=1 skips SparqlCE train+export and runs ONLY E5-FT (then
# significance) -- use it to add E5-FT later without retraining the SparqlCE
# models banked by an earlier SKIP_E5FT=1 run. The earlier run's
# ${TAG}_sparqlce_seed*.json score files are reused by the significance step.
SKIP_SPARQLCE="${SKIP_SPARQLCE:-0}"
FAILED_SPARQLCE=()
FAILED_E5FT=()

# Per-seed steps are made non-fatal (set +e) so one seed's E5-FT OOM can never
# discard another seed's already-trained SparqlCE checkpoint. Failures are
# recorded and reported at the end; set -e is restored before the final stage.
set +e
for i in $(seq 0 $((N_SEEDS - 1))); do
  seed="${SEEDS[$i]}"

  if [ "$SKIP_SPARQLCE" != "1" ]; then
    echo "--- [$KG seed=$seed] training SparqlCE (batch $BATCH, <=$EPOCHS epochs) ---"
    python scripts/train_sparqlce.py --kg "$KG" --seed "$seed" --ablation full \
      --patience "$PATIENCE" --epochs "$EPOCHS" --batch-size "$BATCH" --amp
    if [ $? -ne 0 ]; then
      echo "!!! [$KG seed=$seed] SparqlCE training FAILED -- skipping this seed"
      FAILED_SPARQLCE+=("$seed"); continue
    fi

    # Export SparqlCE scores immediately, so the trained model AND its per-anchor
    # scores are banked before the heavier, OOM-prone E5-FT step runs.
    echo "--- [$KG seed=$seed] exporting SparqlCE scores ---"
    python scripts/eval_rq1_export.py --kg "$KG" --split "$SPLIT" \
      --model-type sparqlce \
      --checkpoint "checkpoints/$KG/seed${seed}_full/best_model.pt" \
      --seed "$seed" --ablation full \
      --out "results/significance/${TAG}_sparqlce_seed${seed}.json"
    [ $? -ne 0 ] && echo "!!! [$KG seed=$seed] SparqlCE export failed (checkpoint still saved)"
  else
    echo "--- [$KG seed=$seed] SKIP_SPARQLCE=1: reusing banked SparqlCE scores ---"
    [ -f "results/significance/${TAG}_sparqlce_seed${seed}.json" ] || \
      echo "!!! [$KG seed=$seed] no banked SparqlCE scores -- significance will be incomplete for this seed"
  fi

  if [ "$SKIP_E5FT" = "1" ]; then
    echo "--- [$KG seed=$seed] SKIP_E5FT=1: skipping E5-Finetuned (run it later) ---"
    continue
  fi

  # E5-FT is non-fatal: if it OOMs/fails, every seed's SparqlCE is already safe.
  # E5-FT is a full fine-tune saved as a standalone HF/sentence-transformers dir,
  # so --model-type sentence-transformer is correct here.
  echo "--- [$KG seed=$seed] training + exporting E5-Finetuned (batch $BATCH) ---"
  python scripts/finetune_e5_baseline.py --kg "$KG" --seed "$seed" \
    --patience "$PATIENCE" --epochs "$EPOCHS" --batch-size "$BATCH" --grad-accum 1 --amp \
    ${E5_LR:+--lr "$E5_LR"} \
  && python scripts/eval_rq1_export.py --kg "$KG" --split "$SPLIT" \
    --model-type sentence-transformer \
    --model-name "checkpoints/e5_ft/$KG/seed${seed}/best_model" \
    --seed "$seed" --ablation baseline-e5ft \
    --out "results/significance/${TAG}_e5ft_seed${seed}.json"
  [ $? -ne 0 ] && { echo "!!! [$KG seed=$seed] E5-FT FAILED -- SparqlCE safe; rerun E5-FT later"; FAILED_E5FT+=("$seed"); }
done
set -e

[ ${#FAILED_SPARQLCE[@]} -ne 0 ] && echo "WARN: SparqlCE training failed for seeds: ${FAILED_SPARQLCE[*]}"

if [ "$SKIP_E5FT" = "1" ]; then
  echo "=== Done (SparqlCE only) ==="
  echo "  checkpoints: checkpoints/$KG/seed*_full/best_model.pt"
  echo "  scores:      results/significance/${TAG}_sparqlce_seed*.json"
  echo "Run E5-FT + significance later:"
  echo "  ./scripts/run_seed_sweep.sh $KG $N_SEEDS $SPLIT $PATIENCE $BATCH $EPOCHS"
  exit 0
fi

if [ ${#FAILED_E5FT[@]} -ne 0 ]; then
  echo "=== E5-FT incomplete for seeds: ${FAILED_E5FT[*]} -- skipping significance ==="
  echo "SparqlCE checkpoints + scores are saved for all seeds. Rerun E5-FT for the"
  echo "failed seeds, then run scripts/compute_significance.py to produce the summary."
  exit 1
fi

echo "=== aggregating + paired significance test ==="
python scripts/compute_significance.py \
  --sparqlce-glob "results/significance/${TAG}_sparqlce_seed*.json" \
  --baseline-glob "results/significance/${TAG}_e5ft_seed*.json" \
  --out "results/significance/${TAG}_summary.json"

echo "Done -> results/significance/${TAG}_summary.json"
