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
#   ./scripts/run_seed_sweep.sh <kg> [n_seeds] [split_json] [patience]
#
#   ./scripts/run_seed_sweep.sh wikidata 5 datasets/wikidata_testset_RQ1_clean.json 3
#   ./scripts/run_seed_sweep.sh dbpedia  5 datasets/dbpedia_testset_RQ1_clean.json  3
#
# Prerequisite: SparqlCE must be retrained AFTER the graph_builder.py fixes --
# old checkpoints were trained on the pre-fix representation. This script
# retrains from scratch, so running it satisfies that.
set -euo pipefail

KG="${1:?usage: run_seed_sweep.sh <kg> [n_seeds] [split_json] [patience]}"
N_SEEDS="${2:-5}"
SPLIT="${3:-datasets/${KG}_testset_RQ1_clean.json}"
PATIENCE="${4:-3}"
SEEDS=(42 43 44 45 46 47 48 49 50 51)

cd "$(dirname "$0")/.."
mkdir -p results/significance

TAG="$(basename "$SPLIT" .json)"
echo "=== KG=$KG  seeds=$N_SEEDS  split=$SPLIT  patience=$PATIENCE ==="
[ -f "$SPLIT" ] || { echo "split $SPLIT not found -- build it with scripts/build_clean_rq1.py --kg $KG"; exit 1; }

for i in $(seq 0 $((N_SEEDS - 1))); do
  seed="${SEEDS[$i]}"

  echo "--- [$KG seed=$seed] training SparqlCE ---"
  python scripts/train_sparqlce.py --kg "$KG" --seed "$seed" --ablation full \
    --patience "$PATIENCE"

  echo "--- [$KG seed=$seed] training E5-Finetuned baseline ---"
  python scripts/finetune_e5_baseline.py --kg "$KG" --seed "$seed" \
    --patience "$PATIENCE"

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
