#!/usr/bin/env bash
# Fatal #1 fix: train N seeds per KG for the full model, eval each, and
# aggregate mean +/- std + significance vs. the strongest baseline.
#
# Usage: ./scripts/run_seed_sweep.sh [n_seeds] [baseline_json_dbpedia] [baseline_json_wikidata]
#
# Baseline JSONs must already exist -- generate them once with:
#   python scripts/eval_rq1_export.py --kg dbpedia --model-type sentence-transformer \
#       --model-name <qwen3-finetuned-path> --ablation baseline-qwen3ft \
#       --out results/significance/dbpedia_baseline_qwen3ft.json
# (or point --checkpoint at the fine-tuned Qwen3 export path if it isn't served
# through sentence-transformers -- adjust the baseline eval call accordingly.)
set -euo pipefail

N_SEEDS="${1:-5}"
BASELINE_DBPEDIA="${2:-results/significance/dbpedia_baseline_qwen3ft.json}"
BASELINE_WIKIDATA="${3:-results/significance/wikidata_baseline_qwen3ft.json}"
SEEDS=(42 43 44 45 46 47 48 49 50 51)

cd "$(dirname "$0")/.."

for kg in dbpedia wikidata; do
  for i in $(seq 0 $((N_SEEDS - 1))); do
    seed="${SEEDS[$i]}"
    echo "=== training $kg seed=$seed ablation=full ==="
    python scripts/train_sparqlce.py --kg "$kg" --seed "$seed" --ablation full

    echo "=== evaluating $kg seed=$seed ==="
    python scripts/eval_rq1_export.py --kg "$kg" \
      --checkpoint "checkpoints/$kg/seed${seed}_full/best_model.pt" \
      --seed "$seed" --ablation full \
      --out "results/significance/${kg}_seed${seed}_full.json"
  done
done

echo "=== aggregating + significance test ==="
python scripts/compute_significance.py \
  --sparqlce-glob "results/significance/dbpedia_seed*_full.json" \
  --baseline "$BASELINE_DBPEDIA" \
  --out results/significance/dbpedia_full_summary.json

python scripts/compute_significance.py \
  --sparqlce-glob "results/significance/wikidata_seed*_full.json" \
  --baseline "$BASELINE_WIKIDATA" \
  --out results/significance/wikidata_full_summary.json

echo "Done. See results/significance/{dbpedia,wikidata}_full_summary.json"
