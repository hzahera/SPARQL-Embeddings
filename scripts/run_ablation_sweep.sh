#!/usr/bin/env bash
# Fatal #2 fix: train the no_gnn and no_decomposition ablation variants
# (3 seeds each by default -- enough for mean+-std without the full 5-seed
# cost of the headline runs) and build the ablation table.
#
# Usage: ./scripts/run_ablation_sweep.sh [n_seeds]
set -euo pipefail

N_SEEDS="${1:-3}"
SEEDS=(42 43 44 45 46)

cd "$(dirname "$0")/.."

for kg in dbpedia wikidata; do
  for ablation in no_gnn no_decomposition; do
    for i in $(seq 0 $((N_SEEDS - 1))); do
      seed="${SEEDS[$i]}"
      echo "=== training $kg seed=$seed ablation=$ablation ==="
      python scripts/train_sparqlce.py --kg "$kg" --seed "$seed" --ablation "$ablation"

      echo "=== evaluating $kg seed=$seed ablation=$ablation ==="
      python scripts/eval_rq1_export.py --kg "$kg" \
        --checkpoint "checkpoints/$kg/seed${seed}_${ablation}/best_model.pt" \
        --seed "$seed" --ablation "$ablation" \
        --out "results/significance/${kg}_seed${seed}_${ablation}.json"
    done
  done
done

# full-model results/significance/{kg}_seed*_full.json from run_seed_sweep.sh
# are reused as the "full" row -- no need to retrain it here.
python scripts/build_ablation_table.py \
  --glob "results/significance/{kg}_seed*_{ablation}.json" \
  --kgs dbpedia wikidata \
  --ablations full no_gnn no_decomposition \
  --out tables/tab-ablation.tex

echo "Done. See tables/tab-ablation.tex"
