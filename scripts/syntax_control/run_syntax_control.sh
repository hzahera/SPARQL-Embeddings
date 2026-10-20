#!/usr/bin/env bash
# Syntax-supervised control, fast plan (1-2 days): 2 encoders x 2 seeds, DBpedia first.
#
#   STEP=check  bash scripts/syntax_control/run_syntax_control.sh   # best epochs of the main runs
#   GRAPHS=dbpedia  bash scripts/syntax_control/run_syntax_control.sh   # labels + train + eval
#
# Steps: check | labels | train | eval | all (default: labels+train+eval)
# Overrides (environment variables):
#   MODELS      default "e5-base-v2 codebert"
#   SEEDS       default "42 43"
#   GRAPHS      default "dbpedia wikidata"
#   EPOCHS      default: value in configs/experiment_syntax_control.yaml (20)
#   EVAL_EVERY  default: value in the config (2)
# Run graphs on separate GPUs in parallel, e.g.
#   CUDA_VISIBLE_DEVICES=0 GRAPHS=dbpedia  bash scripts/syntax_control/run_syntax_control.sh
#   CUDA_VISIBLE_DEVICES=1 GRAPHS=wikidata bash scripts/syntax_control/run_syntax_control.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
export PYTHONHASHSEED=0
MODELS="${MODELS:-e5-base-v2 codebert}"
SEEDS="${SEEDS:-42 43}"
GRAPHS="${GRAPHS:-dbpedia wikidata}"
STEP="${STEP:-all}"
BASE_CFG="configs/experiment_syntax_control.yaml"

if [[ "$STEP" == check ]]; then
  echo "== best epoch of the main answer-label runs (results/)"
  python - "$MODELS" "$GRAPHS" <<'EOF'
import json, sys
from pathlib import Path
models, graphs = sys.argv[1].split(), sys.argv[2].split()
for g in graphs:
    for m in models:
        eps = []
        for f in sorted(Path(f"results/raw/{g}/{m}/tuned").glob("seed_*/training_summary.json")):
            s = json.load(open(f))
            eps.append(s.get("best_epoch") or s.get("best", {}).get("epoch"))
        print(f"{g:9s} {m:16s} best epochs: {eps}")
print("If all best epochs are <= 10, run with EPOCHS=10 (fair: the answer runs picked the same checkpoints).")
EOF
  exit 0
fi

# Optional epoch / eval overrides: write a run-specific copy of the config.
CFG="$BASE_CFG"
if [[ -n "${EPOCHS:-}" || -n "${EVAL_EVERY:-}" ]]; then
  CFG="configs/experiment_syntax_control.run.yaml"
  cp "$BASE_CFG" "$CFG"
  [[ -n "${EPOCHS:-}" ]] && sed -i.bak "s/^  epochs: .*/  epochs: ${EPOCHS}/" "$CFG"
  [[ -n "${EVAL_EVERY:-}" ]] && sed -i.bak "s/^  eval_every_epochs: .*/  eval_every_epochs: ${EVAL_EVERY}/" "$CFG"
  rm -f "$CFG.bak"
  echo "== using $CFG (epochs=${EPOCHS:-default}, eval_every=${EVAL_EVERY:-default})"
fi

if [[ "$STEP" == all || "$STEP" == labels ]]; then
  for g in $GRAPHS; do
    if [[ -f "datasets/${g}-splits/${g}-train-syntax.json" ]]; then
      echo "== [1/3] syntax labels: $g (exists, skipping)"
    else
      echo "== [1/3] syntax labels: $g"
      python scripts/syntax_control/build_syntax_labels.py --graph "$g"
    fi
  done
fi

if [[ "$STEP" == all || "$STEP" == train ]]; then
  echo "== [2/3] training: $MODELS | seeds $SEEDS | $GRAPHS (resumable)"
  python scripts/run_experiment.py \
    --config "$CFG" \
    --models $MODELS --datasets $GRAPHS --seeds $SEEDS \
    --skip-zero-shot --skip-cache
fi

if [[ "$STEP" == all || "$STEP" == eval ]]; then
  echo "== [3/3] evaluation against answer-label checkpoints in results/"
  python scripts/syntax_control/eval_syntax_control.py \
    --models $MODELS --seeds $SEEDS --graphs $GRAPHS
fi
