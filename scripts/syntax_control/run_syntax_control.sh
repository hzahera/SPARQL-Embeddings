#!/usr/bin/env bash
# Syntax-supervised control: build labels, train 3 encoders x 3 seeds x 2 graphs, evaluate.
# Run from anywhere; GPU required for steps 2 and 3.
#   bash scripts/syntax_control/run_syntax_control.sh            # everything
#   STEP=labels bash scripts/syntax_control/run_syntax_control.sh # only step 1
#   STEP=train  ...                                               # only step 2
#   STEP=eval   ...                                               # only step 3
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
export PYTHONHASHSEED=0
MODELS="${MODELS:-e5-base-v2 qwen3-embedding codebert}"
SEEDS="${SEEDS:-42 43 44}"
GRAPHS="${GRAPHS:-dbpedia wikidata}"
STEP="${STEP:-all}"

if [[ "$STEP" == all || "$STEP" == labels ]]; then
  for g in $GRAPHS; do
    echo "== [1/3] syntax labels: $g"
    python scripts/syntax_control/build_syntax_labels.py --graph "$g"
  done
fi

if [[ "$STEP" == all || "$STEP" == train ]]; then
  echo "== [2/3] training (resumable; existing checkpoints are skipped)"
  python scripts/run_experiment.py \
    --config configs/experiment_syntax_control.yaml \
    --models $MODELS --datasets $GRAPHS --seeds $SEEDS \
    --skip-zero-shot --skip-cache
fi

if [[ "$STEP" == all || "$STEP" == eval ]]; then
  echo "== [3/3] evaluation against answer-label checkpoints in results/"
  python scripts/syntax_control/eval_syntax_control.py \
    --models $MODELS --seeds $SEEDS --graphs $GRAPHS
fi
