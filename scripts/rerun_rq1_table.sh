#!/usr/bin/env bash
# Regenerate EVERY row of an RQ1 results table against ONE split, in one run.
#
# Why this exists: tables/tab-RQ1-Wikidata.tex mixed measurements taken on
# different splits -- the zero-shot rows on the older 462-anchor/1968-query
# corpus, the trained rows on the leak-free 292-anchor/448-query one. Retrieval
# scores depend strongly on candidate-pool size, so rows measured on different
# pools cannot be compared, and the row on the smaller pool looks better for a
# reason that has nothing to do with the model. Every row in a table must come
# from one split; this script is how that is guaranteed.
#
# No training happens here. The zero-shot baselines are inference-only, and the
# two supervised rows load checkpoints trained earlier by run_seed_sweep.sh, so
# this is minutes on one GPU, not hours.
#
# Usage:
#   ./scripts/rerun_rq1_table.sh <kg> <split_json> [seed]
#   ./scripts/rerun_rq1_table.sh wikidata datasets/wikidata-testsplit-RQ1.json 42
#
# Output: one per-bin block per model on stdout (MRR@k / Hits@k / nDCG@k for
# k in 1,3,5,10). The "ALL" row of each block is the whole-split number that
# goes in the paper table. Tee it to a file and transcribe from that.
set -euo pipefail
cd "$(dirname "$0")/.."

KG="${1:?usage: rerun_rq1_table.sh <kg> <split_json> [seed]}"
SPLIT="${2:?usage: rerun_rq1_table.sh <kg> <split_json> [seed]}"
SEED="${3:-42}"

[ -f "$SPLIT" ] || { echo "split $SPLIT not found"; exit 1; }

SPARQLCE_CKPT="checkpoints/$KG/seed${SEED}_full/best_model.pt"
E5FT_DIR="checkpoints/e5_ft/$KG/seed${SEED}/best_model"
for p in "$SPARQLCE_CKPT" "$E5FT_DIR"; do
  [ -e "$p" ] || { echo "$p not found -- train seed $SEED first (run_seed_sweep.sh)"; exit 1; }
done

# Confirm the split is leak-free before spending anything on it, and print the
# anchor/corpus counts that MUST appear in the table caption: a reader cannot
# interpret an MRR without knowing how many candidates it ranked against.
python scripts/check_leakage.py --kg "$KG" --split "$SPLIT" || {
  echo "!!! $SPLIT still contains trained-on pairs -- rebuild it with"
  echo "    build_clean_rq1.py --leak-filter pair before regenerating the table."
  exit 1
}

echo ""
echo "=== regenerating every RQ1 row on $SPLIT (kg=$KG, seed=$SEED) ==="
echo ""

# One invocation, one corpus, one set of eval items -- so every row below is
# computed against the identical candidate pool by construction.
#
# Text baselines receive Wikidata label expansion (applied inside
# eval_rq1_stratified.py) and, for E5, the "query: " prefix, matching what
# SparqlCE's graph builder resolves internally. Without both, the text rows are
# handicapped by preprocessing rather than by capacity.
python scripts/eval_rq1_stratified.py \
  --split "$SPLIT" \
  --st intfloat/e5-base-v2 \
  --st "$E5FT_DIR" \
  --st sentence-transformers/all-MiniLM-L6-v2 \
  --st sentence-transformers/all-mpnet-base-v2 \
  --st microsoft/codebert-base \
  --st microsoft/graphcodebert-base \
  --st razent/spbert-mlm-wso-base \
  --code-llm Qwen/Qwen2.5-Coder-0.5B \
  --sparqlce "$SPARQLCE_CKPT"

echo ""
echo "=== done. Transcribe the ALL row of each block into the paper table. ==="
echo "Every row above shares one split, one corpus and one anchor set."
echo "Put the anchor count and pool size in the caption."
