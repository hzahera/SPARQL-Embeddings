#!/usr/bin/env bash
# Non-neural baselines (protocols A/B/C). Run from the repository root.
# Usage: bash scripts/baselines/run_baselines.sh [dbpedia] [wikidata]
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
GRAPHS=("$@"); [ ${#GRAPHS[@]} -eq 0 ] && GRAPHS=(dbpedia wikidata)
export PYTHONHASHSEED=0
mkdir -p "$ROOT/logs"
for g in "${GRAPHS[@]}"; do
  if [ ! -f "$ROOT/scripts/audit/audit_${g}_perquery.json" ]; then
    echo "audit_${g}_perquery.json missing, building it (normalized_match may differ slightly, see README)"
    (cd "$ROOT/scripts/audit" && python audit_splits.py "$ROOT/datasets/${g}-splits" "$g" "audit_${g}.json" > /dev/null)
  fi
  echo "== $g"
  python "$ROOT/scripts/baselines/nonneural.py" "$ROOT" "$g" "$ROOT/scripts/baselines/results_${g}.json" \
    | tee "$ROOT/logs/baselines_${g}.txt"
done
