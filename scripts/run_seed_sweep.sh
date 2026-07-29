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
#   ./scripts/run_seed_sweep.sh wikidata 5 datasets/wikidata-testsplit-RQ1.json 5 128 60
#   ./scripts/run_seed_sweep.sh dbpedia  5 datasets/dbpedia-testsplit-RQ1.json  5 128 60
#
# The committed {kg}-testsplit-RQ1.json files are the canonical eval splits and
# are already leak-free. Any split passed here must be: a group-level train/test
# partition still shares anchor-positive PAIRS with training (84.5% of them, in
# the build these replaced), which inflates both models unequally and turns the
# headline gap into a memorization measurement. This script audits the split and
# refuses a leaky one unless ALLOW_LEAKY_SPLIT=1. See scripts/check_leakage.py.
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
SPLIT="${3:-datasets/${KG}-testsplit-RQ1.json}"
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
# SEEDS="43 44 45 46" in the ENVIRONMENT overrides the default list, to redo
# exactly the seeds that failed without retraining the ones already banked.
# When it is set and no explicit seed COUNT was passed, the count follows the
# override. The significance step globs whatever score files exist for this
# split tag, so seeds banked by an earlier partial run are picked up too.
if [ -n "${SEEDS:-}" ]; then
  read -r -a SEEDS <<< "$SEEDS"
  [ -z "${2:-}" ] && N_SEEDS="${#SEEDS[@]}"
  echo "SEEDS override: running seeds ${SEEDS[*]} (n=$N_SEEDS)"
else
  SEEDS=(42 43 44 45 46 47 48 49 50 51)
fi
if [ "$N_SEEDS" -gt "${#SEEDS[@]}" ]; then
  echo "requested $N_SEEDS seeds but only ${#SEEDS[@]} are defined: ${SEEDS[*]}"
  exit 1
fi

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

# Pre-flight: disk. Each seed writes a SparqlCE checkpoint plus a full E5-FT HF
# directory (~0.4GB), and a mid-training ENOSPC loses that seed's whole run --
# on the first noleak sweep seeds 43-46 died this way, one after another, each
# costing a full training. Checking up front turns hours of wasted GPU into an
# immediate error. MIN_FREE_GB=0 disables.
MIN_FREE_GB="${MIN_FREE_GB:-15}"
if [ "$MIN_FREE_GB" != "0" ]; then
  mkdir -p checkpoints
  FREE_GB=$(df -Pk checkpoints | awk 'NR==2 {printf "%d", $4/1024/1024}')
  echo "disk: ${FREE_GB}GB free on the checkpoints filesystem (need >=${MIN_FREE_GB}GB)"
  if [ "$FREE_GB" -lt "$MIN_FREE_GB" ]; then
    echo "!!! not enough free space for $N_SEEDS seeds."
    echo "    Biggest reclaimable items are usually:"
    echo "      du -sh ~/.cache/huggingface checkpoints results   # look here first"
    echo "      rm -rf ~/.cache/huggingface/hub                   # re-downloads on demand"
    echo "      rm -rf checkpoints/e5_ft/$KG/seed*/best_model     # ~0.4GB each, retrainable"
    echo "    Then rerun. Set MIN_FREE_GB=0 to skip this check."
    exit 1
  fi
fi

# Per-split files (the component protocol, scripts/build_rq1_twoaxis.py). When
# they exist they are the ONLY correct thing to train on: the integer-index path
# addresses splits_${KG}.json, whose training anchors sit inside the new test
# components. Training there while evaluating on $SPLIT would produce a large,
# entirely spurious result -- so this is detected rather than left to the caller.
TRAIN_FILE="${TRAIN_FILE:-datasets/${KG}-train-RQ1.json}"
VAL_FILE="${VAL_FILE:-datasets/${KG}-valid-RQ1.json}"
COMPONENT_SPLITS="datasets/splits_${KG}_components.json"
PER_SPLIT=0
if [ -f "$TRAIN_FILE" ] && [ -f "$VAL_FILE" ]; then
  PER_SPLIT=1
  SPARQLCE_DATA_ARGS=(--train-file "$TRAIN_FILE" --val-file "$VAL_FILE" --test-file "$SPLIT")
  E5_DATA_ARGS=(--train-file "$TRAIN_FILE" --val-file "$VAL_FILE")
  echo "per-split mode: train=$TRAIN_FILE val=$VAL_FILE (both models)"
  # NEGATIVE PARITY. finetune_e5_baseline.py always puts one curated negative per
  # anchor in its InfoNCE candidate set (SquareTextDataset.__getitem__ ->
  # info_nce_loss), but train_sparqlce.py ignores negative_queries unless
  # --hard-negatives is passed. With the old corpus that gap did not matter: the
  # 24 shared negatives were lexically unrelated to every anchor. The component
  # files carry ~16 same-template, result-disjoint negatives per anchor, so
  # leaving it off trains the BASELINE against precisely the signal SparqlCE
  # exists to capture, and SparqlCE without it. Default it on here; set
  # HARD_NEGS=0 to measure the in-batch-only variant deliberately.
  HARD_NEGS="${HARD_NEGS:-1}"
  MAX_NEG="${MAX_NEG:-16}"
  if [ "$HARD_NEGS" != "0" ]; then
    SPARQLCE_DATA_ARGS+=(--hard-negatives --max-neg "$MAX_NEG")
    echo "  negative parity: SparqlCE --hard-negatives --max-neg $MAX_NEG"
  else
    echo "  HARD_NEGS=0: SparqlCE trains on in-batch negatives only, while the"
    echo "  baseline still uses one curated negative per anchor. NOT a like-for-like"
    echo "  comparison -- use only to isolate the hard-negative contribution."
  fi
elif [ -f "$COMPONENT_SPLITS" ]; then
  echo "!!! $COMPONENT_SPLITS exists but $TRAIN_FILE / $VAL_FILE do not."
  echo "    This KG uses the component protocol; training from the integer-index"
  echo "    split would train on queries that are in $SPLIT. Build them:"
  echo "      for s in train valid; do"
  echo "        python scripts/build_rq1_twoaxis.py --kg $KG --from-split \$s --neg-max 0.2 \\"
  echo "          --splits $COMPONENT_SPLITS \\"
  echo "          --result-sets datasets/result_sets_${KG}_v2.json \\"
  echo "          --out datasets/${KG}-\$s-RQ1.json"
  echo "      done"
  exit 1
else
  SPARQLCE_DATA_ARGS=()
  E5_DATA_ARGS=()
  echo "index-split mode: both models read datasets/splits_${KG}.json"
fi

# Pre-flight: refuse to spend GPU hours producing a number measured on eval
# pairs the models were trained on. Cheap (seconds) relative to the sweep.
#
# check_leakage.py compares against the GROUP split (splits_${KG}.json), which is
# not what a component run trains on -- it would report leakage that the run
# cannot have. build_rq1_twoaxis.py already asserts zero memorized pairs against
# the component partition at build time, and refuses to write otherwise, so the
# guarantee is made there instead.
if [ "$PER_SPLIT" = "1" ]; then
  echo "component protocol: leak-freeness asserted at build time by"
  echo "  build_rq1_twoaxis.py (0 memorized pairs); skipping the group-split audit."
elif [ "${ALLOW_LEAKY_SPLIT:-0}" != "1" ]; then
  python scripts/check_leakage.py --kg "$KG" --split "$SPLIT"; leak_rc=$?
  if [ $leak_rc -eq 1 ]; then
    echo "!!! leakage audit could not run (see error above) -- fix it rather than"
    echo "    skipping: an unaudited split is how the 84.5%-leaked sweep happened."
    exit 1
  elif [ $leak_rc -ne 0 ]; then
    echo ""
    echo "!!! $SPLIT contains anchor-positive pairs from the training groups."
    echo "    Build a leak-free split first:"
    echo "      python scripts/build_clean_rq1.py --kg $KG --min-result-size 0 \\"
    echo "        --out datasets/${KG}-testsplit-RQ1.json"
    echo "    then rerun against it. Set ALLOW_LEAKY_SPLIT=1 to"
    echo "    override (only to MEASURE the leakage effect, never for a headline"
    echo "    number)."
    exit 1
  fi
fi

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
      --patience "$PATIENCE" --epochs "$EPOCHS" --batch-size "$BATCH" --amp \
      ${SPARQLCE_DATA_ARGS[@]+"${SPARQLCE_DATA_ARGS[@]}"}
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
  E5_META="checkpoints/e5_ft/$KG/seed${seed}/run_metadata.json"
  rm -f "$E5_META"   # so the LR check below can never pass on a stale file
  python scripts/finetune_e5_baseline.py --kg "$KG" --seed "$seed" \
    --patience "$PATIENCE" --epochs "$EPOCHS" --batch-size "$BATCH" --grad-accum 1 --amp \
    ${E5_LR:+--lr "$E5_LR"} ${E5_DATA_ARGS[@]+"${E5_DATA_ARGS[@]}"} \
  && python scripts/eval_rq1_export.py --kg "$KG" --split "$SPLIT" \
    --model-type sentence-transformer \
    --model-name "checkpoints/e5_ft/$KG/seed${seed}/best_model" \
    --seed "$seed" --ablation baseline-e5ft \
    --out "results/significance/${TAG}_e5ft_seed${seed}.json"
  [ $? -ne 0 ] && { echo "!!! [$KG seed=$seed] E5-FT FAILED -- SparqlCE safe; rerun E5-FT later"; FAILED_E5FT+=("$seed"); }

  # Verify the baseline was actually trained at the batch-scaled LR. A GPU box
  # running a checkout from before the LR-scaling fix silently trains E5-FT at
  # 1e-5 on batch 128 and lands at ~0.480 aggregate MRR@10 instead of ~0.590 --
  # a 0.11 MRR handicap that looks exactly like a real result. This is a
  # comparison-invalidating bug, so it is FATAL rather than a warning.
  if [ -z "$E5_LR" ] && [ -f "$E5_META" ]; then
    python - "$E5_META" "$BATCH" "$seed" <<'PYCHECK' || { echo "!!! [$KG seed=$seed] aborting: baseline LR is wrong, any comparison from this run is invalid"; exit 1; }
import json, sys
meta_path, batch, seed = sys.argv[1], int(sys.argv[2]), sys.argv[3]
m = json.load(open(meta_path))
got, want = float(m.get("lr", 0)), 1e-5 * batch / 32.0
if abs(got - want) > 0.05 * want:
    print(f"!!! seed={seed} E5-FT trained at lr={got:.2e}, expected {want:.2e} "
          f"(1e-5 scaled from effective batch 32 to {batch}).")
    print("    The GPU box is running finetune_e5_baseline.py from before commit "
          "217fb8b -- git pull there and rerun the baseline.")
    sys.exit(1)
print(f"[lr-check] seed={seed} E5-FT lr={got:.2e} (correct for batch {batch})")
PYCHECK
  fi
done
set -e

# A SparqlCE failure used to only WARN here. But the loop `continue`s past
# E5-FT for that seed, so FAILED_E5FT stays empty and the E5-FT gate below never
# fires -- a run where 4 of 5 seeds died still reached the significance step and
# would have written a summary computed from the one surviving seed, labelled
# n_seeds=1 in a file named like the 5-seed one. Fail loudly instead.
if [ ${#FAILED_SPARQLCE[@]} -ne 0 ]; then
  echo "=== SparqlCE training FAILED for seeds: ${FAILED_SPARQLCE[*]} -- skipping significance ==="
  echo "E5-FT was skipped for those seeds too (the loop moves to the next seed),"
  echo "so BOTH sides are incomplete. Fix the cause, then rerun just those seeds:"
  echo "  SEEDS=\"${FAILED_SPARQLCE[*]}\" ./scripts/run_seed_sweep.sh $KG ${#FAILED_SPARQLCE[@]} $SPLIT $PATIENCE $BATCH $EPOCHS"
  echo "and run scripts/compute_significance.py once every seed is present."
  exit 1
fi

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
# Seeds that must ALL be present before a summary is written. With a SEEDS
# override (resuming failed seeds) this run only covers part of the intended
# set, so pass the full one: EXPECT_SEEDS="42,43,44,45,46".
ALL_SEEDS="${EXPECT_SEEDS:-$(IFS=,; echo "${SEEDS[*]:0:$N_SEEDS}")}"
python scripts/compute_significance.py \
  --sparqlce-glob "results/significance/${TAG}_sparqlce_seed*.json" \
  --baseline-glob "results/significance/${TAG}_e5ft_seed*.json" \
  --expect-seeds "$ALL_SEEDS" \
  --out "results/significance/${TAG}_summary.json"

echo "Done -> results/significance/${TAG}_summary.json"
