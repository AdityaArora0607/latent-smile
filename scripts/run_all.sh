#!/usr/bin/env bash
# Full pipeline: attribute evaluator, main models, smile discovery for the
# baseline runs, ablations, and final evaluation on the test split.
#
# Run from the project root (the script also cds there). Training commands use
# --resume, so re-running after an interruption continues from last.pt.
#
# Environment overrides:
#   PY=.venv/bin/python    Python interpreter
#   DATA=data/celeba_64    prepared dataset directory
#   SEEDS="0 1 2"          seeds for baseline, supervised_noadv and supervised
#   SUBSET=N               pass --subset N to training (quick runs)
#   EPOCHS=E               pass --epochs E to training (quick runs)
#   RUNS_DIR=runs          where run directories are written
#   RESULTS=results        where metrics, figures and grids are written
#   MAX_IMAGES=N           evaluate on the first N test images only (quick runs)
#   SKIP_ABLATIONS=1       skip the beta-VAE and latent-size ablations
set -euo pipefail

cd "$(dirname "$0")/.."

PY=${PY:-.venv/bin/python}
DATA=${DATA:-data/celeba_64}
SEEDS=${SEEDS:-"0 1 2"}
SKIP_ABLATIONS=${SKIP_ABLATIONS:-0}
RUNS_DIR=${RUNS_DIR:-runs}
RESULTS=${RESULTS:-results}
EVAL_EXTRA=()
if [ -n "${MAX_IMAGES:-}" ]; then EVAL_EXTRA+=(--max-images "$MAX_IMAGES"); fi

# Quick-run overrides, applied to training and evaluator training only.
# The ${arr[@]+...} form keeps an empty array safe under set -u on bash 3.2.
QUICK=()
if [ -n "${SUBSET:-}" ]; then QUICK+=(--subset "$SUBSET"); fi
if [ -n "${EPOCHS:-}" ]; then QUICK+=(--epochs "$EPOCHS"); fi

header() {
  echo
  echo "=== $* ==="
}

# train <config> <seed>   -> $RUNS_DIR/<name>/seed<seed>
train() {
  local config=$1 seed=$2
  header "train $config seed $seed"
  "$PY" -m src.train --config "$config" --seed "$seed" --data-root "$DATA" --out-dir "$RUNS_DIR" --resume \
    ${QUICK[@]+"${QUICK[@]}"}
}

# discover <run dir>   -> <run dir>/directions.pt
discover() {
  local run=$1
  header "discovery $run"
  "$PY" -m src.discover_smile --run "$run" --data-root "$DATA" \
    --evaluator "$RUNS_DIR/evaluator/attr.pt" --out "$RESULTS"
}

# 1. Evaluator: plain VAE used only to produce reconstructions, then the
#    attribute evaluator trained on real images plus those reconstructions.
train configs/evaluator_vae.yaml 100

header "attribute evaluator"
"$PY" -m src.evaluators --data-root "$DATA" \
  --recon-run "$RUNS_DIR/evaluator_vae/seed100" --out "$RUNS_DIR/evaluator" --results "$RESULTS" \
  ${QUICK[@]+"${QUICK[@]}"}

# 2. Main models, every seed.
for seed in $SEEDS; do
  train configs/baseline.yaml "$seed"
  train configs/supervised_noadv.yaml "$seed"
  train configs/supervised.yaml "$seed"
done

# 3. Smile discovery for each baseline (B) run.
for seed in $SEEDS; do
  discover "$RUNS_DIR/baseline/seed$seed"
done

# 4. Ablations, seed 0 only.
ABLATION_RUNS=()
if [ "$SKIP_ABLATIONS" != "1" ]; then
  train configs/beta_vae.yaml 0
  train configs/baseline_z32.yaml 0
  train configs/baseline_z128.yaml 0
  train configs/supervised_z32.yaml 0
  train configs/supervised_z128.yaml 0

  # Plain VAEs need directions; the supervised ones do not.
  discover $RUNS_DIR/beta_vae/seed0
  discover $RUNS_DIR/baseline_z32/seed0
  discover $RUNS_DIR/baseline_z128/seed0

  ABLATION_RUNS=(
    $RUNS_DIR/beta_vae/seed0
    $RUNS_DIR/baseline_z32/seed0
    $RUNS_DIR/baseline_z128/seed0
    $RUNS_DIR/supervised_z32/seed0
    $RUNS_DIR/supervised_z128/seed0
  )
fi

# 5. Evaluation on the test split, all runs in one call.
RUNS=()
for seed in $SEEDS; do
  RUNS+=(
    "$RUNS_DIR/baseline/seed$seed"
    "$RUNS_DIR/supervised_noadv/seed$seed"
    "$RUNS_DIR/supervised/seed$seed"
  )
done
RUNS+=(${ABLATION_RUNS[@]+"${ABLATION_RUNS[@]}"})

header "evaluate ${#RUNS[@]} runs"
"$PY" -m src.evaluate --runs "${RUNS[@]}" --evaluator "$RUNS_DIR/evaluator/attr.pt" \
  --data-root "$DATA" --out "$RESULTS" ${EVAL_EXTRA[@]+"${EVAL_EXTRA[@]}"}

header "done"
