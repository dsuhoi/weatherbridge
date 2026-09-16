#!/usr/bin/env bash
# S-DYff DYffusion (Spherical FNO) on 0.5° data.
set -eo pipefail
cd "$(dirname "$0")/.."

CLEAN_EXTRA="--no-physical-scales-loss --use-aurora-weights"

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1} \
DEVICES=1 BATCH=1 WORKERS=4 EPOCHS=8 \
TRAIN_YEARS=${TRAIN_YEARS:-"2018"} \
CACHE_IN_RAM=${CACHE_IN_RAM:-1} \
RUN_NAME=${RUN_NAME:-exp_sdyff_0p5_dyffusion} \
MODEL_TYPE=sdyff_dyffusion_residual_linear \
EXTRA_ARGS="--latent-channels 96 --sdyff-num-layers 6 --sdyff-n-modes-lat 16 --sdyff-n-modes-lon 32 --sdyff-dropout 0.1 --sdyff-drop-path 0.1 --sdyff-inference-steps 5 --sdyff-train-refine-steps 1 --precision 32 $CLEAN_EXTRA" \
ENSEMBLE_PASSES=1 \
bash scripts/run_exp_0p5.sh > logs/runner/launcher_sdyff_0p5.log 2>&1 &
P0=$!
echo "S-DYff 0.5° launched: PID=$P0 GPU=$CUDA_VISIBLE_DEVICES"
wait $P0
echo "=== S-DYff 0.5° DONE ==="
date
