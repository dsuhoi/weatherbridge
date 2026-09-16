#!/usr/bin/env bash
# FuXi SwinV2 (param-efficient) on 0.5° data.
set -eo pipefail
cd "$(dirname "$0")/.."

CLEAN_EXTRA="--no-physical-scales-loss --use-aurora-weights"

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} \
DEVICES=1 BATCH=${BATCH:-2} WORKERS=4 EPOCHS=8 \
TRAIN_YEARS=${TRAIN_YEARS:-"2018"} \
CACHE_IN_RAM=${CACHE_IN_RAM:-1} \
RUN_NAME=${RUN_NAME:-exp_fuxi_0p5_swinv2} \
MODEL_TYPE=fuxi_swinv2_residual_linear \
EXTRA_ARGS="--latent-channels 256 --fuxi-depth 8 --fuxi-num-heads 8 --fuxi-window-size-h 5 --fuxi-window-size-w 9 --fuxi-patch-size 4 --fuxi-drop-path 0.1 $CLEAN_EXTRA" \
ENSEMBLE_PASSES=1 \
bash scripts/run_exp_0p5.sh > logs/runner/launcher_fuxi_0p5.log 2>&1 &
P0=$!
echo "FuXi SwinV2 0.5° launched: PID=$P0 on GPU 1"
wait $P0
echo "=== FuXi 0.5° DONE ==="
date
