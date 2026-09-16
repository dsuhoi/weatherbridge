#!/usr/bin/env bash
# FuXi SwinV2 v9 DIRECT mode — ablation paired with ModAFNO direct.
set -eo pipefail
cd "$(dirname "$0")/.."

CLEAN_EXTRA="--no-physical-scales-loss --use-aurora-weights --direct-prediction"

CUDA_VISIBLE_DEVICES=1 \
DEVICES=1 BATCH=4 WORKERS=8 EPOCHS=8 \
RUN_NAME=exp_fuxi_swinv2_v9_direct \
MODEL_TYPE=fuxi_swinv2_residual_linear \
USE_KITCHEN_SINK=0 \
EXTRA_ARGS="--latent-channels 256 --fuxi-depth 8 --fuxi-num-heads 8 --fuxi-window-size-h 5 --fuxi-window-size-w 9 --fuxi-patch-size 4 --fuxi-drop-path 0.1 $CLEAN_EXTRA" \
ENSEMBLE_PASSES=1 \
bash scripts/run_exp.sh > logs/runner/launcher_fuxi_direct.log 2>&1 &
P0=$!
echo "FuXi v9 DIRECT launched: PID=$P0 on GPU 1"

wait $P0
echo "=== FuXi v9 DIRECT DONE ==="
date
