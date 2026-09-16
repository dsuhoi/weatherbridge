#!/usr/bin/env bash
# FuXi Nano: ultra-compact SwinV2 (~2M params).
# embed=128, depth=6 — на 1/4 размера FuXi v9.
set -eo pipefail
cd "$(dirname "$0")/.."

CLEAN_EXTRA="--no-physical-scales-loss --use-aurora-weights"

CUDA_VISIBLE_DEVICES=1 \
DEVICES=1 BATCH=6 WORKERS=8 EPOCHS=8 \
RUN_NAME=exp_fuxi_nano_v9 \
MODEL_TYPE=fuxi_swinv2_residual_linear \
EXTRA_ARGS="--latent-channels 128 --fuxi-depth 6 --fuxi-num-heads 4 --fuxi-window-size-h 5 --fuxi-window-size-w 9 --fuxi-patch-size 4 --fuxi-drop-path 0.1 $CLEAN_EXTRA" \
ENSEMBLE_PASSES=1 \
bash scripts/run_exp.sh > logs/runner/launcher_fuxi_nano.log 2>&1 &
P0=$!
echo "FuXi Nano launched: PID=$P0 on GPU 1"

wait $P0
echo "=== FuXi Nano DONE ==="
date
