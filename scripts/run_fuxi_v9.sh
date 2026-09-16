#!/usr/bin/env bash
# FuXi-style SwinV2 baseline с v9-конфигом.
# ~8M params, FiLM-кондишн (paper-faithful AdaLN-Zero), бутерброд с bilinear+residual.
set -eo pipefail
cd "$(dirname "$0")/.."

CLEAN_EXTRA="--no-physical-scales-loss --use-aurora-weights"

CUDA_VISIBLE_DEVICES=1 \
DEVICES=1 BATCH=4 WORKERS=8 EPOCHS=8 \
RUN_NAME=exp_fuxi_swinv2_v9 \
MODEL_TYPE=fuxi_swinv2_residual_linear \
EXTRA_ARGS="--latent-channels 256 --fuxi-depth 8 --fuxi-num-heads 8 --fuxi-window-size-h 5 --fuxi-window-size-w 9 --fuxi-patch-size 4 --fuxi-drop-path 0.1 $CLEAN_EXTRA" \
ENSEMBLE_PASSES=1 \
bash scripts/run_exp.sh > logs/runner/launcher_fuxi_v9.log 2>&1 &
P0=$!
echo "FuXi SwinV2 v9 launched: PID=$P0 on GPU 1"

wait $P0
echo "=== FuXi SwinV2 v9 DONE ==="
date
