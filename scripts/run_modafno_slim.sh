#!/usr/bin/env bash
# ModAFNO Slim: compact (~12M params) + big BATCH + extended training (12 epochs).
# Compress: embed_dim 512→256, depth 12→8. Faster convergence vs 127M v9 plateau.
set -eo pipefail
cd "$(dirname "$0")/.."

CLEAN_EXTRA="--no-physical-scales-loss --use-aurora-weights"

CUDA_VISIBLE_DEVICES=0 \
DEVICES=1 BATCH=8 WORKERS=8 EPOCHS=12 \
RUN_NAME=exp_modafno_v9_slim \
MODEL_TYPE=modafno_official_residual_linear \
EXTRA_ARGS="--latent-channels 256 --modafno-depth 8 --modafno-num-blocks 8 --modafno-drop-rate 0.0 $CLEAN_EXTRA" \
ENSEMBLE_PASSES=1 \
bash scripts/run_exp.sh > logs/runner/launcher_modafno_slim.log 2>&1 &
P0=$!
echo "ModAFNO Slim launched: PID=$P0 on GPU 0"

wait $P0
echo "=== ModAFNO Slim DONE ==="
date
