#!/usr/bin/env bash
# DC-AE v9 + U-Net-style gated zero-init skip connections.
set -eo pipefail
cd "$(dirname "$0")/.."

CLEAN_EXTRA="--no-physical-scales-loss --use-aurora-weights"

CUDA_VISIBLE_DEVICES=1 \
DEVICES=1 BATCH=2 WORKERS=8 EPOCHS=8 \
RUN_NAME=exp_dcae_v9_skip \
MODEL_TYPE=dcae_adaln_skip_residual_linear \
EXTRA_ARGS="--latent-channels 32 $CLEAN_EXTRA" \
ENSEMBLE_PASSES=1 \
bash scripts/run_exp.sh > logs/runner/launcher_dcae_skip_v9.log 2>&1 &
P0=$!
echo "DC-AE v9 skip launched: PID=$P0 on GPU 1"

wait $P0
echo "=== DC-AE v9 skip DONE ==="
date
