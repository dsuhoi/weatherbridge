#!/usr/bin/env bash
# DC-AE Skip Mini: компактная вариация SOTA (latent=16, smaller block_out).
# Уменьшенная архитектура для Pareto-curve (params vs Δ%).
set -eo pipefail
cd "$(dirname "$0")/.."

CLEAN_EXTRA="--no-physical-scales-loss --use-aurora-weights"

CUDA_VISIBLE_DEVICES=0 \
DEVICES=1 BATCH=2 WORKERS=8 EPOCHS=8 \
RUN_NAME=exp_dcae_v9_skip_mini \
MODEL_TYPE=dcae_adaln_skip_residual_linear \
EXTRA_ARGS="--latent-channels 16 --block-out-channels 96 192 384 $CLEAN_EXTRA" \
ENSEMBLE_PASSES=1 \
bash scripts/run_exp.sh > logs/runner/launcher_dcae_skip_mini.log 2>&1 &
P0=$!
echo "DC-AE Skip Mini launched: PID=$P0 on GPU 0"

wait $P0
echo "=== DC-AE Skip Mini DONE ==="
date
