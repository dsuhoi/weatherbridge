#!/usr/bin/env bash
# DC-AE Skip (SOTA recipe) on 0.5° data.
#
# DEPRECATED (2026-06-12) — prefer Hydra entry point:
#   python train.py +legacy=train_dcae_skip_24ch_6yr
# This script is preserved as a backup invocation for reproducibility of
# the 1° / smoke recipe described below.
set -eo pipefail
cd "$(dirname "$0")/.."

CLEAN_EXTRA="--no-physical-scales-loss --use-aurora-weights"

CUDA_VISIBLE_DEVICES=0 \
DEVICES=1 BATCH=1 WORKERS=4 EPOCHS=8 \
RUN_NAME=exp_dcae_0p5_skip \
MODEL_TYPE=dcae_adaln_skip_residual_linear \
EXTRA_ARGS="--latent-channels 32 $CLEAN_EXTRA" \
ENSEMBLE_PASSES=1 \
bash scripts/run_exp_0p5.sh > logs/runner/launcher_dcae_0p5_skip.log 2>&1 &
P0=$!
echo "DC-AE 0.5° Skip launched: PID=$P0 on GPU 0"
wait $P0
echo "=== DC-AE 0.5° Skip DONE ==="
date
