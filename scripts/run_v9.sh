#!/usr/bin/env bash
# v9 retrain: modern τ-conditioning (no τ-map) + proper DYffusion process.
#   - DC-AE v9: τ via sinusoidal+MLP → native FiLM (temb), no τ-map channel
#   - S-DYff v9: two-network DYffusion (interpolator I_φ + refiner R_θ),
#                K=5 refine steps at inference, K=1 at train
#
# Parallel single-GPU pair (GPU 0 + GPU 1) for 2× throughput vs DDP-2.
set -eo pipefail
cd "$(dirname "$0")/.."

CLEAN_EXTRA="--no-physical-scales-loss --use-aurora-weights"

# DC-AE v9 (AdaLN-Zero/FiLM) — GPU 0
CUDA_VISIBLE_DEVICES=0 \
DEVICES=1 BATCH=2 WORKERS=8 EPOCHS=8 \
RUN_NAME=exp_dcae_v9_adaln \
MODEL_TYPE=dcae_adaln_residual_linear \
EXTRA_ARGS="--latent-channels 32 $CLEAN_EXTRA" \
ENSEMBLE_PASSES=1 \
bash scripts/run_exp.sh > logs/runner/launcher_dcae_v9.log 2>&1 &
P0=$!
echo "DC-AE v9 launched: PID=$P0 on GPU 0"

# S-DYff v9 (DYffusion interpolator+refiner) — GPU 1
CUDA_VISIBLE_DEVICES=1 \
DEVICES=1 BATCH=2 WORKERS=8 EPOCHS=8 \
RUN_NAME=exp_sdyff_v9_dyffusion \
MODEL_TYPE=sdyff_dyffusion_residual_linear \
EXTRA_ARGS="--latent-channels 96 --sdyff-num-layers 6 --sdyff-n-modes-lat 16 --sdyff-n-modes-lon 32 --sdyff-dropout 0.1 --sdyff-drop-path 0.1 --sdyff-inference-steps 5 --sdyff-train-refine-steps 1 --precision 32 $CLEAN_EXTRA" \
ENSEMBLE_PASSES=1 \
bash scripts/run_exp.sh > logs/runner/launcher_sdyff_v9.log 2>&1 &
P1=$!
echo "S-DYff v9 launched: PID=$P1 on GPU 1"

wait $P0 $P1
echo "=== ALL v9 DONE ==="
date
