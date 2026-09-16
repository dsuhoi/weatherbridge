#!/usr/bin/env bash
# Phase 3: direct-prediction ablation (no bilinear-residual trick).
# Compares architecture's INHERENT capacity vs our `bilinear + α·v_θ` wrapper.
#
# Models: DC-AE + ModAFNO direct. S-DYff does not support direct-prediction.
# Same Aurora weights + no-physical-scales as Phase 2 — clean comparison.
set -eo pipefail
cd "$(dirname "$0")/.."

CLEAN_EXTRA="--no-physical-scales-loss --use-aurora-weights --direct-prediction"

# DC-AE v8 direct: same arch as Phase 2, but no bilinear baseline.
DEVICES=2 BATCH=2 WORKERS=8 EPOCHS=8 \
RUN_NAME=exp_dcae_v8_direct \
MODEL_TYPE=dcae_residual_linear \
EXTRA_ARGS="--latent-channels 32 $CLEAN_EXTRA" \
ENSEMBLE_PASSES=1 \
bash scripts/run_exp.sh

# ModAFNO v8 direct
DEVICES=2 BATCH=2 WORKERS=8 EPOCHS=8 \
RUN_NAME=exp_modafno_v8_direct \
MODEL_TYPE=modafno_official_residual_linear \
EXTRA_ARGS="--latent-channels 512 --modafno-depth 12 --modafno-num-blocks 8 --modafno-drop-rate 0.0 $CLEAN_EXTRA" \
ENSEMBLE_PASSES=1 \
bash scripts/run_exp.sh

echo "=== ALL PHASE 3 DIRECT DONE ==="
date
