#!/usr/bin/env bash
# Pair-launch: DC-AE Swin Skip (Model 1) + DC-AE Bilinear-XAttn (Model 2).
# Обе модели extends WeatherDCAEAdaLNSkipModel (v9 SOTA).
set -eo pipefail
cd "$(dirname "$0")/.."

CLEAN_EXTRA="--no-physical-scales-loss --use-aurora-weights --grid-aware-arch"

# Model 1: SwinV2 bottleneck вместо EfficientViTBlock
CUDA_VISIBLE_DEVICES=0 \
DEVICES=1 BATCH=2 WORKERS=8 EPOCHS=8 \
RUN_NAME=exp_dcae_v9_swin_skip \
MODEL_TYPE=dcae_swin_skip_residual_linear \
EXTRA_ARGS="--latent-channels 32 $CLEAN_EXTRA" \
ENSEMBLE_PASSES=1 \
bash scripts/run_exp.sh > logs/runner/launcher_dcae_swin_skip.log 2>&1 &
P0=$!
echo "DC-AE Swin Skip launched: PID=$P0 on GPU 0"

# Model 2: Bilinear injection в decoder через zero-init adapters
CUDA_VISIBLE_DEVICES=1 \
DEVICES=1 BATCH=2 WORKERS=8 EPOCHS=8 \
RUN_NAME=exp_dcae_v9_bilinear_xattn \
MODEL_TYPE=dcae_bilinear_xattn_residual_linear \
EXTRA_ARGS="--latent-channels 32 $CLEAN_EXTRA" \
ENSEMBLE_PASSES=1 \
bash scripts/run_exp.sh > logs/runner/launcher_dcae_bilinear_xattn.log 2>&1 &
P1=$!
echo "DC-AE Bilinear XAttn launched: PID=$P1 on GPU 1"

wait $P0 $P1
echo "=== HYBRID PAIR DONE ==="
date
