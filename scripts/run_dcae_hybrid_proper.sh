#!/usr/bin/env bash
# DC-AE Swin Skip + Bilinear XAttn — proper architecture (block_out=128/256/512).
# Fair comparison к parent DC-AE Skip (36M, RMSE 0.0925 train, -28.7% bilinear).
#
# RAM controls:
#   - Reduced num_workers (4 vs default 8) — fewer dataloader processes
#   - MALLOC_TRIM_THRESHOLD_ to encourage glibc free unused pages
#   - Each training ~200 GB cache+activations
#   - WB2 download peak ~300 GB
#   - Total budget ~700 GB / 2 TB available
set -eo pipefail
cd "$(dirname "$0")/.."

CLEAN_EXTRA="--no-physical-scales-loss --use-aurora-weights"
# Force explicit 3-stage architecture (matches parent Skip 36M)
ARCH_EXTRA="--block-out-channels 128 256 512 --layers-per-block 2 2 2"

export MALLOC_TRIM_THRESHOLD_=131072  # 128 KB; aggressive glibc trim

CUDA_VISIBLE_DEVICES=0 \
DEVICES=1 BATCH=2 WORKERS=4 EPOCHS=8 \
RUN_NAME=exp_dcae_v9_swin_skip_proper \
MODEL_TYPE=dcae_swin_skip_residual_linear \
EXTRA_ARGS="--latent-channels 32 $ARCH_EXTRA $CLEAN_EXTRA" \
ENSEMBLE_PASSES=1 \
bash scripts/run_exp.sh > logs/runner/launcher_swin_skip_proper.log 2>&1 &
P0=$!
echo "DC-AE Swin Skip (proper arch 36M) launched: PID=$P0 GPU 0"

CUDA_VISIBLE_DEVICES=1 \
DEVICES=1 BATCH=2 WORKERS=4 EPOCHS=8 \
RUN_NAME=exp_dcae_v9_bilinear_xattn_proper \
MODEL_TYPE=dcae_bilinear_xattn_residual_linear \
EXTRA_ARGS="--latent-channels 32 $ARCH_EXTRA $CLEAN_EXTRA" \
ENSEMBLE_PASSES=1 \
bash scripts/run_exp.sh > logs/runner/launcher_bilinear_xattn_proper.log 2>&1 &
P1=$!
echo "DC-AE Bilinear XAttn (proper arch 36M) launched: PID=$P1 GPU 1"

wait $P0 $P1
echo "=== HYBRID PAIR (proper arch) DONE ==="
date
