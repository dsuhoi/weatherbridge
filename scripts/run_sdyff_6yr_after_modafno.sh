#!/usr/bin/env bash
# 1) Wait for ModAFNO 6yr (env MODAFNO_PID) to exit.
# 2) Quick post-hoc eval on saved per-epoch checkpoints of DC-AE Skip 6yr +
#    ModAFNO 6yr → metrics/{acc,eval}_0p5_2020_epoch_trace/ (small slice).
# 3) Launch S-DYff DYffusion 6yr on GPU1.
#
# DEPRECATED (2026-06-12) — the launch step at the bottom is now equivalent to:
#   CUDA_VISIBLE_DEVICES=1 python train.py model=sdyff trainer=default \
#       exp_name=exp_sdyff_dyffusion_0p5_6yr data.years='[2014,2015,2016,2017,2018,2019]'
# The mid-script per-epoch ckpt eval still uses tools/eval/batch_eval_memmap.py
# directly because the comma-separated ckpt list is built dynamically; this can
# also route through `python eval.py eval=memmap_6h ++eval.models='...'`.
set -eo pipefail
source /home/user/conda/etc/profile.d/conda.sh
conda activate /home/jovyan/.mlspace/envs/ai_scientist
cd "$(dirname "$0")/.."

mkdir -p logs/runner metrics/acc_0p5_2020_epoch_trace metrics/eval_0p5_2020_epoch_trace

WAIT_PID=${MODAFNO_PID:-}
if [ -n "$WAIT_PID" ]; then
  echo "=== WAITING for PID $WAIT_PID to exit (ModAFNO 6yr) ==="
  date
  while kill -0 "$WAIT_PID" 2>/dev/null; do
    sleep 60
  done
  echo "=== PID $WAIT_PID exited; GPU1 free ==="
  date
fi

# Stabilize a few seconds for CUDA mem release
sleep 10

# === Quick eval of saved per-epoch checkpoints on small 2020 slice ===
echo "=== QUICK EVAL of saved epoch checkpoints (GPU1) ==="
date

MODELS_LIST=""
for ckpt in $(ls -v logs/exp_dcae_skip_0p5_6yr_pad/epoch=*.ckpt 2>/dev/null); do
  e=$(basename "$ckpt" | sed 's/epoch=\([0-9]*\).*/\1/')
  MODELS_LIST="${MODELS_LIST}${MODELS_LIST:+,}dcae_skip_6yr_e${e}:${ckpt}:LAT_CROP=-8"
done
for ckpt in $(ls -v logs/exp_modafno_0p5_6yr/epoch=*.ckpt 2>/dev/null); do
  e=$(basename "$ckpt" | sed 's/epoch=\([0-9]*\).*/\1/')
  MODELS_LIST="${MODELS_LIST}${MODELS_LIST:+,}modafno_6yr_e${e}:${ckpt}:MODAFNO_INP_H=360 MODAFNO_INP_W=720 MODAFNO_NATIVE_H=360 MODAFNO_NATIVE_W=720"
done

if [ -z "$MODELS_LIST" ]; then
  echo "[WARN] No per-epoch ckpts found; skipping quick eval"
else
  echo "Models for quick eval:"
  echo "$MODELS_LIST" | tr ',' '\n' | sed 's/^/  /'
  CUDA_VISIBLE_DEVICES=1 python -u tools/eval/batch_eval_memmap.py \
    --memmap-dir /tmp/wb2_0p5_cache --test-year 2020 \
    --models "$MODELS_LIST" \
    --out-acc-dir metrics/acc_0p5_2020_epoch_trace \
    --out-rmse-dir metrics/eval_0p5_2020_epoch_trace \
    --batch-size 4 --num-workers 2 --samples-per-date 4 --eval-days-per-month 1 \
    2>&1 | tee logs/runner/quick_eval_epoch_trace.log
fi
echo "=== QUICK EVAL DONE ==="
date

# === Launch S-DYff DYffusion 6yr ===
export YEARS="2014 2015 2016 2017 2018 2019"
export MAX_EPOCHS=8
export BATCH_SIZE=1
export LR=1e-4
export LATENT_CHANNELS=256
export NUM_WORKERS=4

SDYFF_OUT="logs/exp_sdyff_dyffusion_0p5_6yr"
mkdir -p "$SDYFF_OUT"
echo "=== LAUNCH S-DYff DYffusion 6yr on GPU1 (0.5°) ==="
date

CUDA_VISIBLE_DEVICES=1 \
  MODEL_TYPE=sdyff_dyffusion_residual_linear \
  SDYFF_NLAT=360 SDYFF_NLON=720 SDYFF_LAT_CROP=0 \
  OUT_DIR="$SDYFF_OUT" \
  nohup python -u scripts/train_0p5_memmap.py \
  > logs/runner/train_sdyff_dyffusion_0p5_6yr.log 2>&1 &
SDYFF_PID=$!
echo "S-DYff PID=$SDYFF_PID    log: logs/runner/train_sdyff_dyffusion_0p5_6yr.log"
echo "=== LAUNCHED at $(date) ==="
