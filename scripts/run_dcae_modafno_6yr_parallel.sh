#!/usr/bin/env bash
# Driver: preprocess 2014-2019 → /tmp memmap, then launch DC-AE Skip 6yr on GPU0
# and ModAFNO 6yr on GPU1 in parallel.
#
# Usage on cluster:
#   cd /home/jovyan/dsuhoi/weather_time_interpolation
#   nohup bash scripts/run_dcae_modafno_6yr_parallel.sh > logs/runner/parallel_6yr_driver.log 2>&1 &
set -eo pipefail
source /home/user/conda/etc/profile.d/conda.sh
conda activate /home/jovyan/.mlspace/envs/ai_scientist
cd "$(dirname "$0")/.."

mkdir -p logs/runner

# --- Stage 1: preprocess 2014-2019 sequentially ---
NEED_YEARS=()
for y in 2014 2015 2016 2017 2018 2019; do
  if [ ! -f "/tmp/wb2_0p5_cache/wb2_${y}.bin" ]; then
    NEED_YEARS+=("$y")
  fi
done

if [ "${#NEED_YEARS[@]}" -gt 0 ]; then
  echo "=== PREPROCESS years: ${NEED_YEARS[*]} ===" ; date
  mkdir -p /tmp/wb2_0p5_cache
  python -u tools/data/preprocess_0p5_to_memmap.py --years "${NEED_YEARS[@]}" \
    2>&1 | tee logs/runner/preprocess_2014_2019.log
  echo "=== PREPROCESS done ===" ; date
else
  echo "=== PREPROCESS skipped (all years present) ==="
fi

# Sanity check: all 6 .bin files exist
for y in 2014 2015 2016 2017 2018 2019; do
  [ -f "/tmp/wb2_0p5_cache/wb2_${y}.bin" ] || { echo "MISSING: /tmp/wb2_0p5_cache/wb2_${y}.bin"; exit 1; }
done

export YEARS="2014 2015 2016 2017 2018 2019"
export MAX_EPOCHS=8
export BATCH_SIZE=1
export LR=1e-4
export LATENT_CHANNELS=256
export NUM_WORKERS=4

# --- Stage 2: launch DC-AE Skip 6yr on GPU0 ---
DCAE_OUT="logs/exp_dcae_skip_0p5_6yr_pad"
mkdir -p "$DCAE_OUT"
echo "=== LAUNCH DC-AE Skip 6yr on GPU0 ===" ; date
CUDA_VISIBLE_DEVICES=0 \
  MODEL_TYPE=dcae_adaln_skip_residual_linear \
  LAT_CROP=-8 \
  OUT_DIR="$DCAE_OUT" \
  nohup python -u scripts/train_0p5_memmap.py \
  > logs/runner/train_dcae_skip_0p5_6yr.log 2>&1 &
DCAE_PID=$!
echo "DC-AE PID=$DCAE_PID"

# --- Stage 3: launch ModAFNO 6yr on GPU1 ---
MODAFNO_OUT="logs/exp_modafno_0p5_6yr"
mkdir -p "$MODAFNO_OUT"
echo "=== LAUNCH ModAFNO 6yr on GPU1 ===" ; date
CUDA_VISIBLE_DEVICES=1 \
  MODEL_TYPE=modafno_residual_linear \
  MODAFNO_INP_H=360 MODAFNO_INP_W=720 \
  MODAFNO_NATIVE_H=360 MODAFNO_NATIVE_W=720 \
  OUT_DIR="$MODAFNO_OUT" \
  nohup python -u scripts/train_0p5_memmap.py \
  > logs/runner/train_modafno_0p5_6yr.log 2>&1 &
MODAFNO_PID=$!
echo "ModAFNO PID=$MODAFNO_PID"

echo "=== BOTH LAUNCHED at $(date) ==="
echo "DC-AE PID=$DCAE_PID    log: logs/runner/train_dcae_skip_0p5_6yr.log"
echo "ModAFNO PID=$MODAFNO_PID  log: logs/runner/train_modafno_0p5_6yr.log"
echo "Driver exiting; trainings continue under nohup."
