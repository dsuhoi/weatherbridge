#!/usr/bin/env bash
# Resume +N epochs continuation for 12h-oddskip 3yr models on fibonacci.
#
# Usage:
#   bash scripts/launch_12h_resume_extension.sh <model_short> <gpu_idx> [extra_epochs]
#
# Model shorts: dcae_skip | sdyff | atmvfi
# Reads last.ckpt from logs/exp_12h_oddskip_<model>_*/last.ckpt and resumes via
# RESUME_CKPT env var (--ckpt-path for ATM-VFI). Trainer max_epochs is extended
# to current_done + EXTRA so Lightning runs EXTRA more epochs from the ckpt state.
#
# Expects on fibonacci:
#   /workspace/code/wti/ (mounted from /home/d.sukhorukov/weather_time_interpolation)

set -eo pipefail
MODEL=${1:-dcae_skip}
GPU=${2:-0}
EXTRA=${3:-6}

echo "[resume] model=$MODEL gpu=$GPU +$EXTRA epochs"

case "$MODEL" in
  dcae_skip)
    NAME=exp_12h_oddskip_dcae_2017_18_19
    MT=dcae_adaln_skip_residual_linear
    BS=2; LC=256; EPOCHS_DONE=7   # ep7 done (stopped earlier on plateau)
    EXTRA_ENV=""
    ;;
  sdyff)
    NAME=exp_12h_oddskip_sdyff_dyff_0p5_2017_18_19
    MT=sdyff_dyffusion_residual_linear
    BS=4; LC=96; EPOCHS_DONE=10
    EXTRA_ENV="SDYFF_NLAT=360 SDYFF_NLON=720 SDYFF_LAT_CROP=0"
    PRECISION=32
    ;;
  *)
    echo "unknown MODEL $MODEL"; exit 1;;
esac

PRECISION=${PRECISION:-bf16-mixed}
NEW_MAX=$((EPOCHS_DONE + EXTRA))
OUT=/workspace/code/wti/logs/${NAME}
RESUME=${OUT}/last.ckpt
LOG=/workspace/code/wti/logs/runner/${NAME}_resume.log

CUDA_VISIBLE_DEVICES=$GPU \
  YEARS="2017 2018 2019" VAL_YEARS=2020 MAX_EPOCHS=$NEW_MAX \
  BATCH_SIZE=$BS LR=1e-4 MODEL_TYPE=$MT LATENT_CHANNELS=$LC \
  KEEP_24CH=1 NUM_WORKERS=6 VAL_NUM_WORKERS=3 \
  LIMIT_VAL_BATCHES=80 SAMPLES_PER_DATE_VAL=1 VAL_BATCH_SIZE=$BS \
  OUT_DIR=logs/${NAME} CKPT_EVERY_N_EPOCHS=2 \
  PRECISION=$PRECISION VAL_EVERY_N_EPOCHS=1 \
  RESUME_CKPT=$RESUME \
  ${EXTRA_ENV} \
  python -u scripts/train_0p5_memmap_12h_oddskip.py > "$LOG" 2>&1 &
PID=$!
echo "started $MODEL @ GPU=$GPU PID=$PID resume=$RESUME → $NEW_MAX max_epochs → $LOG"
echo $PID
