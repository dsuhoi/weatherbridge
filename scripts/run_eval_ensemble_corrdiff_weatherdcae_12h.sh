#!/usr/bin/env bash
# Run on cloud.ru after the 12h CorrDiff DDP training (PID 443633) exits.
#
# Usage:
#   bash /home/jovyan/dsuhoi/weather_time_interpolation/scripts/run_eval_ensemble_corrdiff_weatherdcae_12h.sh
#
# Defaults: N=16 ensemble, bs=2 (lower than non-ensemble due to memory).
# ETA on A100: ~6-10h.
set -euo pipefail

REPO=/home/jovyan/dsuhoi/weather_time_interpolation
LOG_DIR="${REPO}/logs/exp_corrdiff_fm_weatherdcae_12h_6yr_cloudru"

# Pick best ckpt by parsing imp<value>.ckpt from filenames (take max imp).
FM_CKPT=$(ls -1 "${LOG_DIR}"/*imp*.ckpt 2>/dev/null \
  | awk -F'imp' '{print $2"\t"$0}' \
  | awk -F'.ckpt' '{print $1"\t"$0}' \
  | sort -k1 -g -r \
  | head -1 \
  | cut -f3)

if [[ -z "${FM_CKPT:-}" || ! -f "${FM_CKPT}" ]]; then
  FM_CKPT="${LOG_DIR}/last.ckpt"
fi

if [[ ! -f "${FM_CKPT}" ]]; then
  echo "FM_CKPT not found in ${LOG_DIR} — aborting."
  exit 1
fi

BASE_CKPT="${REPO}/logs/exp_12h_oddskip_dcae_noskip_cloudru_2014_19/last.ckpt"
if [[ ! -f "${BASE_CKPT}" ]]; then
  echo "BASE_CKPT not found: ${BASE_CKPT} — aborting."
  exit 1
fi

OUT_RMSE="${REPO}/metrics/eval_12h_2020_ep10/corrdiff_fm_weatherdcae_12h_6yr_ens16.json"
OUT_CRPS="${REPO}/metrics/crps_12h_2020/corrdiff_fm_weatherdcae_12h_6yr.json"
mkdir -p "$(dirname "${OUT_RMSE}")" "$(dirname "${OUT_CRPS}")"

LOG_FILE="${REPO}/logs/runner/eval_corrdiff_weatherdcae_12h_ens16.log"
mkdir -p "$(dirname "${LOG_FILE}")"

# Pick a free GPU. On cloud.ru we have 2x A100. After PID 443633 exits both are free.
GPU="${GPU:-0}"

# Critical: HOURS_PER_TAU_UNIT is hardcoded to env var WTI_HOURS_PER_TAU at config-module
# import time. Must be exported BEFORE python starts so tau_norm = tau_h / 12 (not /6).
export WTI_HOURS_PER_TAU=12.0
export CUDA_VISIBLE_DEVICES="${GPU}"

# Use the ai_scientist conda env directly (no conda activator needed).
PY="/home/jovyan/.mlspace/envs/ai_scientist/bin/python3.11"
if [[ ! -x "${PY}" ]]; then
  PY=python3
fi

# Eval runs natively (no docker on cloud.ru). data_root is the repo itself.
cd "${REPO}"
"${PY}" tools/eval/eval_ensemble_crps.py \
  --mode corrdiff_fm \
  --fm_ckpt "${FM_CKPT}" \
  --base_ckpt "${BASE_CKPT}" \
  --n_ensemble 16 \
  --batch_size 2 \
  --num_workers 4 \
  --samples_per_date 4 \
  --year 2020 \
  --memmap_dir /tmp/wb2_0p5_cache \
  --data_root "${REPO}" \
  --max_tau_hours 12 \
  --out_rmse "${OUT_RMSE}" \
  --out_crps "${OUT_CRPS}" \
  --model_name corrdiff_fm_weatherdcae_12h_6yr 2>&1 | tee "${LOG_FILE}"

echo "============================================="
echo "12h ensemble eval complete."
echo "FM ckpt:   ${FM_CKPT}"
echo "BASE ckpt: ${BASE_CKPT}"
echo "RMSE: ${OUT_RMSE}"
echo "CRPS: ${OUT_CRPS}"
echo "Log:  ${LOG_FILE}"
