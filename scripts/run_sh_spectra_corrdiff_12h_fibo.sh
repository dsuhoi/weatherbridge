#!/usr/bin/env bash
# Compute CorrDiff-FM SH angular power spectra on fibo (B300), single GPU.
#
# Polite to other tenants: 1) uses GPU id supplied by GPU= env (default 3),
# 2) caps total batches via MAX_BATCHES (≈200 → ~30 s/τ on a free B300),
# 3) batch-size 1 so peak VRAM stays under 20 GB.
#
# Output mirrors the existing 24ch 12 h cache:
#   metrics/sh_spectra_12h_ep10_24ch/corrdiff_fm_weatherdcae_12h_6yr_tau{2,3}.npz
#
# Re-run safe (skip-existing).
set -euo pipefail

WTI_HOST_ROOT=/home/d.sukhorukov/weather_time_interpolation
WTI_DOCKER_ROOT=/workspace/code/wti
MEMMAP_HOST=/tmp/wti_cache
MEMMAP_DOCKER=/tmp/wb2_0p5_cache

NAME="${NAME:-corrdiff_fm_weatherdcae_12h_6yr}"
FM_CKPT=${WTI_DOCKER_ROOT}/logs/exp_corrdiff_fm_weatherdcae_12h_6yr_cloudru/6-61264-imp0.018.ckpt
BASE_CKPT=${WTI_DOCKER_ROOT}/logs/exp_12h_oddskip_dcae_noskip_cloudru_2014_19/last.ckpt
OUT_DIR_HOST=${WTI_HOST_ROOT}/metrics/sh_spectra_12h_ep10_24ch
OUT_DIR_DOCKER=${WTI_DOCKER_ROOT}/metrics/sh_spectra_12h_ep10_24ch
LOG=${WTI_HOST_ROOT}/logs/runner/sh_spectra_24ch/${NAME}.log

GPU="${GPU:-3}"
TAUS="${TAUS:-2,3}"
MAX_BATCHES="${MAX_BATCHES:-200}"
ODE_STEPS="${ODE_STEPS:-}"
IMAGE="${IMAGE:-wti-train:v1}"

mkdir -p "$(dirname "${LOG}")" "${OUT_DIR_HOST}"

echo "[corrdiff-sh] $(date -u)"
echo "  GPU=${GPU}  TAUS=${TAUS}  MAX_BATCHES=${MAX_BATCHES}"
echo "  FM:    ${FM_CKPT}"
echo "  BASE:  ${BASE_CKPT}"
echo "  log:   ${LOG}"

ode_arg=""
if [[ -n "${ODE_STEPS}" ]]; then
  ode_arg="--ode-steps ${ODE_STEPS}"
fi

docker run --rm \
  --gpus "device=${GPU}" \
  --shm-size 16g \
  -e NVIDIA_VISIBLE_DEVICES="${GPU}" \
  -e PYTHONPATH="${WTI_DOCKER_ROOT}" \
  -e PIP_DISABLE_PIP_VERSION_CHECK=1 \
  -v "${WTI_HOST_ROOT}":"${WTI_DOCKER_ROOT}":rw \
  -v "${MEMMAP_HOST}":"${MEMMAP_DOCKER}":ro \
  -w "${WTI_DOCKER_ROOT}" \
  "${IMAGE}" \
  bash -c "set -euo pipefail; \
    python -c 'import torch_harmonics' 2>/dev/null || pip install --quiet torch_harmonics==0.6.5; \
    cd ${WTI_DOCKER_ROOT} && \
    python tools/eval/sh_energy_spectra_corrdiff.py \
      --memmap-dir ${MEMMAP_DOCKER} \
      --fm-ckpt '${FM_CKPT}' \
      --base-ckpt '${BASE_CKPT}' \
      --model-name '${NAME}' \
      --out-dir '${OUT_DIR_DOCKER}' \
      --taus '${TAUS}' \
      --max-batches '${MAX_BATCHES}' \
      ${ode_arg} \
      --skip-existing" \
  >> "${LOG}" 2>&1

echo "[corrdiff-sh] $(date -u) done — see ${LOG}"
echo "Output files:"
ls -la "${OUT_DIR_HOST}/${NAME}_tau"*.npz 2>/dev/null || echo "  (none yet)"
