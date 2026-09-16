#!/usr/bin/env bash
# CorrDiff-FM SH spectra on fibo (B300) WITH /mnt/storage workspace —
# necessary because root fs is 100% full.
#
# Layout:
#   read source:  /home/d.sukhorukov/weather_time_interpolation -> /workspace/code/wti (ro)
#   write script: /mnt/storage/d.sukhorukov/wti_storage/tools/eval/sh_energy_spectra_corrdiff.py
#                 → /workspace/code/wti/tools/eval/sh_energy_spectra_corrdiff.py  (overlay file)
#   write npz:    /mnt/storage/d.sukhorukov/wti_storage/metrics/sh_spectra_12h_ep10_24ch
#                 → /workspace/code/wti/metrics/sh_spectra_12h_ep10_24ch  (overlay dir)
#   write logs:   /mnt/storage/d.sukhorukov/wti_storage/logs/runner
#                 → /workspace/code/wti/logs/runner  (overlay dir)
#   write home/pip-cache: /dev/shm/d.sukhorukov.home (tmpfs, 1TB free)
#
# Output: /mnt/storage/d.sukhorukov/wti_storage/metrics/sh_spectra_12h_ep10_24ch/<name>_tau{2,3}.npz
#         Pull back to laptop via scp from /mnt/storage path.
set -euo pipefail

WTI_HOST_ROOT=/home/d.sukhorukov/weather_time_interpolation
WTI_DOCKER_ROOT=/workspace/code/wti
MEMMAP_HOST=/tmp/wti_cache
MEMMAP_DOCKER=/tmp/wb2_0p5_cache
STORAGE_ROOT=/mnt/storage/d.sukhorukov/wti_storage

NAME="${NAME:-corrdiff_fm_weatherdcae_12h_6yr}"
FM_CKPT=${WTI_DOCKER_ROOT}/logs/exp_corrdiff_fm_weatherdcae_12h_6yr_cloudru/6-61264-imp0.018.ckpt
BASE_CKPT=${WTI_DOCKER_ROOT}/logs/exp_12h_oddskip_dcae_noskip_cloudru_2014_19/last.ckpt
OUT_DIR_STORAGE=${STORAGE_ROOT}/metrics/sh_spectra_12h_ep10_24ch
OUT_DIR_DOCKER=${WTI_DOCKER_ROOT}/metrics/sh_spectra_12h_ep10_24ch
LOG_DIR_STORAGE=${STORAGE_ROOT}/logs/runner
LOG=${LOG_DIR_STORAGE}/${NAME}.log

GPU="${GPU:-3}"
TAUS="${TAUS:-2,3}"
MAX_BATCHES="${MAX_BATCHES:-200}"
ODE_STEPS="${ODE_STEPS:-}"
IMAGE="${IMAGE:-wti-train:v1}"

mkdir -p "${OUT_DIR_STORAGE}" "${LOG_DIR_STORAGE}"

# tmpfs home for pip cache / matplotlib etc — survives only for container lifetime.
TMPFS_HOME=/dev/shm/d.sukhorukov.home.${NAME}
rm -rf "${TMPFS_HOME}" || true
mkdir -p "${TMPFS_HOME}"

echo "[corrdiff-sh] $(date -u)"
echo "  GPU=${GPU}  TAUS=${TAUS}  MAX_BATCHES=${MAX_BATCHES}"
echo "  FM:        ${FM_CKPT}"
echo "  BASE:      ${BASE_CKPT}"
echo "  output:    ${OUT_DIR_STORAGE}"
echo "  log:       ${LOG}"
echo "  tmpfs-home:${TMPFS_HOME}"

ode_arg=""
if [[ -n "${ODE_STEPS}" ]]; then
  ode_arg="--ode-steps ${ODE_STEPS}"
fi

docker run --rm \
  --gpus "device=${GPU}" \
  --shm-size 16g \
  --user "$(id -u):$(id -g)" \
  -e NVIDIA_VISIBLE_DEVICES="${GPU}" \
  -e PYTHONPATH="${WTI_DOCKER_ROOT}" \
  -e HOME="/tmp_home" \
  -e PIP_CACHE_DIR="/tmp_home/.cache/pip" \
  -e MPLCONFIGDIR="/tmp_home/.matplotlib" \
  -e PIP_DISABLE_PIP_VERSION_CHECK=1 \
  -v "${WTI_HOST_ROOT}":"${WTI_DOCKER_ROOT}":ro \
  -v "${MEMMAP_HOST}":"${MEMMAP_DOCKER}":ro \
  -v "${STORAGE_ROOT}/tools/eval/sh_energy_spectra_corrdiff.py":"${WTI_DOCKER_ROOT}/tools/eval/sh_energy_spectra_corrdiff.py":ro \
  -v "${OUT_DIR_STORAGE}":"${OUT_DIR_DOCKER}":rw \
  -v "${TMPFS_HOME}":"/tmp_home":rw \
  -w "${WTI_DOCKER_ROOT}" \
  "${IMAGE}" \
  bash -c "set -euo pipefail; \
    python -c 'import torch_harmonics' 2>/dev/null || pip install --quiet --target=/tmp_home/pyextra torch_harmonics==0.6.5; \
    export PYTHONPATH=${WTI_DOCKER_ROOT}:/tmp_home/pyextra; \
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

echo "[corrdiff-sh] $(date -u) done"
echo "Output:"
ls -la "${OUT_DIR_STORAGE}/${NAME}_tau"*.npz 2>/dev/null || echo "  (no npz produced — see log)"
