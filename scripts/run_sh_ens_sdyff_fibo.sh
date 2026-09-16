#!/usr/bin/env bash
# Launch S-DYff ensemble SH spectra (single sample vs ensemble mean vs GT)
# inside an ephemeral wti-train:v1 docker container on fibonacci.
#
# Defaults:
#   GPU=3, N=16, tau=3h, n_dates=24, channels = 8 representative.
# Overrides:
#   GPU=2 bash scripts/run_sh_ens_sdyff_fibo.sh
set -euo pipefail

GPU="${GPU:-3}"
N_ENS="${N_ENS:-16}"
TAU="${TAU:-3}"
N_DATES="${N_DATES:-24}"
CHANNELS="${CHANNELS:-t2m,u10,v10,mslp,T850,U850,Q850,Z700}"
WTI_ROOT="${WTI_ROOT:-/home/d.sukhorukov/weather_time_interpolation}"
HOST_CKPT="${CKPT:-${WTI_ROOT}/logs/exp_sdyff_dyffusion_0p5_6yr/epoch=7-step=70064.ckpt}"
OUT_DIR="${OUT_DIR:-metrics/sh_spectra_ens}"
NAME="${NAME:-sdyff_24ch_6yr}"
LOG_REL="logs/runner/sh_ens_sdyff_tau${TAU}_ens${N_ENS}.log"
HOST_LOG="${WTI_ROOT}/${LOG_REL}"
# Translate host ckpt path to in-container path (everything under WTI_ROOT
# is bind-mounted to /workspace/code/wti).
CONTAINER_CKPT="${HOST_CKPT/${WTI_ROOT}/\/workspace\/code\/wti}"

mkdir -p "${WTI_ROOT}/logs/runner" "${WTI_ROOT}/${OUT_DIR}"

CONTAINER="sh-ens-sdyff-tau${TAU}-ens${N_ENS}"

echo "launching ${CONTAINER} on GPU=${GPU}"
echo "  host ckpt:     ${HOST_CKPT}"
echo "  container ckpt:${CONTAINER_CKPT}"
echo "  out_dir:       ${OUT_DIR}"
echo "  host log:      ${HOST_LOG}"

docker run -d --rm --gpus "device=${GPU}" --shm-size 16g --name "${CONTAINER}" \
  -v "${WTI_ROOT}":/workspace/code/wti \
  -v /tmp/wti_cache:/tmp/wb2_0p5_cache \
  -e SDYFF_NLAT=360 -e SDYFF_NLON=720 -e SDYFF_LAT_CROP=0 \
  wti-train:v1 bash -c "set -e; cd /workspace/code/wti; \
    exec >> /workspace/code/wti/${LOG_REL} 2>&1; \
    echo '=== container start: '\$(date); \
    pip install --no-cache-dir -q torch_harmonics==0.6.5; \
    python -u tools/eval/sh_spectra_ens_sdyff.py \
      --ckpt ${CONTAINER_CKPT} \
      --memmap_dir /tmp/wb2_0p5_cache \
      --year 2020 \
      --n_ensemble ${N_ENS} \
      --n_dates ${N_DATES} \
      --tau_hours ${TAU} \
      --channels ${CHANNELS} \
      --batch_size 2 --num_workers 2 --samples_per_date 4 \
      --out_dir ${OUT_DIR} \
      --model_name ${NAME}; \
    echo '=== container done: '\$(date)"

echo "launched. tail -f ${HOST_LOG}"
