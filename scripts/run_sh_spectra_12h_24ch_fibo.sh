#!/usr/bin/env bash
# Fibonacci-side: 12h SH angular power spectra for the 6 fibo-trained 12h
# models + bilinear baseline, ALL 24 prognostic channels (was 5).
#
# Architecture: launches a fresh ``wti-train:v1`` container per model with
# ``--gpus device=3`` (only GPU 3 is free; the other 7 are running training
# jobs). Sequentially processes each model — ETA ~30-60 min/model on a
# single B300, so ~3-6 h wall-clock.
#
# Output: /home/d.sukhorukov/weather_time_interpolation/metrics/sh_spectra_12h_ep10_24ch/<name>_tau{2,3,5,8}.npz
#
# Logs: /home/d.sukhorukov/weather_time_interpolation/logs/runner/sh_spectra_24ch/<name>.log
#
# The WeatherDCAE 6yr ckpt lives on cloud.ru only — handled by
# ``run_sh_spectra_12h_24ch_cloudru.sh``.
set -euo pipefail

WTI_HOST_ROOT="${WTI_HOST_ROOT:-/home/d.sukhorukov/weather_time_interpolation}"
WTI_DOCKER_ROOT="${WTI_DOCKER_ROOT:-/workspace/code/wti}"
MEMMAP_HOST="${MEMMAP_HOST:-/tmp/wti_cache}"
MEMMAP_DOCKER="${MEMMAP_DOCKER:-/tmp/wb2_0p5_cache}"
OUT_DIR_HOST="${OUT_DIR_HOST:-${WTI_HOST_ROOT}/metrics/sh_spectra_12h_ep10_24ch}"
OUT_DIR_DOCKER="${OUT_DIR_DOCKER:-${WTI_DOCKER_ROOT}/metrics/sh_spectra_12h_ep10_24ch}"
LOG_DIR_HOST="${LOG_DIR_HOST:-${WTI_HOST_ROOT}/logs/runner/sh_spectra_24ch}"
GPU_ID="${GPU_ID:-3}"
TAUS="${TAUS:-2,3,5,8}"
IMAGE="${IMAGE:-wti-train:v1}"

mkdir -p "${OUT_DIR_HOST}" "${LOG_DIR_HOST}"

# (name : ckpt_in_docker : kind : envs)
MODELS=(
  "ATM-VFI_3yr_12h_fibo:${WTI_DOCKER_ROOT}/logs/exp_atmvfi_12h_oddskip/epoch=9-step=43740.ckpt:atm_vfi:"
  "DC-AE_NoSkip_3yr_12h_fibo:${WTI_DOCKER_ROOT}/logs/exp_12h_oddskip_dcae_noskip_3yr_fibo/epoch=9-step=10940.ckpt:hermite:"
  "DC-AE_Skip_3yr_12h_fibo:${WTI_DOCKER_ROOT}/logs/exp_12h_oddskip_dcae_2017_18_19/epoch=9-step=10940.ckpt:hermite:"
  "FuXi_3yr_12h_fibo:${WTI_DOCKER_ROOT}/logs/exp_12h_oddskip_fuxi_2017_18_19/epoch=9-step=5470.ckpt:hermite:"
  "ModAFNO_3yr_12h_fibo:${WTI_DOCKER_ROOT}/logs/exp_12h_oddskip_modafno_full_2017_18_19/epoch=9-step=21870.ckpt:hermite:MODAFNO_INP_H=362 MODAFNO_INP_W=720 MODAFNO_NATIVE_H=360 MODAFNO_NATIVE_W=720"
  "SDyff_3yr_12h_fibo:${WTI_DOCKER_ROOT}/logs/exp_12h_oddskip_sdyff_dyff_0p5_2017_18_19/epoch=9-step=5470.ckpt:hermite:SDYFF_NLAT=360 SDYFF_NLON=720 SDYFF_LAT_CROP=0"
  "bilinear:NONE:bilinear:"
)

echo "[run_sh_spectra_12h_24ch_fibo] $(date -u) starting"
echo "  GPU:    ${GPU_ID}"
echo "  out:    ${OUT_DIR_HOST}"
echo "  taus:   ${TAUS}"

for entry in "${MODELS[@]}"; do
  IFS=':' read -r name ckpt kind envs <<< "${entry}"
  log="${LOG_DIR_HOST}/${name}.log"
  echo ""
  echo "=== ${name} ==="
  echo "  ckpt:  ${ckpt}"
  echo "  kind:  ${kind}"
  echo "  envs:  ${envs:-(none)}"
  echo "  log:   ${log}"

  # If a real ckpt path was given, verify it on the host
  if [ "${ckpt}" != "NONE" ]; then
    host_ckpt="${ckpt/${WTI_DOCKER_ROOT}/${WTI_HOST_ROOT}}"
    if [ ! -e "${host_ckpt}" ]; then
      echo "  [missing ckpt] ${host_ckpt}" | tee -a "${log}"
      continue
    fi
  fi

  ckpt_arg=("--ckpt" "${ckpt}")
  if [ "${ckpt}" = "NONE" ]; then
    ckpt_arg=("--ckpt" "/dev/null")  # bilinear ignores --ckpt but argparse requires it
  fi

  docker run --rm \
    --gpus "device=${GPU_ID}" \
    -e NVIDIA_VISIBLE_DEVICES="${GPU_ID}" \
    -e PYTHONPATH="${WTI_DOCKER_ROOT}" \
    -e PIP_DISABLE_PIP_VERSION_CHECK=1 \
    -v "${WTI_HOST_ROOT}":"${WTI_DOCKER_ROOT}":rw \
    -v "${MEMMAP_HOST}":"${MEMMAP_DOCKER}":ro \
    -w "${WTI_DOCKER_ROOT}" \
    "${IMAGE}" \
    bash -c "set -euo pipefail; \
      python -c 'import torch_harmonics' 2>/dev/null || pip install --quiet torch_harmonics==0.6.5; \
      cd ${WTI_DOCKER_ROOT} && \
      python tools/eval/sh_energy_spectra_12h.py \
        --memmap-dir ${MEMMAP_DOCKER} \
        ${ckpt_arg[*]} \
        --model-name '${name}' \
        --model-kind '${kind}' \
        --keep-n-channels 24 \
        --channels all \
        --taus '${TAUS}' \
        --out-dir '${OUT_DIR_DOCKER}' \
        --envs '${envs}' \
        --skip-existing" \
    >> "${log}" 2>&1 || echo "  [fail] ${name} — see ${log}"
done

echo ""
echo "[run_sh_spectra_12h_24ch_fibo] $(date -u) DONE."
