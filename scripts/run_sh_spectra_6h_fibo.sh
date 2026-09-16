#!/usr/bin/env bash
# Cluster-side: 6 h SH angular power spectra for 4 models with ckpts on fibo.
# This is a fibo-only subset of scripts/run_sh_spectra_6h.sh — the two 24ch
# models (WeatherDCAE NoSkip 6yr ep10, ATM-VFI v2 static 24ch 6yr) live on
# cloud.ru only, so they are not run here.
#
# Run inside docker container, with /tmp/wb2_0p5_cache mounted.
set -euo pipefail

WTI_ROOT="${WTI_ROOT:-/workspace/code/wti}"
MEMMAP="${WTI_MEMMAP_DIR:-/tmp/wb2_0p5_cache}"
OUT_DIR="${OUT_DIR:-${WTI_ROOT}/metrics/sh_spectra_6h}"
LOG_DIR="${LOG_DIR:-${WTI_ROOT}/logs/sh_spectra_6h}"
TAUS="${TAUS:-1,3,5}"

mkdir -p "${OUT_DIR}" "${LOG_DIR}"

MODELS=(
  "dcae_skip_0p5_6yr_pad:${WTI_ROOT}/logs/exp_dcae_skip_0p5_6yr_pad/epoch=7-step=70064.ckpt:hermite:LAT_CROP=-8"
  "sdyff_dyffusion_0p5_6yr:${WTI_ROOT}/logs/exp_sdyff_dyffusion_0p5_6yr/epoch=7-step=70064.ckpt:hermite:SDYFF_NLAT=360 SDYFF_NLON=720 SDYFF_LAT_CROP=0"
  "modafno_0p5_6yr:${WTI_ROOT}/logs/exp_modafno_0p5_6yr/last.ckpt:hermite:MODAFNO_INP_H=360 MODAFNO_INP_W=720 MODAFNO_NATIVE_H=360 MODAFNO_NATIVE_W=720"
  "fuxi_0p5_6yr:${WTI_ROOT}/logs/exp_fuxi_0p5_6yr_full/last.ckpt:hermite:"
)

echo "[run_sh_spectra_6h_fibo] $(date -u) starting"
echo "  memmap: ${MEMMAP}"
echo "  out:    ${OUT_DIR}"
echo "  taus:   ${TAUS}"

cd "${WTI_ROOT}"

for entry in "${MODELS[@]}"; do
  IFS=':' read -r name ckpt kind envs <<< "${entry}"
  log="${LOG_DIR}/${name}.log"
  echo ""
  echo "=== ${name} ==="
  echo "  ckpt:  ${ckpt}"
  echo "  kind:  ${kind}"
  echo "  envs:  ${envs:-(none)}"
  echo "  log:   ${log}"
  if [ ! -e "${ckpt}" ]; then
    echo "  [missing ckpt] ${ckpt}"
    continue
  fi
  python tools/eval/sh_energy_spectra_12h.py \
    --memmap-dir "${MEMMAP}" \
    --ckpt "${ckpt}" \
    --model-name "${name}" \
    --model-kind "${kind}" \
    --keep-n-channels 24 \
    --taus "${TAUS}" \
    --out-dir "${OUT_DIR}" \
    --max-tau-hours 6 \
    --envs "${envs}" \
    2>&1 | tee "${log}" || echo "  [fail] ${name}"
done

echo ""
echo "[run_sh_spectra_6h_fibo] $(date -u) DONE."
