#!/usr/bin/env bash
# Cloud.ru-side: 12h SH angular power spectra for WeatherDCAE NoSkip 6yr (the
# only 12h model whose ckpt lives on cloud.ru), all 24 prognostic channels.
#
# Runs on GPU 0 (single A100-80GB). The eval is per-sample sequential so DDP
# would only help if we sharded the dataloader — for ~30 min per τ on a single
# A100 we stick to single-GPU + sequential models.
#
# Output: /home/jovyan/dsuhoi/weather_time_interpolation/metrics/sh_spectra_12h_ep10_24ch/<name>_tau{2,3,5,8}.npz
set -euo pipefail

WTI_ROOT="${WTI_ROOT:-/home/jovyan/dsuhoi/weather_time_interpolation}"
MEMMAP_DIR="${MEMMAP_DIR:-/tmp/wb2_0p5_cache}"
OUT_DIR="${OUT_DIR:-${WTI_ROOT}/metrics/sh_spectra_12h_ep10_24ch}"
LOG_DIR="${LOG_DIR:-${WTI_ROOT}/logs/runner/sh_spectra_24ch}"
PYTHON="${PYTHON:-/home/jovyan/.mlspace/envs/ai_scientist/bin/python}"
TAUS="${TAUS:-2,3,5,8}"
GPU_ID="${GPU_ID:-0}"

mkdir -p "${OUT_DIR}" "${LOG_DIR}"

# (name : ckpt : kind : envs)
MODELS=(
  "WeatherDCAE_NoSkip_6yr_12h:${WTI_ROOT}/logs/exp_12h_oddskip_dcae_noskip_cloudru_2014_19/last.ckpt:hermite:"
)

echo "[run_sh_spectra_12h_24ch_cloudru] $(date -u) starting"
echo "  GPU:    ${GPU_ID}"
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
    echo "  [missing ckpt] ${ckpt}" | tee -a "${log}"
    continue
  fi
  CUDA_VISIBLE_DEVICES="${GPU_ID}" PYTHONPATH="${WTI_ROOT}" "${PYTHON}" tools/eval/sh_energy_spectra_12h.py \
    --memmap-dir "${MEMMAP_DIR}" \
    --ckpt "${ckpt}" \
    --model-name "${name}" \
    --model-kind "${kind}" \
    --keep-n-channels 24 \
    --channels all \
    --taus "${TAUS}" \
    --out-dir "${OUT_DIR}" \
    --envs "${envs}" \
    --skip-existing \
    >> "${log}" 2>&1 || echo "  [fail] ${name} — see ${log}"
done

echo ""
echo "[run_sh_spectra_12h_24ch_cloudru] $(date -u) DONE."
