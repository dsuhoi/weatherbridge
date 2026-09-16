#!/usr/bin/env bash
# Cluster-side: 6 h SH angular power spectra for the leading models.
#
# Adapts tools/eval/sh_energy_spectra_12h.py to the 6 h pipeline so we
# can compare HF preservation 6 h vs 12 h (does HF degrade more with
# longer interval?).
#
# Targets the same 5 channels {t2m, mslp, u10, v10, T850} at τ ∈ {1,3,5}
# (3 τ values for 6 h vs 4 for 12 h).
#
# Output: metrics/sh_spectra_6h_v1_hf_preservation/*.npz (1 per model × τ)
#         + figs/fig_sh_spectra_6h.{png,pdf} after plot_sh_spectra_12h.py
#         + tools/eval/analyze_sh_hf_preservation.py on the new dir.
#
# Run on fibo/cloud.ru with active GPU; ETA ~ 30 min total (6 models × 3 τ).
set -euo pipefail

WTI_ROOT="${WTI_ROOT:-/workspace/code/wti}"
MEMMAP="${WTI_MEMMAP_DIR:-${WTI_ROOT}/cache/wb2_0p5_cache}"
OUT_DIR="${OUT_DIR:-${WTI_ROOT}/metrics/sh_spectra_6h_v1_hf_preservation}"
LOG_DIR="${LOG_DIR:-${WTI_ROOT}/logs/sh_spectra_6h}"
TAUS="${TAUS:-1,3,5}"

mkdir -p "${OUT_DIR}" "${LOG_DIR}"

# (name : ckpt : kind : envs)
# kind ∈ {hermite, atmvfi}
MODELS=(
  "dcae_skip_0p5_6yr_pad:${WTI_ROOT}/logs/exp_dcae_skip_0p5_6yr_pad/epoch=7-step=70064.ckpt:hermite:LAT_CROP=-8"
  "sdyff_dyffusion_0p5_6yr:${WTI_ROOT}/logs/exp_sdyff_dyffusion_0p5_6yr/epoch=7-step=70064.ckpt:hermite:SDYFF_NLAT=360 SDYFF_NLON=720 SDYFF_LAT_CROP=0"
  "modafno_0p5_6yr:${WTI_ROOT}/logs/exp_modafno_0p5_6yr/last.ckpt:hermite:MODAFNO_INP_H=360 MODAFNO_INP_W=720 MODAFNO_NATIVE_H=360 MODAFNO_NATIVE_W=720"
  "fuxi_0p5_6yr:${WTI_ROOT}/logs/exp_fuxi_0p5_6yr_full/last.ckpt:hermite:"
  "weatherdcae_noskip_24ch_6yr_ep10:${WTI_ROOT}/logs/exp_dcae_noskip_24ch_6yr/last.ckpt:hermite:"
  "atm_vfi_24ch_6yr:${WTI_ROOT}/logs/exp_atmvfi_v2_static_24ch_6yr/last.ckpt:atmvfi:"
)

echo "[run_sh_spectra_6h] $(date -u) starting"
echo "  memmap: ${MEMMAP}"
echo "  out:    ${OUT_DIR}"
echo "  taus:   ${TAUS}"

for entry in "${MODELS[@]}"; do
  IFS=':' read -r name ckpt kind envs <<< "${entry}"
  log="${LOG_DIR}/${name}.log"
  echo ""
  echo "=== ${name} ==="
  echo "  ckpt:  ${ckpt}"
  echo "  kind:  ${kind}"
  echo "  envs:  ${envs:-(none)}"
  echo "  log:   ${log}"
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
echo "[run_sh_spectra_6h] $(date -u) DONE."
echo "  Next:"
echo "    python tools/eval/plot_sh_spectra_12h.py \\"
echo "      --npz-dir ${OUT_DIR} \\"
echo "      --out figs/fig_sh_spectra_6h \\"
echo "      --taus ${TAUS} \\"
echo "      --channels t2m,mslp,u10,T850"
echo "    python tools/eval/analyze_sh_hf_preservation.py \\"
echo "      --npz-dir ${OUT_DIR} \\"
echo "      --out-tex paper/tab_sh_hf_preservation_6h.tex"
