#!/usr/bin/env bash
# Recompute SH energy spectra at lmax=360 (= 0.5° grid Nyquist on the
# 360-lat equal-angle grid) for the 5 paper models × τ∈{2,3}.
#
# Output goes to a NEW directory so the lmax=180 cache survives:
#   metrics/sh_spectra_12h_ep10_24ch_lmax360/{model}_tau{2,3}.npz
# After this finishes, update make_fig3_spectra.py NPZ_DIR to point at
# the new dir (or symlink it back to the canonical name).
#
# Run on cluster GPU when KD-v4 finishes (do NOT preempt PID 588338).
set -euo pipefail

OUT_DIR="metrics/sh_spectra_12h_ep10_24ch_lmax360"
LMAX=360
TAUS="2,3"
DEV="${DEV:-cuda}"

mkdir -p "${OUT_DIR}"

# Models — match stems used in make_fig3_spectra.py MODELS dict.
declare -A CKPTS=(
  [bilinear]=""
  [FuXi_3yr_12h_fibo]="logs/fuxi_24ch_6yr/best.ckpt"
  [ModAFNO_3yr_12h_fibo]="logs/modafno_24ch_6yr/best.ckpt"
  [SDyff_3yr_12h_fibo]="logs/sdyff_24ch_6yr/best.ckpt"
  [WeatherDCAE_NoSkip_6yr_12h]="logs/weatherdcae_noskip_24ch_6yr/best.ckpt"
)

for NAME in "${!CKPTS[@]}"; do
  CKPT="${CKPTS[$NAME]}"
  echo "==> ${NAME}  (lmax=${LMAX})"
  python tools/eval/sh_energy_spectra_12h.py \
    --model_name "${NAME}" \
    --ckpt "${CKPT}" \
    --taus "${TAUS}" \
    --lmax "${LMAX}" \
    --out_dir "${OUT_DIR}" \
    --device "${DEV}" \
    --skip_existing
done

echo
echo "Done. To switch the paper figure to lmax=360 data:"
echo "  ln -sfn sh_spectra_12h_ep10_24ch_lmax360 metrics/sh_spectra_12h_ep10_24ch_current"
echo "  # then edit scripts/make_fig3_spectra.py NPZ_DIR -> *_lmax360"
