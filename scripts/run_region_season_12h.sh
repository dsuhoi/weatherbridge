#!/usr/bin/env bash
# Cluster-side launch script for per-region × per-season 12h eval.
#
# Runs `tools/eval/region_season_12h_eval.py` on the 7 strict-ep10 12h
# checkpoints. Produces `metrics/region_season_12h/<model_name>.json` for
# each. Subsequent table generation: tools/eval/build_region_season_12h_table.py.
#
# ⚠ Requires:
#   - Active GPU (1× A100/B300 enough for sequential pass)
#   - Memmap dataset at $WTI_MEMMAP_DIR (default /workspace/code/wti/cache/wb2_0p5_cache)
#   - Lightning environment with weather_time_interp installed
#   - Checkpoints at exact ep10 paths listed in docs/eval_history.md
#
# Run on `fibonacci` (or cloud.ru if memmap available). Do NOT run if any
# other GPU job is using >40% memory.
#
# Total ETA: ~70 min on B300 (10 min/model × 7).
set -euo pipefail

WTI_ROOT="${WTI_ROOT:-/workspace/code/wti}"
MEMMAP="${WTI_MEMMAP_DIR:-/tmp/wb2_0p5_cache}"
OUT_DIR="${OUT_DIR:-${WTI_ROOT}/metrics/region_season_12h_2020}"
LOG_DIR="${LOG_DIR:-${WTI_ROOT}/logs/region_season_12h}"

mkdir -p "${OUT_DIR}" "${LOG_DIR}"

# (name : ckpt : envs) — single line per model, semicolon-separated for
# the underlying script.
MODELS=(
  "ATM-VFI_3yr_12h_fibo:${WTI_ROOT}/logs/exp_atmvfi_12h_oddskip/epoch=9-step=43740.ckpt:"
  "DC-AE_NoSkip_3yr_12h_fibo:${WTI_ROOT}/logs/exp_12h_oddskip_dcae_noskip_3yr_fibo/epoch=9-step=10940.ckpt:"
  "DC-AE_Skip_3yr_12h_fibo:${WTI_ROOT}/logs/exp_12h_oddskip_dcae_2017_18_19/epoch=9-step=10940.ckpt:"
  "FuXi_3yr_12h_fibo:${WTI_ROOT}/logs/exp_12h_oddskip_fuxi_2017_18_19/epoch=9-step=5470.ckpt:"
  "ModAFNO_3yr_12h_fibo:${WTI_ROOT}/logs/exp_12h_oddskip_modafno_full_2017_18_19/epoch=9-step=21870.ckpt:MODAFNO_INP_H=362 MODAFNO_INP_W=720 MODAFNO_NATIVE_H=360 MODAFNO_NATIVE_W=720"
  "SDyff_3yr_12h_fibo:${WTI_ROOT}/logs/exp_12h_oddskip_sdyff_dyff_0p5_2017_18_19/epoch=9-step=5470.ckpt:SDYFF_NLAT=360 SDYFF_NLON=720 SDYFF_LAT_CROP=0"
  "WeatherDCAE_NoSkip_6yr_12h:${WTI_ROOT}/logs/exp_12h_oddskip_dcae_noskip_cloudru_2014_19/last.ckpt:"
)

# The 12 h leaderboard has 7 models. WeatherDCAE 6yr lives on cloud.ru and
# must be staged onto the execution host first. Missing checkpoints abort the
# run so a partial model set cannot be mistaken for a complete comparison.

echo "[run_region_season_12h] $(date -u) starting"
echo "  memmap: ${MEMMAP}"
echo "  out:    ${OUT_DIR}"

for entry in "${MODELS[@]}"; do
  name="${entry%%:*}"
  rest="${entry#*:}"
  ckpt="${rest%%:*}"
  envs="${rest#*:}"
  log="${LOG_DIR}/${name}.log"
  echo ""
  echo "=== ${name} ==="
  echo "  ckpt:  ${ckpt}"
  echo "  envs:  ${envs:-(none)}"
  echo "  log:   ${log}"
  python tools/eval/region_season_12h_eval.py \
    --memmap-dir "${MEMMAP}" \
    --test-year 2020 \
    --models "${name}:${ckpt}:${envs}" \
    --out-dir "${OUT_DIR}" \
    --eval-hours "4,6,8" \
    --max-tau-hours 12 \
    --samples-per-date 2 \
    --eval-days-per-month 8 \
    --keep-n-channels 24 \
    --batch-size 4 \
    --num-workers 2 \
    2>&1 | tee "${log}"
done

echo ""
echo "[run_region_season_12h] $(date -u) DONE. JSONs in ${OUT_DIR}"
echo "  Next: python tools/eval/build_region_season_12h_table.py"
