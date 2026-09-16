#!/usr/bin/env bash
set -euo pipefail

SOURCE="${SOURCE:-/home/jovyan/shares/SR006.nfs2/dsuhoi/weather_time_interpolation_spectral_v6}"
LOG_ROOT="${LOG_ROOT:-/home/jovyan/shares/SR006.nfs2/dsuhoi/capmatched_logs}"
PY="${PY:-/home/jovyan/.mlspace/envs/ai_scientist/bin/python}"
WEATHERBRIDGE_CKPT="${WEATHERBRIDGE_CKPT:-$LOG_ROOT/exp_flow_pp3_spectral_14m_6h_refinev1_s202707_bs4/last.ckpt}"
DCAE_CKPT="${DCAE_CKPT:-$LOG_ROOT/exp_weatherdcae_14m_6h_6yr_refinev1_s202707_bs4/last.ckpt}"
PIXELATTN_CKPT="${PIXELATTN_CKPT:-$LOG_ROOT/exp_atmvfi_24ch_6h_2014_19_sparse135_lr1e4_matched/last.ckpt}"

cd "$SOURCE"
export PYTHONPATH="$SOURCE:/home/jovyan/shares/SR006.nfs2/dsuhoi/pylibs_mamba:/home/jovyan/shares/SR006.nfs2/dsuhoi/wti_shim"

run_export() {
  local checkpoint="$1"
  local label="$2"
  local output_root="$3"
  "$PY" tools/eval/export_ciara_wind_comparator_case.py \
    --checkpoint "$checkpoint" \
    --model-label "$label" \
    --output-root "$output_root" \
    --device cuda:0
}

run_export "$WEATHERBRIDGE_CKPT" WeatherBridge metrics/case_studies_weatherbridge
run_export "$DCAE_CKPT" WeatherDCAE-14M metrics/case_studies_weatherdcae_14m
run_export "$PIXELATTN_CKPT" PixelAttn-VFI metrics/case_studies_pixelattn_vfi
