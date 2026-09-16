#!/usr/bin/env bash
set -euo pipefail

SOURCE="${SOURCE:-/home/jovyan/shares/SR006.nfs2/dsuhoi/weather_time_interpolation_spectral_v6}"
LOG_ROOT="${LOG_ROOT:-/home/jovyan/shares/SR006.nfs2/dsuhoi/capmatched_logs}"
PY="${PY:-/home/jovyan/.mlspace/envs/ai_scientist/bin/python}"
POLL_SECONDS="${POLL_SECONDS:-60}"
GPU_CANDIDATES="${GPU_CANDIDATES:-0 1}"
WEATHERBRIDGE_CKPT="${WEATHERBRIDGE_CKPT:-$LOG_ROOT/exp_flow_pp3_spectral_14m_6h_refinev1_s202707_bs4/last.ckpt}"
DCAE_CKPT="${DCAE_CKPT:-$LOG_ROOT/exp_weatherdcae_14m_6h_6yr_refinev1_s202707_bs4/last.ckpt}"
PIXELATTN_CKPT="${PIXELATTN_CKPT:-$LOG_ROOT/exp_atmvfi_24ch_6h_2014_19_sparse135_lr1e4_matched/last.ckpt}"
LOG="$LOG_ROOT/haishen_comparator_cases.log"
MARKER="$SOURCE/metrics/case_studies_weatherbridge/typhoon_haishen/.matched_cases_complete"

cd "$SOURCE"
export PYTHONPATH="$SOURCE:/home/jovyan/shares/SR006.nfs2/dsuhoi/pylibs_mamba:/home/jovyan/shares/SR006.nfs2/dsuhoi/wti_shim"

for checkpoint in "$WEATHERBRIDGE_CKPT" "$DCAE_CKPT" "$PIXELATTN_CKPT"; do
  test -s "$checkpoint"
done

exec > >(tee -a "$LOG") 2>&1
GPU=""
while [[ -z "$GPU" ]]; do
  for candidate in $GPU_CANDIDATES; do
    exec {lock_fd}>"$LOG_ROOT/.upr_lite_gpu${candidate}.lock"
    if flock -n "$lock_fd"; then
      GPU="$candidate"
      GPU_LOCK_FD="$lock_fd"
      break
    fi
    exec {lock_fd}>&-
  done
  if [[ -z "$GPU" ]]; then
    echo "[$(date -Is)] waiting for a free GPU lock"
    sleep "$POLL_SECONDS"
  fi
done

echo "[$(date -Is)] exporting matched Haishen cases on GPU $GPU"
export CUDA_VISIBLE_DEVICES="$GPU"
"$PY" tools/eval/export_haishen_comparator_cases.py \
  --checkpoint "$WEATHERBRIDGE_CKPT" \
  --model-label WeatherBridge \
  --output-root metrics/case_studies_weatherbridge \
  --device cuda:0
"$PY" tools/eval/export_haishen_comparator_cases.py \
  --checkpoint "$DCAE_CKPT" \
  --model-label WeatherDCAE-14M \
  --output-root metrics/case_studies_weatherdcae_14m \
  --device cuda:0
"$PY" tools/eval/export_haishen_comparator_cases.py \
  --checkpoint "$PIXELATTN_CKPT" \
  --model-label PixelAttn-VFI \
  --output-root metrics/case_studies_pixelattn_vfi \
  --device cuda:0

flock -u "$GPU_LOCK_FD"
exec {GPU_LOCK_FD}>&-
printf 'complete %s\n' "$(date -Is)" > "$MARKER"
echo "[$(date -Is)] complete"
