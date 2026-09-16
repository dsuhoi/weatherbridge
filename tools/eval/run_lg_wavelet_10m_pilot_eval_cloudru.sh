#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."

LOG_ROOT="${LOG_ROOT:-/home/jovyan/shares/SR006.nfs2/dsuhoi/capmatched_logs}"
METRICS_ROOT="${METRICS_ROOT:-/home/jovyan/shares/SR006.nfs2/dsuhoi/weather_time_interpolation_runtime/metrics/lg_wavelet_10m_pilot_6h_2020}"
PYTHON_BIN="${PYTHON_BIN:-/home/jovyan/.mlspace/envs/ai_scientist/bin/python}"
EXTRA_PYTHONPATH="${EXTRA_PYTHONPATH:-/home/jovyan/shares/SR006.nfs2/dsuhoi/pylibs_mamba:/home/jovyan/shares/SR006.nfs2/dsuhoi/wti_shim}"
DATA_ROOT="${DATA_ROOT:-/home/jovyan/dsuhoi/weather_time_interpolation/data}"
CLIMATOLOGY="${CLIMATOLOGY:-/workspace-SR006.nfs2/weather_data/time_interpolation_0p5/climatology_1990-2019_0p5_canonical_v2.zarr}"
CLIMATOLOGY_CACHE="${CLIMATOLOGY_CACHE:-/home/jovyan/shares/SR006.nfs2/dsuhoi/climatology_eval_cache_24ch}"
GPU="${GPU:-1}"
MIN_FREE_MIB="${MIN_FREE_MIB:-65000}"
MAX_UTIL="${MAX_UTIL:-5}"
SLEEP_SEC="${SLEEP_SEC:-60}"
export PYTHONPATH="$EXTRA_PYTHONPATH:$PWD:${PYTHONPATH:-}"

EXP_NAME="exp_lg_wavelet_10m_6h_s202707_pilot2ep_bs4"
TRAIN_COMPLETE="$LOG_ROOT/$EXP_NAME.complete"
CANDIDATE="$LOG_ROOT/$EXP_NAME/last.ckpt"
FLOW_STANDARD="$LOG_ROOT/exp_flow_pp3_135_14m_6h_s202707_protocol_v2/epoch=1-step=3284.ckpt"
FLOW_SPECTRAL="$LOG_ROOT/exp_flow_pp3_spectral_14m_6h_s202707_v1_bs4/epoch=1-step=3284.ckpt"
QUEUE_LOG="$LOG_ROOT/$EXP_NAME.eval.queue.log"
COMPLETE_MARKER="$LOG_ROOT/$EXP_NAME.eval.complete"
TERMINAL_MARKER="$LOG_ROOT/$EXP_NAME.eval.terminal"

TRAINER_SHA256="b11ace95c7320f3ce87c0949b55d84d789d99ff3cbda93eb300362e7c765bca1"
MODEL_SHA256="1094b424af4bc1cceb9710e6ce9bc19c7f3adea16c217f12a88ed498f35a7e82"

mkdir -p "$LOG_ROOT" "$METRICS_ROOT" "$CLIMATOLOGY_CACHE"
exec 9>"$LOG_ROOT/$EXP_NAME.eval.queue.lock"
if ! flock -n 9; then
  echo "[$(date -Is)] $EXP_NAME evaluation already active" \
    | tee -a "$QUEUE_LOG"
  exit 0
fi
rm -f "$COMPLETE_MARKER" "$TERMINAL_MARKER"
trap 'touch "$TERMINAL_MARKER"' ERR

actual_trainer_sha256="$(
  sha256sum tools/train/train_capacity_matched_6h.py | awk '{print $1}'
)"
actual_model_sha256="$(
  sha256sum weather_time_interp/model/local_global_wavelet_bridge_model.py |
    awk '{print $1}'
)"
if [[ "$actual_trainer_sha256" != "$TRAINER_SHA256" ||
      "$actual_model_sha256" != "$MODEL_SHA256" ]]; then
  echo "evaluation source SHA-256 mismatch" >&2
  exit 2
fi

while [[ ! -e "$TRAIN_COMPLETE" ]]; do
  echo "[$(date -Is)] wait $EXP_NAME two-epoch checkpoint" \
    | tee -a "$QUEUE_LOG"
  sleep "$SLEEP_SEC"
done

"$PYTHON_BIN" tools/train/checkpoint_status.py \
  "$CANDIDATE" \
  --min-epochs 2 \
  --expected-arch lg_wavelet_10m \
  --expected-total-steps 13136 \
  --min-global-step 3284 \
  --expected-trainer-sha256 "$TRAINER_SHA256" \
  --expected-highpass-boundary antipodal_vector_parity \
  --expected-lambda-hf 0.05 \
  --expected-delta-t 6 \
  --require-training-protocol \
  --expected-train-years 2014,2015,2016,2017,2018,2019 \
  --expected-val-years 2020 \
  --expected-train-taus 1,3,5 \
  --expected-eval-taus 1,2,3,4,5 \
  --expected-seed 202707 \
  --expected-effective-batch-size 16 \
  --quiet

for checkpoint in "$FLOW_STANDARD" "$FLOW_SPECTRAL"; do
  if [[ ! -s "$checkpoint" ]]; then
    echo "missing frozen Flow reference: $checkpoint" >&2
    exit 2
  fi
done

while true; do
  free_mib="$(nvidia-smi --query-gpu=memory.free \
    --format=csv,noheader,nounits -i "$GPU" | tr -d ' ')"
  util="$(nvidia-smi --query-gpu=utilization.gpu \
    --format=csv,noheader,nounits -i "$GPU" | tr -d ' ')"
  if [[ "$free_mib" -ge "$MIN_FREE_MIB" && "$util" -le "$MAX_UTIL" ]]; then
    break
  fi
  sleep "$SLEEP_SEC"
done

export CUDA_VISIBLE_DEVICES="$GPU"
echo "[$(date -Is)] evaluate $EXP_NAME gpu=$GPU" | tee -a "$QUEUE_LOG"

RMSE_DIR="$METRICS_ROOT/rmse_acc"
SH_DIR="$METRICS_ROOT/spherical_spectra"
mkdir -p "$RMSE_DIR" "$SH_DIR"

"$PYTHON_BIN" -u tools/eval/batch_eval_12h_memmap.py \
  --memmap-dir /tmp/wb2_0p5_cache \
  --test-year 2020 \
  --climatology "$CLIMATOLOGY" \
  --climatology-cache-dir "$CLIMATOLOGY_CACHE" \
  --lazy-climatology \
  --stats-path "$DATA_ROOT/json_stats_0p5.nc" \
  --surface-stats-path "$DATA_ROOT/surface_stats_0p5.json" \
  --static-path "$DATA_ROOT/static_features_0p5.pt" \
  --models "lg_wavelet_10m:$CANDIDATE:,flow_2ep:$FLOW_STANDARD:,flow_spectral_2ep:$FLOW_SPECTRAL:" \
  --out-dir "$RMSE_DIR" \
  --paper-tag "lg_wavelet_10m_pilot_6h_2020" \
  --batch-size 2 \
  --num-workers 2 \
  --samples-per-date 2 \
  --eval-days-per-month 8 \
  --max-tau-hours 6 \
  --eval-hours 1,2,3,4,5 \
  --seen-tau 1,3,5 \
  --unseen-tau 2,4 \
  --keep-n-channels 24 \
  --proper-rmse \
  --save-window-metrics \
  2>&1 | tee -a "$QUEUE_LOG"

for model_spec in \
  "lg_wavelet_10m:$CANDIDATE" \
  "flow_2ep:$FLOW_STANDARD" \
  "flow_spectral_2ep:$FLOW_SPECTRAL"; do
  name="${model_spec%%:*}"
  checkpoint="${model_spec#*:}"
  "$PYTHON_BIN" -u tools/eval/sh_energy_spectra_12h.py \
    --memmap-dir /tmp/wb2_0p5_cache \
    --test-year 2020 \
    --stats-path "$DATA_ROOT/json_stats_0p5.nc" \
    --surface-stats-path "$DATA_ROOT/surface_stats_0p5.json" \
    --static-path "$DATA_ROOT/static_features_0p5.pt" \
    --ckpt "$checkpoint" \
    --model-name "$name" \
    --model-kind capmatched \
    --out-dir "$SH_DIR" \
    --taus 1,2,3,4,5 \
    --channels all \
    --lmax 180 \
    --keep-n-channels 24 \
    --batch-size 2 \
    --samples-per-date 2 \
    --eval-days-per-month 2 \
    --hf-ell-min 90 \
    --max-tau-hours 6 \
    --skip-existing \
    2>&1 | tee -a "$QUEUE_LOG"
done

for reference in flow_2ep flow_spectral_2ep; do
  "$PYTHON_BIN" tools/eval/paired_block_bootstrap.py \
    --left "lg_wavelet_10m:$RMSE_DIR/window_metrics/lg_wavelet_10m.npz" \
    --right "$reference:$RMSE_DIR/window_metrics/$reference.npz" \
    --taus 1,2,3,4,5 \
    --block-days 7 \
    --draws 5000 \
    --seed 2027 \
    --out-json "$RMSE_DIR/paired_lg_wavelet_vs_${reference}.json" \
    2>&1 | tee -a "$QUEUE_LOG"

  for tau in 1 2 3 4 5; do
    "$PYTHON_BIN" tools/eval/spectral_block_bootstrap.py \
      --left "lg_wavelet_10m:$SH_DIR/lg_wavelet_10m_tau${tau}.npz" \
      --right "$reference:$SH_DIR/${reference}_tau${tau}.npz" \
      --channels all \
      --block-days 7 \
      --draws 5000 \
      --seed 2027 \
      --out-json "$SH_DIR/paired_lg_wavelet_vs_${reference}_tau${tau}.json" \
      2>&1 | tee -a "$QUEUE_LOG"
  done
done

touch "$COMPLETE_MARKER"
trap - ERR
echo "[$(date -Is)] $EXP_NAME evaluation complete" | tee -a "$QUEUE_LOG"
