#!/usr/bin/env bash
# Produce paired temporal uncertainty estimates after canonical evaluations finish.
set -euo pipefail

RUNTIME=${RUNTIME:-/home/jovyan/shares/SR006.nfs2/dsuhoi/weather_time_interpolation_runtime}
PY=${PY:-/home/jovyan/.mlspace/envs/ai_scientist/bin/python}
YEARS_6=${YEARS_6:-"2020 2021"}
YEARS_12=${YEARS_12:-"2020"}
POLL_SECONDS=${POLL_SECONDS:-120}

cd "$RUNTIME"
mkdir -p metrics/journal_unified/stats logs/runner
LOG=logs/runner/journal_stats_queue.log

wait_file() {
  local path="$1"
  while [[ ! -s "$path" ]]; do
    echo "[$(date -Is)] wait for $path" | tee -a "$LOG"
    sleep "$POLL_SECONDS"
  done
}

compare_with_linear() {
  local horizon="$1"
  local year="$2"
  local name="$3"
  local npz="metrics/journal_unified/${horizon}h_${year}/window_metrics/${name}.npz"
  local out="metrics/journal_unified/stats/${name}_vs_linear_${horizon}h_${year}.json"
  wait_file "$npz"
  if [[ -s "$out" ]]; then
    return
  fi
  "$PY" tools/eval/paired_block_bootstrap.py \
    --left "${name}:${npz}:model" \
    --right "linear:${npz}:bilinear" \
    --block-days 7 --draws 5000 --seed 2027 \
    --out-json "$out" | tee -a "$LOG"
}

compare_weatherbridge() {
  local horizon="$1"
  local year="$2"
  local bridge="$3"
  shift 3
  local root="metrics/journal_unified/${horizon}h_${year}/window_metrics"
  local bridge_npz="$root/${bridge}.npz"
  local out="metrics/journal_unified/stats/${bridge}_pairwise_${horizon}h_${year}.json"
  local args=()
  wait_file "$bridge_npz"
  for name in "$@"; do
    local npz="$root/${name}.npz"
    wait_file "$npz"
    args+=(--right "${name}:${npz}:model")
  done
  if [[ -s "$out" ]]; then
    return
  fi
  "$PY" tools/eval/paired_block_bootstrap.py \
    --left "${bridge}:${bridge_npz}:model" \
    "${args[@]}" \
    --block-days 7 --draws 5000 --seed 2027 \
    --out-json "$out" | tee -a "$LOG"
}

compare_pair() {
  local horizon="$1"
  local year="$2"
  local left="$3"
  local right="$4"
  local root="metrics/journal_unified/${horizon}h_${year}/window_metrics"
  local left_npz="$root/${left}.npz"
  local right_npz="$root/${right}.npz"
  local out="metrics/journal_unified/stats/${left}_vs_${right}_${horizon}h_${year}.json"
  wait_file "$left_npz"
  wait_file "$right_npz"
  if [[ -s "$out" ]]; then
    return
  fi
  "$PY" tools/eval/paired_block_bootstrap.py \
    --left "${left}:${left_npz}:model" \
    --right "${right}:${right_npz}:model" \
    --block-days 7 --draws 5000 --seed 2027 \
    --out-json "$out" | tee -a "$LOG"
}

for year in $YEARS_6; do
  six_hour_models=(
    fuxi_24ch_6yr_ep8
    modafno_24ch_6yr_ep8
    sdyff_24ch_6yr_ep8
    weatherdcae_14m_3yr_ep8_matched
    weatherdcae_skip_14m_3yr_ep8_matched
    atm_vfi_6yr_ep8_matched
    weatherbridge_pp3_14m_6yr_ep8
  )
  for model in "${six_hour_models[@]}"; do
    compare_with_linear 6 "$year" "$model"
  done
  compare_weatherbridge 6 "$year" weatherbridge_pp3_14m_6yr_ep8 \
    fuxi_24ch_6yr_ep8 \
    modafno_24ch_6yr_ep8 \
    sdyff_24ch_6yr_ep8 \
    weatherdcae_14m_3yr_ep8_matched \
    weatherdcae_skip_14m_3yr_ep8_matched \
    atm_vfi_6yr_ep8_matched
  compare_pair 6 "$year" \
    weatherdcae_skip_14m_3yr_ep8_matched \
    weatherdcae_14m_3yr_ep8_matched

done

for year in $YEARS_12; do
  twelve_hour_models=(
    fuxi_3yr_ep10
    modafno_3yr_ep10
    sdyff_3yr_ep10
    weatherdcae_14m_3yr_ep10_matched
    atm_vfi_3yr_ep10_matched
    weatherbridge_14m_3yr_ep10
  )
  for model in "${twelve_hour_models[@]}"; do
    compare_with_linear 12 "$year" "$model"
  done
  compare_weatherbridge 12 "$year" weatherbridge_14m_3yr_ep10 \
    fuxi_3yr_ep10 \
    modafno_3yr_ep10 \
    sdyff_3yr_ep10 \
    weatherdcae_14m_3yr_ep10_matched \
    atm_vfi_3yr_ep10_matched
done
