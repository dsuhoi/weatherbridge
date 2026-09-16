#!/usr/bin/env bash
# Train the matched Detail seed used by full-year model selection.
set -euo pipefail

SOURCE=${SOURCE:-/home/jovyan/shares/SR006.nfs2/dsuhoi/weather_time_interpolation_spectral_v6}
LOG_ROOT=${LOG_ROOT:-/home/jovyan/shares/SR006.nfs2/dsuhoi/capmatched_logs}
POLL_SECONDS=${POLL_SECONDS:-120}
PRIMARY_STATES=(
  "$LOG_ROOT/npj_seed_replicates_v2.state"
  "$LOG_ROOT/npj_seed_replicates_v1.state"
)
LOG="$LOG_ROOT/npj_detail_reference_v1.wrapper.log"
WRAPPER_STATE="$LOG_ROOT/npj_detail_reference_v1.wrapper.state"

cd "$SOURCE"
mkdir -p "$WRAPPER_STATE"
exec 8>"$LOG_ROOT/.npj_detail_reference_v1.wrapper.lock"
if ! flock -n 8; then
  echo "[$(date -Is)] Detail reference wrapper already active" | tee -a "$LOG"
  exit 0
fi

SOURCE_SHA=$(sha256sum \
  scripts/run_detail_reference_queue_cloudru.sh \
  tools/train/run_npj_seed_replicates_cloudru.sh | sha256sum | awk '{print $1}')
if [[ -n "${EXPECTED_SOURCE_SHA:-}" && "$SOURCE_SHA" != "$EXPECTED_SOURCE_SHA" ]]; then
  echo "source hash mismatch: $SOURCE_SHA != $EXPECTED_SOURCE_SHA" | tee -a "$LOG"
  exit 2
fi
printf '%s\n' "$SOURCE_SHA" >"$WRAPPER_STATE/source.sha256"

wait_for_horizon() {
  local horizon="$1"
  shift
  local required=() family seed id missing found state
  for family in "$@"; do
    for seed in 202707 202708 202709; do
      required+=("${family}_${horizon}_${seed}")
    done
  done
  while true; do
    missing=0
    for id in "${required[@]}"; do
      found=0
      for state in "${PRIMARY_STATES[@]}"; do
        if [[ -e "$state/$id.done" ]]; then
          found=1
          break
        fi
      done
      if [[ "$found" -eq 0 ]]; then
        missing=1
        break
      fi
    done
    [[ "$missing" -eq 0 ]] && return 0
    echo "[$(date -Is)] waiting for primary ${horizon}h matched seeds" | tee -a "$LOG"
    sleep "$POLL_SECONDS"
  done
}

wait_for_horizon 6 refine flow_spectral
JOBS_OVERRIDE="detail:6:202707" \
WORKER_GPUS="${DETAIL_WORKER_GPUS:-1}" \
STATE_DIR="$LOG_ROOT/npj_detail_reference_v1_6h.state" \
QUEUE_LOG="$LOG_ROOT/npj_detail_reference_v1_6h.queue.log" \
EVAL_ROOT="/home/jovyan/shares/SR006.nfs2/dsuhoi/weather_time_interpolation_runtime/metrics/npj_seed_ifs_hres_2021_v3" \
  bash tools/train/run_npj_seed_replicates_cloudru.sh
test -s "$LOG_ROOT/exp_flow_pp3_detail_14m_6h_refinev1_s202707_bs4/last.ckpt"
test -e "$LOG_ROOT/npj_detail_reference_v1_6h.state/detail_6_202707.done"

wait_for_horizon 12 refine flow_spectral dcae
JOBS_OVERRIDE="detail:12:202707" \
WORKER_GPUS="${DETAIL_WORKER_GPUS:-1}" \
STATE_DIR="$LOG_ROOT/npj_detail_reference_v1_12h.state" \
QUEUE_LOG="$LOG_ROOT/npj_detail_reference_v1_12h.queue.log" \
EVAL_ROOT="/home/jovyan/shares/SR006.nfs2/dsuhoi/weather_time_interpolation_runtime/metrics/npj_seed_ifs_hres_2021_v3" \
  bash tools/train/run_npj_seed_replicates_cloudru.sh
test -s "$LOG_ROOT/exp_flow_pp3_detail_14m_12h_2017_19_refinev1_s202707_bs4/last.ckpt"
test -e "$LOG_ROOT/npj_detail_reference_v1_12h.state/detail_12_202707.done"
