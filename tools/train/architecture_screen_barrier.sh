#!/usr/bin/env bash

architecture_screen_experiments=(
  exp_upr_query_match_14m_hf_135_14m_6h_s202707_protocol_v1
  exp_upr_spherical_implicit_global_14m_24ch_6h_2014_19_sparse135_lr1e4_eb16_s202707
  exp_flow_spherical_ep_14m_6h_s202707
  exp_weather_amt_l_14m_6h_s202707_protocol_v4
)

release_candidate_gpu() {
  if [[ -n "${lock_fd:-}" ]]; then
    flock -u "$lock_fd"
    exec {lock_fd}>&-
  fi
  gpu=""
  lock_fd=""
}

acquire_candidate_gpu() {
  while [[ -z "${gpu:-}" ]]; do
    local candidate
    for candidate in $GPU_CANDIDATES; do
      local candidate_fd
      exec {candidate_fd}>"$LOG_ROOT/.upr_lite_gpu${candidate}.lock"
      if ! flock -n "$candidate_fd"; then
        exec {candidate_fd}>&-
        continue
      fi
      local free_mib
      local util
      free_mib="$(nvidia-smi --query-gpu=memory.free \
        --format=csv,noheader,nounits -i "$candidate" | tr -d ' ')"
      util="$(nvidia-smi --query-gpu=utilization.gpu \
        --format=csv,noheader,nounits -i "$candidate" | tr -d ' ')"
      if [[ "$free_mib" -ge "$MIN_FREE_MIB" && "$util" -le "$MAX_UTIL" ]]; then
        sleep "$GPU_STABLE_SEC"
        free_mib="$(nvidia-smi --query-gpu=memory.free \
          --format=csv,noheader,nounits -i "$candidate" | tr -d ' ')"
        util="$(nvidia-smi --query-gpu=utilization.gpu \
          --format=csv,noheader,nounits -i "$candidate" | tr -d ' ')"
        if [[ "$free_mib" -ge "$MIN_FREE_MIB" && \
              "$util" -le "$MAX_UTIL" ]]; then
          gpu="$candidate"
          lock_fd="$candidate_fd"
          break
        fi
      fi
      flock -u "$candidate_fd"
      exec {candidate_fd}>&-
    done
    if [[ -z "${gpu:-}" ]]; then
      sleep "$SLEEP_SEC"
    fi
  done
}

wait_for_architecture_screen_barrier() {
  local current_experiment="$1"
  local barrier_log="$2"
  touch "$LOG_ROOT/${current_experiment}.screened"
  release_candidate_gpu
  local experiment
  for experiment in "${architecture_screen_experiments[@]}"; do
    while [[ ! -e "$LOG_ROOT/${experiment}.screened" && \
             ! -e "$LOG_ROOT/${experiment}.terminal" ]]; do
      echo "[$(date -Is)] wait architecture screen=$experiment" \
        | tee -a "$barrier_log"
      sleep "$SLEEP_SEC"
    done
  done
  acquire_candidate_gpu
}

wait_for_architecture_completion() {
  local barrier_log="$1"
  local experiment
  for experiment in "${architecture_screen_experiments[@]}"; do
    while [[ ! -e "$LOG_ROOT/${experiment}.terminal" ]]; do
      echo "[$(date -Is)] yield objective arm to architecture=$experiment" \
        | tee -a "$barrier_log"
      sleep "$SLEEP_SEC"
    done
  done
}
