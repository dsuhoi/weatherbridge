#!/usr/bin/env bash
# Fill the 2x2 architecture x objective comparison on the primary 6 h task.
set -euo pipefail

SOURCE=${SOURCE:-/home/jovyan/shares/SR006.nfs2/dsuhoi/weather_time_interpolation_spectral_v6}
RUNTIME=${RUNTIME:-/home/jovyan/shares/SR006.nfs2/dsuhoi/weather_time_interpolation_runtime}
COMPONENT_MARKER=${COMPONENT_MARKER:-$RUNTIME/metrics/weatherbridge_component_ablation_v1/state/.complete}
POLL_SECONDS=${POLL_SECONDS:-120}

while [[ ! -s "$COMPONENT_MARKER" ]]; do
  echo "[$(date -Is)] wait WeatherBridge component evaluation"
  sleep "$POLL_SECONDS"
done

cd "$SOURCE"
export RUN_IFS=0
PRIMARY_STATE_DIR=/home/jovyan/shares/SR006.nfs2/dsuhoi/capmatched_logs/loss_architecture_factorial_v1.state
RECOVERY_STATE_DIR=/home/jovyan/shares/SR006.nfs2/dsuhoi/capmatched_logs/loss_architecture_factorial_v1_recovery.state
export QUEUE_LOG=/home/jovyan/shares/SR006.nfs2/dsuhoi/capmatched_logs/loss_architecture_factorial_v1.queue.log

recover_flow_pixel_seed() {
  export JOBS_OVERRIDE="flow_pixel:6:202707"
  export STATE_DIR="$RECOVERY_STATE_DIR"
  export LAUNCH_LOCK="$STATE_DIR/launch.lock"
  mkdir -p "$STATE_DIR"
  rm -f "$STATE_DIR/flow_pixel_6_202707.failed" \
    "$STATE_DIR/flow_pixel_6_202707.claimed"
  bash tools/train/run_npj_seed_replicates_cloudru.sh
  touch "$PRIMARY_STATE_DIR/flow_pixel_6_202707.done"
  rm -f "$PRIMARY_STATE_DIR/flow_pixel_6_202707.failed"
}

if [[ "${RECOVER_ONLY:-0}" == 1 ]]; then
  recover_flow_pixel_seed
  exit 0
fi

export JOBS_OVERRIDE="flow_pixel:6:202707 flow_pixel:6:202708 flow_pixel:6:202709 dcae_spectral:6:202707 dcae_spectral:6:202708 dcae_spectral:6:202709"
export STATE_DIR="$PRIMARY_STATE_DIR"
export LAUNCH_LOCK="$STATE_DIR/launch.lock"
set +e
bash tools/train/run_npj_seed_replicates_cloudru.sh
queue_status=$?
set -e

if [[ -f "$PRIMARY_STATE_DIR/flow_pixel_6_202707.failed" \
      && ! -f "$PRIMARY_STATE_DIR/flow_pixel_6_202707.done" ]]; then
  recover_flow_pixel_seed
fi

failed_count="$(find "$PRIMARY_STATE_DIR" -maxdepth 1 -type f -name '*.failed' | wc -l)"
done_count="$(find "$PRIMARY_STATE_DIR" -maxdepth 1 -type f -name '*.done' | wc -l)"
if [[ "$failed_count" -ne 0 || "$done_count" -ne 6 ]]; then
  exit 1
fi
