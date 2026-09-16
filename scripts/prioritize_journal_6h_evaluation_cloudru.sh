#!/usr/bin/env bash
# Pause unstarted 12 h jobs at a claim barrier, run the 6 h decision suite,
# then resume the immutable matched queue. Active jobs are never interrupted.
set -euo pipefail

SOURCE=${SOURCE:-/home/jovyan/shares/SR006.nfs2/dsuhoi/weather_time_interpolation_spectral_v6}
RUNTIME=${RUNTIME:-/home/jovyan/shares/SR006.nfs2/dsuhoi/weather_time_interpolation_runtime}
LOG_ROOT=${LOG_ROOT:-/home/jovyan/shares/SR006.nfs2/dsuhoi/capmatched_logs}
POLL_SECONDS=${POLL_SECONDS:-120}
PY=${PY:-/home/jovyan/.mlspace/envs/ai_scientist/bin/python}
STATE="$LOG_ROOT/npj_seed_replicates_v1.state"
LOG="$LOG_ROOT/prioritize_journal_6h_evaluation.log"
TOKEN="held_for_journal_6h_evaluation_v1"
SPECTRAL_MARKER="$SOURCE/metrics/journal_spectra_v6/state/.complete_6h"
DETAILED_MARKER="$RUNTIME/metrics/detailed_benchmark_v2/6h/.complete"

cd "$SOURCE"
exec >>"$LOG" 2>&1
exec 8>"$LOG_ROOT/.prioritize_journal_6h_evaluation.lock"
if ! flock -n 8; then
  echo "[$(date -Is)] priority wrapper already active"
  exit 0
fi

HOLD_JOBS=(
  flow_spectral_12_202708
  flow_spectral_12_202709
  dcae_12_202707
  dcae_12_202708
  dcae_12_202709
  refine_12_202707
  refine_12_202708
  refine_12_202709
)

exec 7>"$STATE/claims.lock"
flock 7
for id in "${HOLD_JOBS[@]}"; do
  if [[ ! -e "$STATE/$id.done" && ! -e "$STATE/$id.claimed" ]]; then
    printf '%s\n' "$TOKEN" >"$STATE/$id.claimed"
    echo "[$(date -Is)] held job=$id"
  fi
done
flock -u 7

for id in dcae_6_202709 flow_spectral_12_202707; do
  until [[ -e "$STATE/$id.done" ]]; do
    if [[ -e "$STATE/$id.failed" ]]; then
      echo "[$(date -Is)] active job failed: $id"
      exit 2
    fi
    echo "[$(date -Is)] waiting for active job=$id"
    sleep "$POLL_SECONDS"
  done
done

until [[ -e "$DETAILED_MARKER" && -s "$SPECTRAL_MARKER" ]]; do
  echo "[$(date -Is)] waiting for validated 6h evaluation markers"
  sleep "$POLL_SECONDS"
done

expected_spectral_sha=$(awk 'NR == 1 {print $1}' \
  "$SOURCE/metrics/journal_spectra_v6/state/source.sha256")
actual_spectral_sha=$("$PY" - "$SPECTRAL_MARKER" <<'PY'
import json
import sys

payload = json.load(open(sys.argv[1]))
if payload.get("status") != "complete" or payload.get("artifact_count") not in {45, 65}:
    raise SystemExit("invalid 6h spectral marker")
print(payload.get("source_sha256", ""))
PY
)
if [[ "$actual_spectral_sha" != "$expected_spectral_sha" ]]; then
  echo "[$(date -Is)] spectral source hash mismatch"
  exit 2
fi

flock 7
for id in "${HOLD_JOBS[@]}"; do
  marker="$STATE/$id.claimed"
  if [[ -e "$marker" && "$(cat "$marker")" == "$TOKEN" ]]; then
    rm -f "$marker"
    echo "[$(date -Is)] released job=$id"
  fi
done
flock -u 7

# The original queue owns this lock until both active workers exit.
exec 6>"$LOG_ROOT/npj_seed_replicates_v1.launch.lock"
until flock -n 6; do
  echo "[$(date -Is)] waiting for original matched queue to exit"
  sleep "$POLL_SECONDS"
done
flock -u 6
exec 6>&-

nohup bash tools/train/run_npj_seed_replicates_cloudru.sh \
  >>"$LOG_ROOT/npj_seed_replicates_v1.resume.nohup.log" 2>&1 &
echo "[$(date -Is)] resumed matched queue pid=$!"
