#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."

OLD_PGID="${OLD_PGID:?set OLD_PGID to the running pre-barrier process group}"
LOG_ROOT="${LOG_ROOT:-/home/jovyan/shares/SR006.nfs2/dsuhoi/capmatched_logs}"
SLEEP_SEC="${SLEEP_SEC:-15}"
exp_name="exp_upr_query_match_14m_hf_135_14m_6h_s202707_protocol_v1"
screened_marker="$LOG_ROOT/${exp_name}.screened"
terminal_marker="$LOG_ROOT/${exp_name}.terminal"
relaunch_log="$LOG_ROOT/query_match_arch_barrier.nohup.log"

pgid_active() {
  ps -eo pgid= | awk -v expected="$OLD_PGID" '
    $1 == expected { found = 1 }
    END { exit !found }
  '
}

while [[ ! -e "$screened_marker" && ! -e "$terminal_marker" ]]; do
  echo "[$(date -Is)] wait QueryMatch screen"
  sleep "$SLEEP_SEC"
done

# The old runner writes screened immediately before its promotion decision.
# Give a rejected candidate time to write terminal and exit on its own.
sleep 3
if [[ -e "$terminal_marker" ]]; then
  echo "[$(date -Is)] QueryMatch terminal before scheduler switch"
  exit 0
fi

echo "[$(date -Is)] QueryMatch screened; stop old pgid=$OLD_PGID"
kill -TERM -- "-$OLD_PGID" 2>/dev/null || true
for _ in $(seq 1 24); do
  if ! pgid_active; then
    break
  fi
  sleep 5
done
if pgid_active; then
  echo "[$(date -Is)] old QueryMatch group did not stop" >&2
  exit 2
fi

if [[ -e "$terminal_marker" ]]; then
  mv "$terminal_marker" \
    "${terminal_marker}.scheduler_stop_$(date +%Y%m%dT%H%M%S)"
fi

nohup nice -n 10 \
  bash tools/train/run_upr_query_match_14m_6h_pilot_cloudru.sh \
  >"$relaunch_log" 2>&1 &
echo "[$(date -Is)] relaunched patched QueryMatch pid=$!"
