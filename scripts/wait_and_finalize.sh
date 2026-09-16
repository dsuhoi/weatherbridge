#!/usr/bin/env bash
# Watches Phase B refresh log; once it reports "Phase B finished", runs
# finalize_s3_refresh.sh (validate + cutover legacy ↔ canonical).
#
# Run with: nohup bash scripts/wait_and_finalize.sh > logs/runner/wait_and_finalize.log 2>&1 &
set -eo pipefail
cd "$(dirname "$0")/.."

REFRESH_LOG=logs/runner/refresh_s3.log
FINAL_LOG=logs/runner/finalize_s3.log

echo "[wait_and_finalize] start $(date), watching $REFRESH_LOG"

while true; do
  if grep -q "Phase B finished" "$REFRESH_LOG" 2>/dev/null; then
    echo "[wait_and_finalize] detected 'Phase B finished' at $(date), running finalize..."
    bash scripts/finalize_s3_refresh.sh > "$FINAL_LOG" 2>&1
    rc=$?
    echo "[wait_and_finalize] finalize exit code: $rc"
    if [ "$rc" -eq 0 ]; then
      echo "[wait_and_finalize] cutover SUCCESS at $(date)"
    else
      echo "[wait_and_finalize] cutover FAILED, leaving v2/legacy paths intact (manual fix required)"
    fi
    exit $rc
  fi
  if grep -qE "FATAL|Traceback|RuntimeError" "$REFRESH_LOG" 2>/dev/null; then
    echo "[wait_and_finalize] Phase B reported error — aborting watch:"
    grep -E "FATAL|Traceback|RuntimeError" "$REFRESH_LOG" | head -5
    exit 4
  fi
  sleep 300
done
