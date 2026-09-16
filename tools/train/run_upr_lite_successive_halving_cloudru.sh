#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."

LOG_ROOT="${LOG_ROOT:-/home/jovyan/shares/SR006.nfs2/dsuhoi/capmatched_logs}"
PYTHON_BIN="${PYTHON_BIN:-/home/jovyan/.mlspace/envs/ai_scientist/bin/python}"
SLEEP_SEC="${SLEEP_SEC:-60}"
SCREEN_JSON="${SCREEN_JSON:-metrics/upr_lite_screen/two_epoch_screen.json}"
LOG="$LOG_ROOT/upr_lite_successive_halving.log"

declare -A EXPERIMENTS=(
  [upr_lite]="exp_upr_lite_24ch_6h_2014_19_sparse135_lr1e4_gwarp_b8_s202707"
  [upr_lite_lap]="exp_upr_lite_lap_24ch_6h_2014_19_sparse135_lr1e4_gwarp_b8_s202707"
  [upr_lite_column]="exp_upr_lite_column_24ch_6h_2014_19_sparse135_lr1e4_gwarp_b8_s202707"
  [upr_lite_continuous]="exp_upr_lite_continuous_24ch_6h_2014_19_sparse135_lr1e4_gwarp_b8_s202707"
  [upr_lite_continuous_m]="exp_upr_lite_continuous_m_24ch_6h_2014_19_sparse135_lr1e4_b8_s202707"
  [upr_lite_implicit_global]="exp_upr_lite_implicit_global_24ch_6h_2014_19_sparse135_lr1e4_b8_s202707"
)

mkdir -p "$LOG_ROOT" "$(dirname "$SCREEN_JSON")"
while true; do
  "$PYTHON_BIN" tools/eval/select_upr_lite_two_epoch_screen.py \
    --log-root "$LOG_ROOT" \
    --output "$SCREEN_JSON" \
    --allow-incomplete >>"$LOG" 2>&1
  complete="$("$PYTHON_BIN" -c '
import json, sys
print(int(json.load(open(sys.argv[1]))["complete"]))
' "$SCREEN_JSON")"
  if [[ "$complete" -eq 1 ]]; then
    break
  fi
  sleep "$SLEEP_SEC"
done

mapfile -t selected < <("$PYTHON_BIN" -c '
import json, sys
print(*json.load(open(sys.argv[1]))["selected"], sep="\n")
' "$SCREEN_JSON")
if [[ "${#selected[@]}" -ne 3 ]]; then
  echo "expected three selected arms, got ${#selected[@]}" >&2
  exit 2
fi

echo "[$(date -Is)] selected=${selected[*]}" | tee -a "$LOG"
pids=()
for arch in "${selected[@]}"; do
  ARCH="$arch" \
  EXP_NAME="${EXPERIMENTS[$arch]}" \
  SLEEP_SEC="$SLEEP_SEC" \
    bash tools/train/run_upr_lite_variant_6h_cloudru.sh >>"$LOG" 2>&1 &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  wait "$pid" || status=1
done
if [[ "$status" -ne 0 ]]; then
  echo "one or more selected-arm queues failed" >&2
  exit "$status"
fi
echo "[$(date -Is)] selected arms complete" | tee -a "$LOG"
