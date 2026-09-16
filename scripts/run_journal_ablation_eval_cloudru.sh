#!/usr/bin/env bash
# Full-year paired evaluation for WeatherBridge component ablations.
set -euo pipefail

RUNTIME=${RUNTIME:-/home/jovyan/shares/SR006.nfs2/dsuhoi/weather_time_interpolation_runtime}
LOG_ROOT=${LOG_ROOT:-/home/jovyan/shares/SR006.nfs2/dsuhoi/capmatched_logs}
PY=${PY:-/home/jovyan/.mlspace/envs/ai_scientist/bin/python}
GPU_CANDIDATES=${GPU_CANDIDATES:-"0 1"}
MIN_FREE_MIB=${MIN_FREE_MIB:-60000}
MAX_UTIL=${MAX_UTIL:-5}
POLL_SECONDS=${POLL_SECONDS:-120}
FIELD_MARKER=${FIELD_MARKER:-metrics/journal_unified/.final_eval_complete}

cd "$RUNTIME"
export PYTHONPATH="$RUNTIME:/home/jovyan/shares/SR006.nfs2/dsuhoi/pylibs_mamba:/home/jovyan/shares/SR006.nfs2/dsuhoi/wti_shim"

OUT=metrics/journal_unified/ablation_6h_2020
LOG=logs/runner/journal_ablation_eval_6h.log
MARKER=metrics/journal_unified/.ablation_eval_complete
mkdir -p "$OUT" logs/runner
exec 9>"$LOG_ROOT/.journal_ablation_eval_queue.lock"
if ! flock -n 9; then
  echo "[$(date -Is)] journal ablation-eval queue already active" | tee -a "$LOG"
  exit 0
fi

while ! "$PY" - "$FIELD_MARKER" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
if not path.is_file():
    raise SystemExit(1)
payload = json.loads(path.read_text())
raise SystemExit(0 if payload.get("status") == "complete" else 1)
PY
do
  echo "[$(date -Is)] wait canonical field evaluation" | tee -a "$LOG"
  sleep "$POLL_SECONDS"
done

declare -A CHECKPOINTS=(
  [wb_vanilla]="$LOG_ROOT/exp_wb_vanilla_14m_6h/last.ckpt"
  [wb_skip]="$LOG_ROOT/exp_wb_skip_14m_6h/last.ckpt"
  [flow]="$LOG_ROOT/exp_flow_135v2_14m_6h/last.ckpt"
  [flow_noskip]="$LOG_ROOT/exp_flow_noskip_135_matched_14m_6h/last.ckpt"
  [flow_ungated]="$LOG_ROOT/exp_flow_ungated_135_matched_14m_6h/last.ckpt"
  [flow_accel]="$LOG_ROOT/exp_flow_accel_135_14m_6h/last.ckpt"
  [flow_pp]="$LOG_ROOT/exp_flow_pp_135_14m_6h/last.ckpt"
  [flow_pp2]="$LOG_ROOT/exp_flow_pp2_135_14m_6h/last.ckpt"
  [flow_spectral]="$LOG_ROOT/exp_flow_pp3_spectral_14m_6h_s202707_v1_bs4/last.ckpt"
  [weatherbridge]="$LOG_ROOT/exp_flow_pp3_detail_14m_6h_s202707_v1_bs4/last.ckpt"
  [flow_dual]="$LOG_ROOT/exp_flow_dual_135_14m_6h/last.ckpt"
)
declare -A ARCHES=(
  [wb_vanilla]=wb_vanilla
  [wb_skip]=wb_skip
  [flow]=flow
  [flow_noskip]=flow_noskip
  [flow_ungated]=flow_ungated
  [flow_accel]=flow_accel
  [flow_pp]=flow_pp
  [flow_pp2]=flow_pp2
  [flow_spectral]=flow_pp3
  [weatherbridge]=flow_pp3_detail
  [flow_dual]=flow_dual
)
MODELS=(
  wb_vanilla
  wb_skip
  flow
  flow_noskip
  flow_ungated
  flow_accel
  flow_pp
  flow_pp2
  flow_spectral
  weatherbridge
  flow_dual
)

for name in "${MODELS[@]}"; do
  while ! "$PY" tools/train/checkpoint_status.py \
    "${CHECKPOINTS[$name]}" \
    --min-epochs 8 \
    --expected-arch "${ARCHES[$name]}" \
    --expected-total-steps 13136 \
    --min-global-step 13136 \
    --expected-delta-t 6 \
    --quiet; do
    echo "[$(date -Is)] wait complete checkpoint name=$name" | tee -a "$LOG"
    sleep "$POLL_SECONDS"
  done
done

exec 8>/tmp/weatherbridge_journal_eval_6h.lock
flock 8
GPU=""
GPU_LOCK_FD=""
while [[ -z "$GPU" ]]; do
  for candidate in $GPU_CANDIDATES; do
    exec {candidate_fd}>"$LOG_ROOT/.upr_lite_gpu${candidate}.lock"
    if ! flock -n "$candidate_fd"; then
      exec {candidate_fd}>&-
      continue
    fi
    free_mib="$(nvidia-smi --query-gpu=memory.free \
      --format=csv,noheader,nounits -i "$candidate" | tr -d ' ')"
    util="$(nvidia-smi --query-gpu=utilization.gpu \
      --format=csv,noheader,nounits -i "$candidate" | tr -d ' ')"
    if [[ "$free_mib" -ge "$MIN_FREE_MIB" && "$util" -le "$MAX_UTIL" ]]; then
      GPU="$candidate"
      GPU_LOCK_FD="$candidate_fd"
      break
    fi
    flock -u "$candidate_fd"
    exec {candidate_fd}>&-
  done
  if [[ -z "$GPU" ]]; then
    sleep "$POLL_SECONDS"
  fi
done
export CUDA_VISIBLE_DEVICES="$GPU"

for name in "${MODELS[@]}"; do
  result="$OUT/${name}.json"
  if "$PY" tools/eval/eval_artifact_status.py \
    "$result" \
    --checkpoint "${CHECKPOINTS[$name]}" \
    --required-taus 1,2,3,4,5 \
    --acc-mode disabled \
    --require-physical-metrics \
    --quiet >>"$LOG" 2>&1; then
    echo "[$(date -Is)] skip validated $result" | tee -a "$LOG"
    continue
  fi

  echo "[$(date -Is)] start ablation=$name GPU=$GPU" | tee -a "$LOG"
  "$PY" -u tools/eval/batch_eval_12h_memmap.py \
    --memmap-dir /tmp/wb2_0p5_cache \
    --test-year 2020 \
    --stats-path data/json_stats_0p5.nc \
    --surface-stats-path data/surface_stats_0p5.json \
    --static-path data/static_features_0p5.pt \
    --models "${name}:${CHECKPOINTS[$name]}:" \
    --out-dir "$OUT" \
    --paper-tag journal_6h_2020_full_year_ablation \
    --batch-size 4 \
    --num-workers 2 \
    --samples-per-date 4 \
    --full-year \
    --max-tau-hours 6 \
    --eval-hours 1,2,3,4,5 \
    --seen-tau 1,3,5 \
    --unseen-tau 2,4 \
    --keep-n-channels 24 \
    --proper-rmse \
    --save-window-metrics \
    --save-physical-metrics \
    --no-acc 2>&1 | tee -a "$LOG"
  "$PY" tools/eval/eval_artifact_status.py \
    "$result" \
    --checkpoint "${CHECKPOINTS[$name]}" \
    --required-taus 1,2,3,4,5 \
    --acc-mode disabled \
    --require-physical-metrics \
    --quiet
  echo "[$(date -Is)] done ablation=$name" | tee -a "$LOG"
done

"$PY" - "$MARKER" "$GPU" <<'PY'
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

path = Path(sys.argv[1])
temporary = path.with_suffix(path.suffix + ".tmp")
temporary.write_text(json.dumps({
    "schema_version": 1,
    "status": "complete",
    "completed_at": datetime.now(timezone.utc).isoformat(),
    "gpu": int(sys.argv[2]),
    "models": [
        "wb_vanilla", "wb_skip", "flow", "flow_noskip", "flow_ungated",
        "flow_accel", "flow_pp", "flow_pp2", "flow_spectral",
        "weatherbridge", "flow_dual",
    ],
}, sort_keys=True) + "\n")
os.replace(temporary, path)
PY
echo "[$(date -Is)] full-year 6 h ablations complete" | tee -a "$LOG"
flock -u "$GPU_LOCK_FD"
