#!/usr/bin/env bash
# Post-evaluation architecture selection and conditional Detail confirmation.
set -euo pipefail

SOURCE=${SOURCE:-/home/jovyan/shares/SR006.nfs2/dsuhoi/weather_time_interpolation_spectral_v6}
RUNTIME=${RUNTIME:-/home/jovyan/shares/SR006.nfs2/dsuhoi/weather_time_interpolation_runtime}
LOG_ROOT=${LOG_ROOT:-/home/jovyan/shares/SR006.nfs2/dsuhoi/capmatched_logs}
PY=${PY:-/home/jovyan/.mlspace/envs/ai_scientist/bin/python}
POLL_SECONDS=${POLL_SECONDS:-120}
GPU_CANDIDATES=${GPU_CANDIDATES:-"0 1"}
MIN_FREE_MIB=${MIN_FREE_MIB:-70000}
MAX_UTIL=${MAX_UTIL:-5}
SOURCE_ARCHIVE_ROOT=${SOURCE_ARCHIVE_ROOT:-/home/jovyan/shares/SR006.nfs2/dsuhoi}

cd "$SOURCE"
export PYTHONPATH="$SOURCE:/home/jovyan/shares/SR006.nfs2/dsuhoi/pylibs_mamba:/home/jovyan/shares/SR006.nfs2/dsuhoi/wti_shim"
OUT=metrics/journal_champion_v1
STATE="$OUT/state"
LOG=logs/runner/journal_champion_v1_queue.log
DETAILED="$RUNTIME/metrics/detailed_benchmark_v2"
SPECTRA="$SOURCE/metrics/journal_spectra_v6"
SEED_ROOT="$RUNTIME/metrics/npj_seed_ifs_hres_2021_v3"
SEED_GEOMETRY="$SEED_ROOT/geometry_v3/primary.complete"
mkdir -p "$STATE" logs/runner

exec 8>"$LOG_ROOT/.journal_champion_v1_queue.lock"
if ! flock -n 8; then
  echo "[$(date -Is)] journal champion queue already active" | tee -a "$LOG"
  exit 0
fi

SOURCE_FILES=(
  scripts/run_journal_champion_queue_cloudru.sh
  tools/eval/select_journal_champion.py
  tools/eval/benchmark_capmatched_inference.py
  tools/repro/materialize_completion_source_snapshot.py
  tools/train/run_npj_seed_replicates_cloudru.sh
  tools/train/train_capacity_matched_6h.py
  weather_time_interp/model/weatherbridge_flow_model.py
  weather_time_interp/model/dcae_adaln_model.py
  scripts/run_npj_seed_geometry_recheck_cloudru.sh
)
compute_source_sha() {
  sha256sum "${SOURCE_FILES[@]}" | sha256sum | awk '{print $1}'
}

SOURCE_SHA=$(compute_source_sha)
if [[ -n "${EXPECTED_SOURCE_SHA:-}" && "$SOURCE_SHA" != "$EXPECTED_SOURCE_SHA" ]]; then
  echo "source hash mismatch: $SOURCE_SHA != $EXPECTED_SOURCE_SHA" | tee -a "$LOG"
  exit 2
fi
printf '%s  %s\n' "$SOURCE_SHA" "${SOURCE_FILES[*]}" >"$STATE/source.sha256"

verify_source_snapshot() {
  local current_sha
  current_sha=$(compute_source_sha)
  if [[ "$current_sha" != "$SOURCE_SHA" ]]; then
    echo "source changed while queue was active: $current_sha != $SOURCE_SHA" \
      | tee -a "$LOG"
    exit 2
  fi
}

wait_for_inputs() {
  local missing
  while true; do
    missing=0
    for marker in \
      "$DETAILED/6h/.complete" \
      "$DETAILED/12h/.complete" \
      "$SPECTRA/state/.complete" \
      "$SEED_GEOMETRY"; do
      if [[ ! -s "$marker" ]]; then
        echo "[$(date -Is)] wait marker $marker" | tee -a "$LOG"
        missing=1
      fi
    done
    [[ "$missing" -eq 0 ]] && return 0
    sleep "$POLL_SECONDS"
  done
}

acquire_gpu() {
  local candidate candidate_fd free_mib util
  while true; do
    for candidate in $GPU_CANDIDATES; do
      exec {candidate_fd}>"$LOG_ROOT/.upr_lite_gpu${candidate}.lock"
      if ! flock -n "$candidate_fd"; then
        exec {candidate_fd}>&-
        continue
      fi
      free_mib=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$candidate" | tr -d ' ')
      util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i "$candidate" | tr -d ' ')
      if [[ "$free_mib" -ge "$MIN_FREE_MIB" && "$util" -le "$MAX_UTIL" ]]; then
        GPU="$candidate"
        GPU_LOCK_FD="$candidate_fd"
        export CUDA_VISIBLE_DEVICES="$GPU"
        return 0
      fi
      flock -u "$candidate_fd"
      exec {candidate_fd}>&-
    done
    sleep "$POLL_SECONDS"
  done
}

run_selector() {
  local output="$1"
  verify_source_snapshot
  "$PY" tools/eval/select_journal_champion.py \
    --detailed-root "$DETAILED" \
    --spectra-root "$SPECTRA" \
    --seed-root "$SEED_ROOT" \
    --cost-json "$OUT/inference_cost_a100.json" \
    --out-json "$output" | tee -a "$LOG"
}

wait_for_inputs
verify_source_snapshot
for marker in \
  "$DETAILED/6h/.complete" \
  "$DETAILED/12h/.complete" \
  "$SPECTRA/state/.complete"; do
  "$PY" tools/repro/materialize_completion_source_snapshot.py \
    --marker "$marker" \
    --search-root "$SOURCE_ARCHIVE_ROOT" | tee -a "$LOG"
done
acquire_gpu
"$PY" tools/eval/benchmark_capmatched_inference.py \
  --model "weatherbridge_detail:$LOG_ROOT/exp_flow_pp3_detail_14m_6h_refinev1_s202707_bs4/last.ckpt" \
  --model "refine:$LOG_ROOT/exp_flow_universal_latent_refine_14m_6h_s202707_v1_bs4/last.ckpt" \
  --model "flow_spectral:$LOG_ROOT/exp_flow_pp3_spectral_14m_6h_refinev1_s202707_bs4/last.ckpt" \
  --model "dcae:$LOG_ROOT/exp_weatherdcae_14m_6h_6yr_refinev1_s202707_bs4/last.ckpt" \
  --static-path data/static_features_0p5.pt --batch-size 1 \
  --warmup 5 --iterations 20 --repeats 7 --tau-values 0.25,0.5,0.75 \
  --out-json "$OUT/inference_cost_a100.json" | tee -a "$LOG"
flock -u "$GPU_LOCK_FD"
exec {GPU_LOCK_FD}>&-

run_selector "$OUT/preliminary.json"
action=$("$PY" -c 'import json,sys; print(json.load(open(sys.argv[1]))["action"])' "$OUT/preliminary.json")
if [[ "$action" == train_detail_confirmation ]]; then
  echo "[$(date -Is)] preliminary winner is Detail; launch confirmatory seeds" | tee -a "$LOG"
  JOBS_OVERRIDE="detail:6:202708 detail:6:202709 detail:12:202708 detail:12:202709" \
  STATE_DIR="$LOG_ROOT/npj_detail_confirmation_v2.state" \
  QUEUE_LOG="$LOG_ROOT/npj_detail_confirmation_v2.queue.log" \
  EVAL_ROOT="$SEED_ROOT" \
    bash tools/train/run_npj_seed_replicates_cloudru.sh
  JOBS_OVERRIDE="detail:6:202708 detail:6:202709 detail:12:202708 detail:12:202709" \
  STATE_DIR="$SEED_ROOT/geometry_v3/detail_confirmation_state" \
  MARKER="$SEED_ROOT/geometry_v3/detail_confirmation.complete" \
    bash scripts/run_npj_seed_geometry_recheck_cloudru.sh
fi

run_selector "$OUT/final.json"
verify_source_snapshot
"$PY" - "$OUT/final.json" "$STATE/.complete" "$SOURCE_SHA" \
  "${SOURCE_FILES[@]}" <<'PY'
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

result_path = Path(sys.argv[1])
marker_path = Path(sys.argv[2])
source_sha = sys.argv[3]
source_files = [Path(value) for value in sys.argv[4:]]
payload = json.loads(result_path.read_text())
marker = {
    "schema_version": 1,
    "status": "complete",
    "completed_at": datetime.now(timezone.utc).isoformat(),
    "source_sha256": source_sha,
    "source_files_sha256": {
        str(source): __import__("hashlib").sha256(
            source.read_bytes()
        ).hexdigest()
        for source in source_files
    },
    "selector_status": payload["status"],
    "winner": payload["winner"],
    "preliminary_winner": payload["preliminary_winner"],
    "diagnostic_leader": payload["diagnostic_leader"],
    "result_sha256": __import__("hashlib").sha256(result_path.read_bytes()).hexdigest(),
}
target = Path(marker_path)
temporary = target.with_suffix(".tmp")
temporary.write_text(json.dumps(marker, sort_keys=True) + "\n")
os.replace(temporary, target)
PY
echo "[$(date -Is)] journal champion selection complete" | tee -a "$LOG"
