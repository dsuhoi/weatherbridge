#!/usr/bin/env bash
# Evaluate only the canonical journal checkpoints on common full-year indices.
set -euo pipefail

RUNTIME=${RUNTIME:-/home/jovyan/shares/SR006.nfs2/dsuhoi/weather_time_interpolation_runtime}
LEGACY=${LEGACY:-/home/jovyan/dsuhoi/weather_time_interpolation}
LOG_ROOT=${LOG_ROOT:-/home/jovyan/shares/SR006.nfs2/dsuhoi/capmatched_logs}
PY=${PY:-/home/jovyan/.mlspace/envs/ai_scientist/bin/python}
GPU_CANDIDATES=${GPU_CANDIDATES:-"0 1"}
MIN_FREE_MIB=${MIN_FREE_MIB:-60000}
MAX_UTIL=${MAX_UTIL:-5}
YEARS=${YEARS:-"2020 2021"}
POLL_SECONDS=${POLL_SECONDS:-120}
CLIMATOLOGY_CACHE=${CLIMATOLOGY_CACHE:-/home/jovyan/shares/SR006.nfs2/dsuhoi/climatology_eval_cache_24ch}
UPR_GATE=${UPR_GATE:-metrics/upr_lite_candidate_validation.json}
UPR_SELECTION=${UPR_SELECTION:-metrics/upr_lite_candidate_selection.json}
UPR_FORECAST=${UPR_FORECAST:-metrics/upr_lite_forecast_anchor_2021_v2/summary.json}
UPR_EXCHANGE=${UPR_EXCHANGE:-metrics/upr_lite_anchor_exchange_summary.json}
UPR_REGION=${UPR_REGION:-metrics/upr_lite_region_season_summary_6h.json}
UPR_SEED_COMPARISON=${UPR_SEED_COMPARISON:-metrics/upr_vs_pp3_seed_comparison.json}
UPR_OOD_SPECTRA=${UPR_OOD_SPECTRA:-metrics/upr_vs_pp3_ood_spectra_2021.json}
UPR_12H_GATE=${UPR_12H_GATE:-metrics/upr_lite_transfer_validation_12h.json}
UPR_12H_SELECTION=${UPR_12H_SELECTION:-metrics/upr_lite_transfer_selection_12h.json}
UPR_12H_FORECAST=${UPR_12H_FORECAST:-metrics/upr_lite_forecast_anchor_12h_2021_v1/summary.json}
UPR_12H_EXCHANGE=${UPR_12H_EXCHANGE:-metrics/upr_lite_anchor_exchange_summary_12h.json}
UPR_12H_REGION=${UPR_12H_REGION:-metrics/upr_lite_region_season_summary_12h.json}
UPR_12H_OOD_SPECTRA=${UPR_12H_OOD_SPECTRA:-metrics/upr_lite_transfer_ood_spectra_12h_2021.json}

cd "$RUNTIME"
export PYTHONPATH="$RUNTIME:/home/jovyan/shares/SR006.nfs2/dsuhoi/pylibs_mamba:/home/jovyan/shares/SR006.nfs2/dsuhoi/wti_shim"
export PATH=/home/jovyan/.mlspace/envs/ai_scientist/bin:/usr/local/bin:/usr/bin:/bin
mkdir -p logs/runner metrics/journal_unified "$CLIMATOLOGY_CACHE"

LOG=logs/runner/journal_final_eval_queue.log
LOCK=/tmp/weatherbridge_journal_eval_6h.lock
MARKER=metrics/journal_unified/.final_eval_complete
exec 8>"$LOG_ROOT/.journal_final_eval_queue.lock"
if ! flock -n 8; then
  echo "[$(date -Is)] journal final-eval queue already active" | tee -a "$LOG"
  exit 0
fi
exec 9>"$LOCK"

"$PY" - "$MARKER" <<'PY'
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

path = Path(sys.argv[1])
temporary = path.with_suffix(path.suffix + ".tmp")
temporary.write_text(json.dumps({
    "schema_version": 1,
    "status": "waiting",
    "updated_at": datetime.now(timezone.utc).isoformat(),
}) + "\n")
os.replace(temporary, path)
PY

while ! "$PY" tools/eval/check_validation_gate.py \
  --validation "$UPR_GATE" \
  --selection "$UPR_SELECTION" >>"$LOG" 2>&1
do
  echo "[$(date -Is)] wait confirmed UPR gate=$UPR_GATE" \
    | tee -a "$LOG"
  sleep "$POLL_SECONDS"
done

while ! "$PY" tools/eval/check_validation_gate.py \
  --validation "$UPR_12H_GATE" \
  --selection "$UPR_12H_SELECTION" >>"$LOG" 2>&1
do
  echo "[$(date -Is)] wait confirmed 12h UPR gate=$UPR_12H_GATE" \
    | tee -a "$LOG"
  sleep "$POLL_SECONDS"
done

while ! "$PY" tools/eval/check_postselection_generalization_gate.py \
  --selection "$UPR_SELECTION" \
  --forecast "$UPR_FORECAST" \
  --exchange "$UPR_EXCHANGE" \
  --region "$UPR_REGION" \
  --seed-report "$UPR_SEED_COMPARISON" \
  --ood-spectra "$UPR_OOD_SPECTRA" \
  --delta-t-hours 6 >>"$LOG" 2>&1
do
  echo "[$(date -Is)] wait confirmed 6h seed/OOD diagnostics" \
    | tee -a "$LOG"
  sleep "$POLL_SECONDS"
done

while ! "$PY" tools/eval/check_postselection_generalization_gate.py \
  --selection "$UPR_12H_SELECTION" \
  --forecast "$UPR_12H_FORECAST" \
  --exchange "$UPR_12H_EXCHANGE" \
  --region "$UPR_12H_REGION" \
  --ood-spectral-summary "$UPR_12H_OOD_SPECTRA" \
  --delta-t-hours 12 >>"$LOG" 2>&1
do
  echo "[$(date -Is)] wait confirmed 12h post-selection diagnostics" \
    | tee -a "$LOG"
  sleep "$POLL_SECONDS"
done

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

wait_checkpoint() {
  local checkpoint="$1"
  while [[ ! -s "$checkpoint" ]]; do
    echo "[$(date -Is)] wait for $checkpoint" | tee -a "$LOG"
    sleep "$POLL_SECONDS"
  done
}

wait_training_done() {
  local experiment="$1"
  while pgrep -f -- "--exp_name ${experiment}" >/dev/null; do
    echo "[$(date -Is)] wait for training completion: $experiment" | tee -a "$LOG"
    sleep "$POLL_SECONDS"
  done
}

evaluate_model() {
  local horizon="$1"
  local samples_per_date="$2"
  local seen_tau="$3"
  local unseen_tau="$4"
  local eval_tau="$5"
  local name="$6"
  local checkpoint="$7"
  local model_env="${8:-}"
  local training_experiment="${9:-}"

  wait_checkpoint "$checkpoint"
  if [[ -n "$training_experiment" ]]; then
    wait_training_done "$training_experiment"
  fi
  for year in $YEARS; do
    local out="metrics/journal_unified/${horizon}h_${year}"
    local result="$out/${name}.json"
    if "$PY" tools/eval/eval_artifact_status.py \
      "$result" \
      --checkpoint "$checkpoint" \
      --required-taus "$eval_tau" \
      --acc-mode enabled \
      --require-physical-metrics \
      --quiet >>"$LOG" 2>&1; then
      echo "[$(date -Is)] skip validated $result" | tee -a "$LOG"
      continue
    fi

    mkdir -p "$out"
    flock 9
    echo "[$(date -Is)] start ${name} horizon=${horizon} year=${year} GPU=${GPU}" | tee -a "$LOG"
    "$PY" -u tools/eval/batch_eval_12h_memmap.py \
      --memmap-dir /tmp/wb2_0p5_cache \
      --test-year "$year" \
      --climatology /workspace-SR006.nfs2/weather_data/time_interpolation_0p5/climatology_1990-2019_0p5_canonical_v2.zarr \
      --climatology-cache-dir "$CLIMATOLOGY_CACHE" \
      --lazy-climatology \
      --stats-path data/json_stats_0p5.nc \
      --surface-stats-path data/surface_stats_0p5.json \
      --static-path data/static_features_0p5.pt \
      --models "${name}:${checkpoint}:${model_env}" \
      --out-dir "$out" \
      --paper-tag "journal_${horizon}h_${year}_full_year" \
      --batch-size 4 --num-workers 2 \
      --samples-per-date "$samples_per_date" --full-year \
      --max-tau-hours "$horizon" --eval-hours "$eval_tau" \
      --seen-tau "$seen_tau" --unseen-tau "$unseen_tau" \
      --keep-n-channels 24 \
      --proper-rmse --save-window-metrics --save-physical-metrics \
      2>&1 | tee -a "$LOG"
    "$PY" tools/eval/eval_artifact_status.py \
      "$result" \
      --checkpoint "$checkpoint" \
      --required-taus "$eval_tau" \
      --acc-mode enabled \
      --require-physical-metrics \
      --quiet
    echo "[$(date -Is)] done ${name} horizon=${horizon} year=${year}" | tee -a "$LOG"
    flock -u 9
  done
}

# Finish any missing ready-checkpoint entries first. Existing complete model
# outputs are skipped individually, so an interrupted multi-model run does not
# force unrelated recomputation.
evaluate_model 6 4 "1,3,5" "2,4" "1,2,3,4,5" \
  fuxi_24ch_6yr_ep8 \
  "$LEGACY/logs/exp_fuxi_24ch_6yr_v2/epoch=7-step=17520.ckpt"
evaluate_model 6 4 "1,3,5" "2,4" "1,2,3,4,5" \
  modafno_24ch_6yr_ep8 \
  "$LEGACY/logs/exp_modafno_24ch_6yr/epoch=7-step=35032.ckpt" \
  "MODAFNO_INP_H=360 MODAFNO_INP_W=720 MODAFNO_NATIVE_H=360 MODAFNO_NATIVE_W=720"
evaluate_model 6 4 "1,3,5" "2,4" "1,2,3,4,5" \
  sdyff_24ch_6yr_ep8 \
  "$LEGACY/logs/exp_sdyff_24ch_6yr/epoch=7-step=35032.ckpt" \
  "SDYFF_NLAT=360 SDYFF_NLON=720 SDYFF_LAT_CROP=0"
evaluate_model 6 4 "1,3,5" "2,4" "1,2,3,4,5" \
  weatherbridge_pp3_14m_6yr_ep8 \
  "$LOG_ROOT/exp_flow_pp3_135_14m_6h_s202707_protocol_v2/last.ckpt" \
  "" \
  exp_flow_pp3_135_14m_6h_s202707_protocol_v2

# The compact six-hour reference is already available.
evaluate_model 6 4 "1,3,5" "2,4" "1,2,3,4,5" \
  weatherdcae_14m_3yr_ep8_matched \
  "$LOG_ROOT/exp_weatherdcae_14m_6h_skip_ablation_noskip_s202707_v2/last.ckpt"

# Canonical 12 h checkpoints already available.
evaluate_model 12 2 "1,2,3,5,7,9,10,11" "4,6,8" "1,2,3,4,5,6,7,8,9,10,11" \
  weatherbridge_14m_3yr_ep10 \
  "$LOG_ROOT/exp_flow_pp3_12h_2017_19_held468_lr1e4_sp2_protocol_v3/last.ckpt" \
  "" \
  exp_flow_pp3_12h_2017_19_held468_lr1e4_sp2_protocol_v3
evaluate_model 12 2 "1,2,3,5,7,9,10,11" "4,6,8" "1,2,3,4,5,6,7,8,9,10,11" \
  fuxi_3yr_ep10 \
  "$LOG_ROOT/exp_fuxi_24ch_12h_2017_19_lr1e4/epoch=9-step=5470.ckpt"
evaluate_model 12 2 "1,2,3,5,7,9,10,11" "4,6,8" "1,2,3,4,5,6,7,8,9,10,11" \
  modafno_3yr_ep10 \
  "$LEGACY/logs/_12h_migrated/modafno_12h.ckpt" \
  "MODAFNO_INP_H=362 MODAFNO_INP_W=720 MODAFNO_NATIVE_H=360 MODAFNO_NATIVE_W=720"
evaluate_model 12 2 "1,2,3,5,7,9,10,11" "4,6,8" "1,2,3,4,5,6,7,8,9,10,11" \
  sdyff_3yr_ep10 \
  "$LEGACY/logs/_12h_migrated/sdyff_12h.ckpt" \
  "SDYFF_NLAT=360 SDYFF_NLON=720 SDYFF_LAT_CROP=0"

# These entries wait for the matched training queue instead of falling back to
# historical checkpoints.
evaluate_model 6 4 "1,3,5" "2,4" "1,2,3,4,5" \
  weatherdcae_skip_14m_3yr_ep8_matched \
  "$LOG_ROOT/exp_weatherdcae_14m_6h_skip_ablation_skip_s202707_v2/last.ckpt"
evaluate_model 6 4 "1,3,5" "2,4" "1,2,3,4,5" \
  atm_vfi_6yr_ep8_matched \
  "$LOG_ROOT/exp_atmvfi_24ch_6h_2014_19_sparse135_lr1e4_s202707_protocol_v2/last.ckpt" \
  "" \
  exp_atmvfi_24ch_6h_2014_19_sparse135_lr1e4_s202707_protocol_v2
evaluate_model 12 2 "1,2,3,5,7,9,10,11" "4,6,8" "1,2,3,4,5,6,7,8,9,10,11" \
  weatherdcae_14m_3yr_ep10_matched \
  "$LOG_ROOT/exp_weatherdcae_14m_24ch_12h_2017_19_lr1e4_protocol_v3/last.ckpt" \
  "" \
  exp_weatherdcae_14m_24ch_12h_2017_19_lr1e4_protocol_v3
evaluate_model 12 2 "1,2,3,5,7,9,10,11" "4,6,8" "1,2,3,4,5,6,7,8,9,10,11" \
  atm_vfi_3yr_ep10_matched \
  "$LOG_ROOT/exp_atmvfi_24ch_12h_2017_19_oddskip_lr1e4_protocol_v3/last.ckpt" \
  "" \
  exp_atmvfi_24ch_12h_2017_19_oddskip_lr1e4_protocol_v3

"$PY" - "$MARKER" "$GPU" <<'PY'
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

path = Path(sys.argv[1])
repo = Path.cwd()
sources = (
    repo / "scripts" / "run_journal_final_eval_queue_cloudru.sh",
    repo / "tools" / "eval" / "batch_eval_12h_memmap.py",
    repo / "tools" / "eval" / "eval_artifact_status.py",
)
payload = {
    "schema_version": 1,
    "status": "complete",
    "completed_at": datetime.now(timezone.utc).isoformat(),
    "gpu": int(sys.argv[2]),
    "source_sha256": {
        str(source.relative_to(repo)): hashlib.sha256(
            source.read_bytes()
        ).hexdigest()
        for source in sources
    },
}
temporary = path.with_suffix(path.suffix + ".tmp")
temporary.write_text(json.dumps(payload, sort_keys=True) + "\n")
os.replace(temporary, path)
PY
echo "[$(date -Is)] canonical full-year evaluation complete" | tee -a "$LOG"
flock -u "$GPU_LOCK_FD"
