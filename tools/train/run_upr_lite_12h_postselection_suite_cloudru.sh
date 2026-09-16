#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."
source tools/train/architecture_candidate_profile.sh

LOG_ROOT="${LOG_ROOT:-/home/jovyan/shares/SR006.nfs2/dsuhoi/capmatched_logs}"
PYTHON_BIN="${PYTHON_BIN:-/home/jovyan/.mlspace/envs/ai_scientist/bin/python}"
EXTRA_PYTHONPATH="${EXTRA_PYTHONPATH:-/home/jovyan/shares/SR006.nfs2/dsuhoi/pylibs_mamba:/home/jovyan/shares/SR006.nfs2/dsuhoi/wti_shim}"
SELECTION="${SELECTION:-metrics/upr_lite_candidate_selection.json}"
SCREEN_JSON="${SCREEN_JSON:-metrics/upr_lite_screen/two_epoch_screen.json}"
SLEEP_SEC="${SLEEP_SEC:-120}"
LOG="$LOG_ROOT/upr_lite_12h_postselection_suite.log"
LOCK="$LOG_ROOT/upr_lite_12h_postselection_suite.lock"
PP3_CHECKPOINT="$LOG_ROOT/exp_flow_pp3_12h_2017_19_held468_lr1e4_sp2_protocol_v3/last.ckpt"
PP3_TRAINER_SHA256="e8d3bd5830ff7cc5a15dca986e94570c7e1b6a3707b8b7e3d5709dd17dcd6b6a"

mkdir -p "$LOG_ROOT"
export PYTHONPATH="$EXTRA_PYTHONPATH:$PWD:${PYTHONPATH:-}"

exec 8>"$LOCK"
if ! flock -n 8; then
  echo "[$(date -Is)] 12h post-selection suite already active" | tee -a "$LOG"
  exit 0
fi

launch_worker() {
  local name="$1"
  shift
  local launcher_log="$LOG_ROOT/${name}.launcher.log"
  nohup nice -n 10 env \
    LOG_ROOT="$LOG_ROOT" \
    PYTHON_BIN="$PYTHON_BIN" \
    EXTRA_PYTHONPATH="$EXTRA_PYTHONPATH" \
    SELECTION="$SELECTION" \
    SCREEN_JSON="$SCREEN_JSON" \
    SLEEP_SEC="$SLEEP_SEC" \
    "$@" >"$launcher_log" 2>&1 &
  echo "[$(date -Is)] launched worker=$name pid=$!" | tee -a "$LOG"
}

while [[ ! -s "$SCREEN_JSON" ]] || [[ "$("$PYTHON_BIN" -c '
import json, sys
print(int(json.load(open(sys.argv[1]))["complete"]))
' "$SCREEN_JSON")" -ne 1 ]]; do
  echo "[$(date -Is)] wait complete screen=$SCREEN_JSON" | tee -a "$LOG"
  sleep "$SLEEP_SEC"
done
screen_mtime="$(stat -c %Y "$SCREEN_JSON")"
while [[ ! -s "$SELECTION" ]] || [[ "$(stat -c %Y "$SELECTION")" -le "$screen_mtime" ]]; do
  echo "[$(date -Is)] wait fresh selection=$SELECTION" | tee -a "$LOG"
  sleep "$SLEEP_SEC"
done

quality_candidate="$("$PYTHON_BIN" -c '
from pathlib import Path
import json, sys
from tools.eval.summarize_upr_lite_seeds import (
    choose_transfer_candidate,
    validate_frozen_selection_for_followup,
)
selection = json.loads(Path(sys.argv[1]).read_text())
validate_frozen_selection_for_followup(selection)
print(choose_transfer_candidate(selection, "quality"))
' "$SELECTION")"
quality_layout="$(architecture_candidate_layout "$quality_candidate")"
quality_arch="$(architecture_candidate_model_arch "$quality_candidate")"
quality_lambda_hf="$(architecture_candidate_lambda_hf "$quality_candidate")"
quality_highpass_boundary="$(
  architecture_candidate_highpass_boundary "$quality_candidate"
)"
quality_trainer_sha256="$(
  architecture_candidate_trainer_sha256 "$quality_candidate"
)"
quality_checkpoint="$LOG_ROOT/exp_${quality_candidate}_24ch_12h_2017_19_held468_lr1e4_sp2_${quality_layout}_s202707/last.ckpt"

launch_worker \
  weatherbridge_pp3_12h \
  bash tools/train/run_weatherbridge_pp3_12h_matched_cloudru.sh
launch_worker \
  upr_lite_quality_12h \
  env TRANSFER_OBJECTIVE=quality \
  bash tools/train/run_upr_lite_winner_12h_cloudru.sh
launch_worker \
  upr_lite_12h_transfer_eval \
  bash tools/eval/run_upr_lite_12h_transfer_queue_cloudru.sh
launch_worker \
  temporal_router_12h \
  bash tools/eval/run_temporal_expert_router_12h_queue_cloudru.sh

while ! "$PYTHON_BIN" tools/train/checkpoint_status.py \
  "$PP3_CHECKPOINT" \
  --min-epochs 10 \
  --expected-arch flow_pp3 \
  --expected-total-steps 10930 \
  --min-global-step 10930 \
  --expected-delta-t 12 \
  --require-training-protocol \
  --expected-train-years 2017,2018,2019 \
  --expected-val-years 2020 \
  --expected-train-taus 1,2,3,5,7,9,10,11 \
  --expected-eval-taus 1,2,3,4,5,6,7,8,9,10,11 \
  --expected-seed 202707 \
  --expected-effective-batch-size 16 \
  --expected-optimizer-steps-per-epoch 1093 \
  --expected-samples-per-date-train 2 \
  --expected-samples-per-date-val 2 \
  --expected-lambda-hf 0 \
  --expected-precision bf16-mixed \
  --expected-trainer-sha256 "$PP3_TRAINER_SHA256" \
  --quiet; do
  echo "[$(date -Is)] wait primary 12h Flow-PP3" | tee -a "$LOG"
  sleep "$SLEEP_SEC"
done

while ! "$PYTHON_BIN" tools/train/checkpoint_status.py \
  "$quality_checkpoint" \
  --min-epochs 10 \
  --expected-arch "$quality_arch" \
  --expected-total-steps 10930 \
  --min-global-step 10930 \
  --expected-delta-t 12 \
  --require-training-protocol \
  --expected-train-years 2017,2018,2019 \
  --expected-val-years 2020 \
  --expected-train-taus 1,2,3,5,7,9,10,11 \
  --expected-eval-taus 1,2,3,4,5,6,7,8,9,10,11 \
  --expected-seed 202707 \
  --expected-effective-batch-size 16 \
  --expected-optimizer-steps-per-epoch 1093 \
  --expected-samples-per-date-train 2 \
  --expected-samples-per-date-val 2 \
  --expected-lambda-hf "$quality_lambda_hf" \
  --expected-highpass-boundary "$quality_highpass_boundary" \
  --expected-precision bf16-mixed \
  --expected-trainer-sha256 "$quality_trainer_sha256" \
  --quiet; do
  echo "[$(date -Is)] wait primary 12h quality=$quality_candidate" \
    | tee -a "$LOG"
  sleep "$SLEEP_SEC"
done

launch_worker \
  upr_lite_efficiency_12h \
  env TRANSFER_OBJECTIVE=efficiency \
  bash tools/train/run_upr_lite_winner_12h_cloudru.sh
launch_worker \
  weatherdcae_12h_reference \
  env REFERENCE_ARCH=dcae_14m \
  bash tools/train/run_12h_reference_retrain_cloudru.sh
launch_worker \
  upr_lite_flow_12h \
  env TRANSFER_OBJECTIVE=flow \
  bash tools/train/run_upr_lite_winner_12h_cloudru.sh

echo "[$(date -Is)] primary 12h arms complete; secondary workers launched" \
  | tee -a "$LOG"
