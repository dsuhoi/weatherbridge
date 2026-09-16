#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."
source tools/train/architecture_candidate_profile.sh

LOG_ROOT="${LOG_ROOT:-/home/jovyan/shares/SR006.nfs2/dsuhoi/capmatched_logs}"
PYTHON_BIN="${PYTHON_BIN:-/home/jovyan/.mlspace/envs/ai_scientist/bin/python}"
EXTRA_PYTHONPATH="${EXTRA_PYTHONPATH:-/home/jovyan/shares/SR006.nfs2/dsuhoi/pylibs_mamba:/home/jovyan/shares/SR006.nfs2/dsuhoi/wti_shim}"
SELECTION="${SELECTION:-metrics/upr_lite_candidate_selection.json}"
ABLATION="${ABLATION:-metrics/upr_lite_highpass_ablation.json}"
MIN_FREE_MIB="${MIN_FREE_MIB:-76000}"
MAX_UTIL="${MAX_UTIL:-5}"
SLEEP_SEC="${SLEEP_SEC:-120}"
GPU_CANDIDATES="${GPU_CANDIDATES:-0 1}"
BASE_TRAINER="legacy/training_snapshots/train_capacity_matched_6h_e8d3bd58.py"
BASE_TRAINER_SHA256="e8d3bd5830ff7cc5a15dca986e94570c7e1b6a3707b8b7e3d5709dd17dcd6b6a"
QUERY_MATCH_TRAINER="legacy/training_snapshots/train_capacity_matched_6h_32f4ce54.py"
QUERY_MATCH_TRAINER_SHA256="32f4ce54ddd3d033ea4ea55112d6e65658caf73961e568680aa627e8794e1f30"
LOG="$LOG_ROOT/upr_lite_nohf_replicates.queue.log"

mkdir -p "$LOG_ROOT"

while [[ ! -s "$SELECTION" ]]; do
  echo "[$(date -Is)] wait selection=$SELECTION" | tee -a "$LOG"
  sleep "$SLEEP_SEC"
done
selection_mtime="$(stat -c %Y "$SELECTION")"
while [[ ! -s "$ABLATION" ]] || \
  [[ "$(stat -c %Y "$ABLATION")" -le "$selection_mtime" ]]; do
  echo "[$(date -Is)] wait fresh no-HF ablation=$ABLATION" | tee -a "$LOG"
  sleep "$SLEEP_SEC"
done

readarray -t promotion < <("$PYTHON_BIN" -c '
import hashlib, json, sys
selection_path, ablation_path = sys.argv[1:]
selection = json.load(open(selection_path))
ablation = json.load(open(ablation_path))
selection_sha = hashlib.sha256(open(selection_path, "rb").read()).hexdigest()
candidate = ablation.get("candidate")
valid = (
    ablation.get("schema_version") == 2
    and ablation.get("selection_sha256") == selection_sha
    and ablation.get("nohf_model") == f"{candidate}_nohf"
    and ablation.get("reference") == "weatherbridge_ref"
    and ablation.get("training_seed_count") == 1
    and ablation.get("architecture_only_single_seed_promotion_passed")
    is True
)
print(candidate or "")
print(int(valid))
' "$SELECTION" "$ABLATION")
candidate="${promotion[0]}"
promoted="${promotion[1]}"
if [[ -z "$candidate" || "$promoted" -ne 1 ]]; then
  echo "[$(date -Is)] stop no-HF replicates: promotion gate failed" \
    | tee -a "$LOG"
  exit 0
fi
TRAINER_ENTRYPOINT="$(architecture_candidate_trainer "$candidate")"
TRAINER_SHA256="$(architecture_candidate_trainer_sha256 "$candidate")"
actual_trainer_sha256="$(sha256sum "$TRAINER_ENTRYPOINT" | awk '{print $1}')"
if [[ "$actual_trainer_sha256" != "$TRAINER_SHA256" ]]; then
  echo "trainer snapshot SHA-256 mismatch: $actual_trainer_sha256" >&2
  exit 2
fi

model_arch="$(architecture_candidate_model_arch "$candidate")"
layout="$(architecture_candidate_layout "$candidate")"
batch_size="$(architecture_candidate_batch_size "$candidate")"
val_batch_size="$(architecture_candidate_val_batch_size "$candidate")"
accumulate="$(architecture_candidate_accumulate "$candidate")"
expected_highpass_boundary="$(
  architecture_candidate_highpass_boundary "$candidate"
)"

checkpoint_valid() {
  local checkpoint="$1"
  local arch="$2"
  local seed="$3"
  local boundary="$4"
  local trainer_sha256="$5"
  local expected_batch_size="${6:-$batch_size}"
  local expected_accumulate="${7:-$accumulate}"
  "$PYTHON_BIN" tools/train/checkpoint_status.py \
    "$checkpoint" \
    --min-epochs 8 \
    --expected-arch "$arch" \
    --expected-total-steps 13136 \
    --min-global-step 13136 \
    --expected-delta-t 6 \
    --require-training-protocol \
    --expected-train-years 2014,2015,2016,2017,2018,2019 \
    --expected-val-years 2020 \
    --expected-train-taus 1,3,5 \
    --expected-eval-taus 1,2,3,4,5 \
    --expected-seed "$seed" \
    --expected-effective-batch-size 16 \
    --expected-batch-size-per-device "$expected_batch_size" \
    --expected-accumulate-grad-batches "$expected_accumulate" \
    --expected-samples-per-date-train 4 \
    --expected-samples-per-date-val 2 \
    --expected-lambda-hf 0 \
    --expected-highpass-boundary "$boundary" \
    --expected-precision bf16-mixed \
    --expected-trainer-sha256 "$trainer_sha256" \
    --quiet
}

pilot_checkpoint="$LOG_ROOT/exp_${candidate}_nohf_24ch_6h_2014_19_sparse135_lr1e4_${layout}_s202707/last.ckpt"
reference_checkpoint="$LOG_ROOT/exp_flow_pp3_135_14m_6h_s202707_protocol_v2/last.ckpt"
if ! checkpoint_valid \
  "$pilot_checkpoint" "$model_arch" 202707 "$expected_highpass_boundary" \
  "$TRAINER_SHA256"; then
  echo "[$(date -Is)] no-HF promotion checkpoint failed protocol validation" \
    | tee -a "$LOG"
  exit 2
fi
if ! checkpoint_valid \
  "$reference_checkpoint" flow_pp3 202707 periodic_lon_replicate_lat \
  "$BASE_TRAINER_SHA256" 4 4; then
  echo "[$(date -Is)] PP3 promotion checkpoint failed protocol validation" \
    | tee -a "$LOG"
  exit 2
fi

run_seed() {
  local seed="$1"
  local exp_name="exp_${candidate}_nohf_24ch_6h_2014_19_sparse135_lr1e4_${layout}_s${seed}"
  local checkpoint="$LOG_ROOT/$exp_name/last.ckpt"
  local run_log="$LOG_ROOT/$exp_name.log"
  if checkpoint_valid \
    "$checkpoint" "$model_arch" "$seed" "$expected_highpass_boundary" \
    "$TRAINER_SHA256"; then
    echo "[$(date -Is)] skip valid no-HF seed=$seed" | tee -a "$LOG"
    return 0
  fi

  while true; do
    local gpu
    for gpu in $GPU_CANDIDATES; do
      local lock_fd
      exec {lock_fd}>"$LOG_ROOT/.upr_lite_gpu${gpu}.lock"
      if ! flock -n "$lock_fd"; then
        exec {lock_fd}>&-
        continue
      fi
      free_mib="$(nvidia-smi --query-gpu=memory.free \
        --format=csv,noheader,nounits -i "$gpu" | tr -d ' ')"
      util="$(nvidia-smi --query-gpu=utilization.gpu \
        --format=csv,noheader,nounits -i "$gpu" | tr -d ' ')"
      if [[ "$free_mib" -lt "$MIN_FREE_MIB" || "$util" -gt "$MAX_UTIL" ]]; then
        flock -u "$lock_fd"
        exec {lock_fd}>&-
        continue
      fi

      resume=()
      if [[ -s "$checkpoint" ]]; then
        resume=(--ckpt_path "$checkpoint")
      fi
      log_start_bytes="$(stat -c %s "$run_log" 2>/dev/null || echo 0)"
      echo "[$(date -Is)] launch no-HF candidate=$candidate seed=$seed gpu=$gpu" \
        | tee -a "$LOG"
      set +e
      (
        export CUDA_VISIBLE_DEVICES="$gpu"
        export PYTHONPATH="$EXTRA_PYTHONPATH:$PWD:${PYTHONPATH:-}"
        "$PYTHON_BIN" -u "$TRAINER_ENTRYPOINT" \
          --arch "$model_arch" \
          --exp_name "$exp_name" \
          --log_root "$LOG_ROOT" \
          --gpus 0 \
          --bs "$batch_size" \
          --val_bs "$val_batch_size" \
          --accumulate "$accumulate" \
          --workers 4 \
          --val_workers 2 \
          --release_memmap_pages \
          --precision bf16-mixed \
          --seed "$seed" \
          --years 2014 2015 2016 2017 2018 2019 \
          --val_years 2020 \
          --max_epochs 8 \
          --lr 1e-4 \
          --window_hours 6 \
          --train_tau_subset 1 3 5 \
          --eval_tau 1 2 3 4 5 \
          --samples_per_date_train 4 \
          --samples_per_date_val 2 \
          --lambda_hf_override 0 \
          --ckpt_every_n_epochs 1 \
          "${resume[@]}"
      ) >>"$run_log" 2>&1
      status=$?
      set -e
      flock -u "$lock_fd"
      exec {lock_fd}>&-

      if checkpoint_valid \
        "$checkpoint" "$model_arch" "$seed" \
        "$expected_highpass_boundary" "$TRAINER_SHA256"; then
        echo "[$(date -Is)] complete no-HF seed=$seed" | tee -a "$LOG"
        return 0
      fi
      if [[ "$status" -ne 0 ]] && \
        tail -c "+$((log_start_bytes + 1))" "$run_log" | \
          grep -Eqi "out of memory|CUDA error: out of memory"; then
        free_after="$(nvidia-smi --query-gpu=memory.free \
          --format=csv,noheader,nounits -i "$gpu" | tr -d ' ')"
        if [[ "$free_after" -ge "$MIN_FREE_MIB" ]]; then
          echo "[$(date -Is)] intrinsic OOM violates frozen bs=$batch_size accumulate=$accumulate seed=$seed" \
            | tee -a "$LOG"
          return "$status"
        fi
        echo "[$(date -Is)] external contention after OOM seed=$seed" \
          | tee -a "$LOG"
      elif [[ "$status" -ne 0 ]]; then
        echo "[$(date -Is)] no-HF seed=$seed failed status=$status" \
          | tee -a "$LOG"
        return "$status"
      fi
      sleep "$SLEEP_SEC"
      break
    done
    sleep "$SLEEP_SEC"
  done
}

run_seed 202708 &
pid_a=$!
run_seed 202709 &
pid_b=$!
status=0
wait "$pid_a" || status=1
wait "$pid_b" || status=1
exit "$status"
