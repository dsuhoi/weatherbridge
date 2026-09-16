#!/usr/bin/env bash
# Run the matched-query-budget WeatherBridge generalization control.

set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."

PY="${PY:-/home/jovyan/.mlspace/envs/ai_scientist/bin/python}"
MEMMAP="${MEMMAP:-/tmp/wb2_0p5_cache}"
STATIC="${STATIC:-/tmp/static_features_0p5.pt}"
STATS="${STATS:-data/json_stats_0p5.nc}"
SURFACE_STATS="${SURFACE_STATS:-data/surface_stats_0p5.json}"
RUN_ROOT="${RUN_ROOT:-/home/jovyan/shares/SR006.nfs2/dsuhoi/6h_query_generalization_matchedbudget_s202707}"
LOCK_ROOT="${LOCK_ROOT:-/home/jovyan/shares/SR006.nfs2/dsuhoi/capmatched_logs}"
EXTRA_PYTHONPATH="${EXTRA_PYTHONPATH:-/home/jovyan/shares/SR006.nfs2/dsuhoi/pylibs_mamba:/home/jovyan/shares/SR006.nfs2/dsuhoi/wti_shim}"
GPU="${GPU:-0}"
SEED="${SEED:-202707}"
MICROBATCHES_PER_EPOCH=6568
OPTIMIZER_STEPS_PER_EPOCH=1642
TOTAL_STEPS=13136

TRAIN_YEARS=(2014 2015 2016 2017 2018 2019)
VAL_YEARS=(2020)
EVAL_TAUS=(1 2 3 4 5)

mkdir -p "$RUN_ROOT" "$LOCK_ROOT"
exec > >(tee -a "$RUN_ROOT/queue.log") 2>&1
export PYTHONPATH="$PWD/legacy/scripts:$EXTRA_PYTHONPATH:$PWD:${PYTHONPATH:-}"

for path in "$MEMMAP" "$STATIC" "$STATS" "$SURFACE_STATS"; do
  [[ -e "$path" ]] || { echo "missing required input: $path" >&2; exit 2; }
done

checkpoint_complete() {
  local checkpoint="$1" train_taus_csv="$2"
  "$PY" tools/train/checkpoint_status.py "$checkpoint" \
    --min-epochs 8 \
    --expected-arch flow_pp3 \
    --expected-total-steps "$TOTAL_STEPS" \
    --min-global-step "$TOTAL_STEPS" \
    --expected-delta-t 6 \
    --require-training-protocol \
    --expected-train-years 2014,2015,2016,2017,2018,2019 \
    --expected-val-years 2020 \
    --expected-train-taus "$train_taus_csv" \
    --expected-eval-taus 1,2,3,4,5 \
    --expected-seed "$SEED" \
    --expected-effective-batch-size 16 \
    --expected-batch-size-per-device 4 \
    --expected-accumulate-grad-batches 4 \
    --expected-train-batches-per-epoch "$MICROBATCHES_PER_EPOCH" \
    --expected-optimizer-steps-per-epoch "$OPTIMIZER_STEPS_PER_EPOCH" \
    --expected-samples-per-date-train 4 \
    --expected-samples-per-date-val 2 \
    --expected-lambda-hf 0.05 \
    --expected-lambda-spec 0.02 \
    --expected-spectral-mask-profile advected \
    --expected-loss-profile uniform \
    --expected-trainable-scope all \
    --expected-anchor-swap-probability 0 \
    --expected-highpass-boundary periodic_lon_replicate_lat \
    --expected-precision bf16-mixed \
    --quiet
}

run_variant() {
  local label="$1" exp_name="$2"
  shift 2
  local -a train_taus=("$@")
  local train_taus_csv checkpoint
  train_taus_csv="$(IFS=,; echo "${train_taus[*]}")"
  checkpoint="$RUN_ROOT/$exp_name/last.ckpt"

  if checkpoint_complete "$checkpoint" "$train_taus_csv"; then
    echo "[$(date -Is)] already complete variant=$label checkpoint=$checkpoint"
    return 0
  fi

  local -a resume=()
  [[ -s "$checkpoint" ]] && resume=(--ckpt_path "$checkpoint")
  echo "[$(date -Is)] start variant=$label train_taus=$train_taus_csv total_steps=$TOTAL_STEPS"
  "$PY" -u tools/train/train_capacity_matched_6h.py \
    --arch flow_pp3 \
    --exp_name "$exp_name" \
    --log_root "$RUN_ROOT" \
    --gpus 0 \
    --bs 4 \
    --val_bs 2 \
    --accumulate 4 \
    --workers 4 \
    --val_workers 2 \
    --release_memmap_pages \
    --precision bf16-mixed \
    --seed "$SEED" \
    --years "${TRAIN_YEARS[@]}" \
    --val_years "${VAL_YEARS[@]}" \
    --max_epochs 8 \
    --lr 1e-4 \
    --warmup_steps 500 \
    --window_hours 6 \
    --train_tau_subset "${train_taus[@]}" \
    --eval_tau "${EVAL_TAUS[@]}" \
    --samples_per_date_train 4 \
    --samples_per_date_val 2 \
    --lambda_hf_override 0.05 \
    --lambda_spec_override 0.02 \
    --lambda_band_override 0 \
    --spectral_mask_profile advected \
    --loss_profile uniform \
    --trainable_scope all \
    --anchor_swap_probability 0 \
    --train_batches_per_epoch "$MICROBATCHES_PER_EPOCH" \
    --ckpt_every_n_epochs 8 \
    --memmap_dir "$MEMMAP" \
    --static_path "$STATIC" \
    --stats_path "$STATS" \
    --surface_stats_path "$SURFACE_STATS" \
    --compile-mode default \
    --compile-scope model \
    --compile-backend inductor \
    "${resume[@]}" \
    >>"$RUN_ROOT/${exp_name}.log" 2>&1

  checkpoint_complete "$checkpoint" "$train_taus_csv"
  echo "[$(date -Is)] complete variant=$label checkpoint=$checkpoint"
}

exec 9>"$LOCK_ROOT/.upr_lite_gpu${GPU}.lock"
echo "[$(date -Is)] queued: waiting for gpu=$GPU training lock"
flock 9
export CUDA_VISIBLE_DEVICES="$GPU"
echo "[$(date -Is)] acquired gpu=$GPU training lock"

run_variant sparse-135 \
  exp_weatherbridge_6h_sparse135_matchedbudget13136_s202707 \
  1 3 5
run_variant dense-12345 \
  exp_weatherbridge_6h_fullhours_matchedbudget13136_s202707 \
  1 2 3 4 5

touch "$RUN_ROOT/.complete"
echo "[$(date -Is)] matched-budget WeatherBridge control complete"
