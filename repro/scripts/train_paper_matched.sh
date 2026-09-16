#!/usr/bin/env bash
# Paper architecture/loss/query presets; reuse the existing matched trainer.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

DRY_RUN=0
if [[ "${1:-}" == --dry-run ]]; then
  DRY_RUN=1
  shift
fi
if (( $# < 2 )); then
  echo "Usage: bash $0 [--dry-run] {weatherbridge|weatherdcae|pixelattn_vfi} {6|12} [trainer arguments]" >&2
  exit 2
fi
MODEL="$1"
HORIZON="$2"
shift 2
HF=0
FFT=0
MASK=all
case "$MODEL" in
  weatherbridge) ARCH=weatherbridge; HF=0.05; FFT=0.02; MASK=advected ;;
  weatherdcae) ARCH=weatherdcae ;;
  pixelattn_vfi) ARCH=atmvfi ;;
  *) echo "Unsupported paper model: $MODEL" >&2; exit 2 ;;
esac
case "$HORIZON" in
  6)
    YEARS=(2014 2015 2016 2017 2018 2019)
    TRAIN_HOURS=(1 3 5)
    EPOCHS=8; BATCHES=6568; SAMPLES=4
    ;;
  12)
    YEARS=(2017 2018 2019)
    TRAIN_HOURS=(1 2 3 5 7 9 10 11)
    EPOCHS=10; BATCHES=4372; SAMPLES=2
    ;;
  *) echo "Unsupported anchor interval: $HORIZON" >&2; exit 2 ;;
esac
EVAL_HOURS=()
for (( hour=1; hour<HORIZON; hour++ )); do EVAL_HOURS+=("$hour"); done

# No LR override: retain the matched trainer's existing default or user's flag.
COMMAND=("${PYTHON_BIN:-python}" -m tools.train.train_capacity_matched_6h
  --arch "$ARCH" --years "${YEARS[@]}" --val_years 2020
  --window_hours "$HORIZON" --train_tau_subset "${TRAIN_HOURS[@]}"
  --eval_tau "${EVAL_HOURS[@]}" --samples_per_date_train "$SAMPLES"
  --bs 4 --val_bs 4 --accumulate 4 --gpus 0 --seed 202707
  --max_epochs "$EPOCHS" --train_batches_per_epoch "$BATCHES"
  --warmup_steps 500 --loss_profile uniform
  --lambda_hf_override "$HF" --lambda_spec_override "$FFT"
  --lambda_band_override 0 --lambda_sht_override 0
  --spectral_mask_profile "$MASK"
  --exp_name "paper_${MODEL}_${HORIZON}h" "$@")
if (( DRY_RUN )); then
  printf '%q ' "${COMMAND[@]}"
  printf '\n'
else
  exec "${COMMAND[@]}"
fi
