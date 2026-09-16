#!/usr/bin/env bash
set -euo pipefail

WAIT_PID="${1:?usage: $0 UNIVERSAL_PYRAMID_QUEUE_PID}"
GPU="${GPU:-1}"
PY="${PY:-/home/jovyan/.mlspace/envs/ai_scientist/bin/python}"
RUNTIME="${RUNTIME:-/home/jovyan/shares/SR006.nfs2/dsuhoi/weather_time_interpolation_runtime}"
SOURCE="${SOURCE:-/home/jovyan/shares/SR006.nfs2/dsuhoi/weatherbridge_universal_pyramid_v1_source}"
LOG_ROOT="${LOG_ROOT:-/home/jovyan/shares/SR006.nfs2/dsuhoi/capmatched_logs}"
DATA="${DATA:-/tmp/wb2_0p5_cache}"
DATA_ROOT="${DATA_ROOT:-/home/jovyan/dsuhoi/weather_time_interpolation/data}"
CLIM="${CLIM:-/workspace-SR006.nfs2/weather_data/time_interpolation_0p5/climatology_1990-2019_0p5_canonical_v2.zarr}"
CACHE="${CACHE:-/tmp/wti_climatology_cache_universal_pyramid}"
OUT="$RUNTIME/metrics/universal_pyramid_10m_6h"
LOG="$LOG_ROOT/universal_pyramid_10m_postrun.log"

CANDIDATE_RUN="$LOG_ROOT/exp_upr_universal_latent_q4_10m_6h_s202707_v1_bs8"
UPR_RUN="$LOG_ROOT/exp_upr_implicit_global_14m_24ch_6h_2014_19_sparse135_lr1e4_eb16_s202707"
FLOW_RUN="$LOG_ROOT/exp_flow_pp3_spectral_14m_6h_s202707_v1_bs4"
DETAIL_RUN="$LOG_ROOT/exp_flow_pp3_detail_14m_6h_s202707_v1_bs4"
CANDIDATE_CKPT="$CANDIDATE_RUN/last.ckpt"
UPR_CKPT="$UPR_RUN/last.ckpt"
FLOW_CKPT="$FLOW_RUN/last.ckpt"
DETAIL_CKPT="$DETAIL_RUN/last.ckpt"

mkdir -p "$OUT" "$CACHE"
exec > >(tee -a "$LOG") 2>&1

echo "[$(date -Is)] waiting for Universal-Pyramid queue pid=$WAIT_PID"
while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 120; done
for checkpoint in \
  "$CANDIDATE_CKPT" "$UPR_CKPT" "$FLOW_CKPT" "$DETAIL_CKPT"; do
  test -s "$checkpoint"
done

exec 7>"$LOG_ROOT/.universal_pyramid_postrun_gpu${GPU}.lock"
flock 7
export CUDA_VISIBLE_DEVICES="$GPU"
export PYTHONPATH="$SOURCE:$RUNTIME:${PYTHONPATH:-}"

evaluate_year() {
  local year="$1"
  local year_out="$OUT/$year"
  local rmse_out="$year_out/rmse_acc"
  mkdir -p "$rmse_out"
  if [[ ! -e "$rmse_out/.complete" ]]; then
    (
      cd "$SOURCE"
      "$PY" -u tools/eval/batch_eval_12h_memmap.py \
        --memmap-dir "$DATA" \
        --test-year "$year" \
        --climatology "$CLIM" \
        --climatology-cache-dir "$CACHE" \
        --lazy-climatology \
        --stats-path "$DATA_ROOT/json_stats_0p5.nc" \
        --surface-stats-path "$DATA_ROOT/surface_stats_0p5.json" \
        --static-path "$DATA_ROOT/static_features_0p5.pt" \
        --models "universal_pyramid:$CANDIDATE_CKPT,upr_quality:$UPR_CKPT,weatherbridge_detail:$DETAIL_CKPT,flow_spectral:$FLOW_CKPT" \
        --out-dir "$rmse_out" \
        --paper-tag "universal_pyramid_10m_6h_${year}" \
        --batch-size 2 \
        --num-workers 2 \
        --samples-per-date 2 \
        --eval-days-per-month 8 \
        --max-tau-hours 6 \
        --eval-hours 1,2,3,4,5 \
        --seen-tau 1,3,5 \
        --unseen-tau 2,4 \
        --keep-n-channels 24 \
        --proper-rmse \
        --save-window-metrics \
        --save-physical-metrics \
        --save-temporal-metrics
    )
    touch "$rmse_out/.complete"
  fi

  local window="$rmse_out/window_metrics"
  "$PY" "$SOURCE/tools/eval/paired_block_bootstrap.py" \
    --left "universal_pyramid:$window/universal_pyramid.npz" \
    --right "upr_quality:$window/upr_quality.npz" \
    --right "weatherbridge_detail:$window/weatherbridge_detail.npz" \
    --right "flow_spectral:$window/flow_spectral.npz" \
    --taus 1,2,3,4,5 \
    --block-days 7 \
    --draws 5000 \
    --seed 2027 \
    --cellwise \
    --out-json "$rmse_out/paired_rmse.json"
  for metric in acc temporal_curvature physical; do
    "$PY" "$SOURCE/tools/eval/paired_aux_block_bootstrap.py" \
      --left "universal_pyramid:$window/universal_pyramid.npz" \
      --right "upr_quality:$window/upr_quality.npz" \
      --right "weatherbridge_detail:$window/weatherbridge_detail.npz" \
      --right "flow_spectral:$window/flow_spectral.npz" \
      --metric "$metric" \
      --draws 5000 \
      --seed 2027 \
      --out-json "$rmse_out/paired_${metric}.json"
  done
}

evaluate_diagnostics() {
  local year="$1"
  local year_out="$OUT/$year"
  "$PY" -u "$SOURCE/tools/eval/eval_anchor_exchange_consistency.py" \
    --memmap-dir "$DATA" \
    --test-year "$year" \
    --stats-path "$DATA_ROOT/json_stats_0p5.nc" \
    --surface-stats-path "$DATA_ROOT/surface_stats_0p5.json" \
    --static-path "$DATA_ROOT/static_features_0p5.pt" \
    --models "universal_pyramid:$CANDIDATE_CKPT,upr_quality:$UPR_CKPT,weatherbridge_detail:$DETAIL_CKPT,flow_spectral:$FLOW_CKPT" \
    --max-tau-hours 6 \
    --samples-per-date 2 \
    --eval-days-per-month 2 \
    --batch-size 2 \
    --num-workers 2 \
    --keep-n-channels 24 \
    --out-json "$year_out/anchor_exchange.json" \
    --device cuda:0

  local spectra="$year_out/spectra"
  mkdir -p "$spectra"
  for spec in \
    "universal_pyramid:$CANDIDATE_CKPT" \
    "upr_quality:$UPR_CKPT" \
    "weatherbridge_detail:$DETAIL_CKPT" \
    "flow_spectral:$FLOW_CKPT"; do
    local name="${spec%%:*}"
    local checkpoint="${spec#*:}"
    "$PY" -u "$SOURCE/tools/eval/sh_energy_spectra_12h.py" \
      --memmap-dir "$DATA" \
      --test-year "$year" \
      --stats-path "$DATA_ROOT/json_stats_0p5.nc" \
      --surface-stats-path "$DATA_ROOT/surface_stats_0p5.json" \
      --static-path "$DATA_ROOT/static_features_0p5.pt" \
      --ckpt "$checkpoint" \
      --model-name "$name" \
      --model-kind capmatched \
      --out-dir "$spectra" \
      --taus 1,2,3,4,5 \
      --channels all \
      --lmax 180 \
      --hf-ell-min 80 \
      --keep-n-channels 24 \
      --batch-size 2 \
      --samples-per-date 2 \
      --eval-days-per-month 2 \
      --max-tau-hours 6 \
      --device cuda:0 \
      --skip-existing
  done
  for tau in 1 2 3 4 5; do
    "$PY" "$SOURCE/tools/eval/spectral_block_bootstrap.py" \
      --left "universal_pyramid:$spectra/universal_pyramid_tau${tau}.npz" \
      --right "upr_quality:$spectra/upr_quality_tau${tau}.npz" \
      --right "weatherbridge_detail:$spectra/weatherbridge_detail_tau${tau}.npz" \
      --right "flow_spectral:$spectra/flow_spectral_tau${tau}.npz" \
      --channels all \
      --block-days 7 \
      --draws 5000 \
      --seed 2027 \
      --out-json "$spectra/paired_universal_pyramid_tau${tau}.json"
  done
}

evaluate_regions() {
  local year="$1"
  local region_out="$OUT/$year/region_season"
  if [[ -e "$region_out/.complete" ]]; then
    return
  fi
  mkdir -p "$region_out"
  (
    cd "$SOURCE"
    "$PY" -u tools/eval/region_season_12h_eval.py \
      --memmap-dir "$DATA" \
      --test-year "$year" \
      --stats-path "$DATA_ROOT/json_stats_0p5.nc" \
      --surface-stats-path "$DATA_ROOT/surface_stats_0p5.json" \
      --static-path "$DATA_ROOT/static_features_0p5.pt" \
      --models "universal_pyramid:$CANDIDATE_CKPT,upr_quality:$UPR_CKPT,weatherbridge_detail:$DETAIL_CKPT,flow_spectral:$FLOW_CKPT" \
      --out-dir "$region_out" \
      --batch-size 2 \
      --num-workers 2 \
      --samples-per-date 2 \
      --eval-days-per-month 8 \
      --max-tau-hours 6 \
      --eval-hours 2,4 \
      --keep-n-channels 24 \
      --device cuda:0
  )
  touch "$region_out/.complete"
}

evaluate_year 2020

"$PY" -u "$SOURCE/tools/eval/benchmark_capmatched_inference.py" \
  --model "universal_pyramid:$CANDIDATE_CKPT" \
  --model "upr_quality:$UPR_CKPT" \
  --model "flow_spectral:$FLOW_CKPT" \
  --static-path "$DATA_ROOT/static_features_0p5.pt" \
  --batch-size 1 \
  --warmup 3 \
  --iterations 5 \
  --repeats 3 \
  --tau-values 0.1666667,0.3333333,0.5,0.6666667,0.8333333 \
  --out-json "$OUT/inference_cost.json"

"$PY" "$RUNTIME/tools/eval/select_universal_pyramid.py" \
  --candidate "$OUT/2020/rmse_acc/universal_pyramid.json" \
  --flow-reference "$OUT/2020/rmse_acc/flow_spectral.json" \
  --quality-reference "$OUT/2020/rmse_acc/upr_quality.json" \
  --inference "$OUT/inference_cost.json" \
  --quality-gap-limit 0.05 \
  --latency-ratio-limit 0.80 \
  --cell-regression-limit 0 \
  --out-json "$OUT/selection.json"

eligible="$($PY -c 'import json,sys; print(int(json.load(open(sys.argv[1]))["eligible_for_ood_evaluation"]))' "$OUT/selection.json")"
if [[ "$eligible" -ne 1 ]]; then
  touch "$OUT/.rejected_2020"
  echo "[$(date -Is)] candidate rejected by frozen 2020 quality/compute gate"
  exit 0
fi

"$PY" "$SOURCE/tools/eval/freeze_ood_selection.py" \
  --selection "$OUT/selection.json" \
  --winner universal_pyramid \
  --model "universal_pyramid:$CANDIDATE_CKPT" \
  --model "upr_quality:$UPR_CKPT" \
  --model "weatherbridge_detail:$DETAIL_CKPT" \
  --model "flow_spectral:$FLOW_CKPT" \
  --output "$OUT/selection.freeze.json"

evaluate_diagnostics 2020
evaluate_regions 2020
evaluate_year 2021
evaluate_diagnostics 2021
evaluate_regions 2021
"$PY" "$SOURCE/tools/eval/summarize_anchor_exchange_consistency.py" \
  --selection "$OUT/selection.freeze.json" \
  --artifact-2020 "$OUT/2020/anchor_exchange.json" \
  --artifact-2021 "$OUT/2021/anchor_exchange.json" \
  --block-days 7 \
  --draws 5000 \
  --seed 2027 \
  --all-taus 1,2,3,4,5 \
  --seen-taus 1,3,5 \
  --unseen-taus 2,4 \
  --output "$OUT/anchor_exchange_generalization.json"
"$PY" "$SOURCE/tools/eval/summarize_region_season_generalization.py" \
  --selection "$OUT/selection.freeze.json" \
  --root-2020 "$OUT/2020/region_season" \
  --root-2021 "$OUT/2021/region_season" \
  --models universal_pyramid,upr_quality,weatherbridge_detail,flow_spectral \
  --block-days 7 \
  --draws 5000 \
  --seed 2027 \
  --output "$OUT/region_season_generalization.json"
touch "$OUT/.complete"
echo "[$(date -Is)] Universal-Pyramid postrun complete"
