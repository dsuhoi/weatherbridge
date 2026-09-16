#!/usr/bin/env bash
set -euo pipefail

WAIT_PID="${1:?usage: $0 TRAIN_QUEUE_PID}"
GPU="${GPU:-0}"
PY="${PY:-/home/jovyan/.mlspace/envs/ai_scientist/bin/python}"
RUNTIME="${RUNTIME:-/home/jovyan/shares/SR006.nfs2/dsuhoi/weather_time_interpolation_runtime}"
SOURCE="${SOURCE:-/home/jovyan/shares/SR006.nfs2/dsuhoi/weatherbridge_universal_content_refine_v2_eval_source}"
LOG_ROOT="${LOG_ROOT:-/home/jovyan/shares/SR006.nfs2/dsuhoi/capmatched_logs}"
DATA="${DATA:-/tmp/wb2_0p5_cache}"
DATA_ROOT="${DATA_ROOT:-/home/jovyan/dsuhoi/weather_time_interpolation/data}"
CLIM="${CLIM:-/workspace-SR006.nfs2/weather_data/time_interpolation_0p5/climatology_1990-2019_0p5_canonical_v2.zarr}"
CACHE="${CACHE:-/tmp/wti_climatology_cache_universal_refine}"
OUT="$RUNTIME/metrics/universal_latent_refine_6h"
LOG="$LOG_ROOT/universal_latent_refine_postrun.log"

STANDARD_RUN="$LOG_ROOT/exp_flow_universal_latent_refine_14m_6h_s202707_v1_bs4"
CONTENT_RUN="$LOG_ROOT/exp_flow_universal_content_refine_14m_6h_s202707_v1_bs4"
DETAIL_RUN="$LOG_ROOT/exp_flow_pp3_detail_14m_6h_s202707_v1_bs4"
FLOW_RUN="$LOG_ROOT/exp_flow_pp3_spectral_14m_6h_s202707_v1_bs4"
DCAE_RUN="$LOG_ROOT/exp_weatherdcae_14m_6h_6yr_s202707_matched_bs4"

mkdir -p "$OUT" "$CACHE"
exec > >(tee -a "$LOG") 2>&1

echo "[$(date -Is)] waiting for Refine training queue pid=$WAIT_PID"
while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 120; done

STANDARD_CKPT="$STANDARD_RUN/last.ckpt"
CONTENT_CKPT="$CONTENT_RUN/last.ckpt"
DETAIL_CKPT="$DETAIL_RUN/last.ckpt"
FLOW_CKPT="$FLOW_RUN/last.ckpt"
DCAE_CKPT="$DCAE_RUN/last.ckpt"
for checkpoint in \
  "$STANDARD_CKPT" "$CONTENT_CKPT" \
  "$DETAIL_CKPT" "$FLOW_CKPT" "$DCAE_CKPT"; do
  test -s "$checkpoint"
done

exec 7>"$LOG_ROOT/.universal_refine_postrun_gpu${GPU}.lock"
flock 7
export CUDA_VISIBLE_DEVICES="$GPU"
export PYTHONPATH="$SOURCE:$RUNTIME:${PYTHONPATH:-}"

SELECTION_RMSE="$OUT/2020/rmse_acc"
mkdir -p "$SELECTION_RMSE"
if [[ ! -s "$SELECTION_RMSE/standard.json" || \
      ! -s "$SELECTION_RMSE/content_refine.json" || \
      ! -s "$SELECTION_RMSE/weatherdcae_14m_6yr.json" ]]; then
  (
    cd "$SOURCE"
    "$PY" -u tools/eval/batch_eval_12h_memmap.py \
      --memmap-dir "$DATA" \
      --test-year 2020 \
      --climatology "$CLIM" \
      --climatology-cache-dir "$CACHE" \
      --lazy-climatology \
      --stats-path "$DATA_ROOT/json_stats_0p5.nc" \
      --surface-stats-path "$DATA_ROOT/surface_stats_0p5.json" \
      --static-path "$DATA_ROOT/static_features_0p5.pt" \
      --models "standard:$STANDARD_CKPT,content_refine:$CONTENT_CKPT,weatherbridge_detail:$DETAIL_CKPT,flow_spectral:$FLOW_CKPT,weatherdcae_14m_6yr:$DCAE_CKPT" \
      --out-dir "$SELECTION_RMSE" \
      --paper-tag "universal_latent_refine_6h_2020_selection" \
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
  touch "$SELECTION_RMSE/.complete"
fi

SELECTION="$OUT/selection.json"
"$PY" "$SOURCE/tools/eval/select_strict_dcae_champion.py" \
  --candidate "standard:$SELECTION_RMSE/standard.json" \
  --candidate "content_refine:$SELECTION_RMSE/content_refine.json" \
  --reference "weatherdcae_14m_6yr:$SELECTION_RMSE/weatherdcae_14m_6yr.json" \
  --out-json "$SELECTION"

winner="$($PY -c 'import json,sys; print(json.load(open(sys.argv[1]))["winner"])' "$SELECTION")"
case "$winner" in
  standard) WINNER_CKPT="$STANDARD_CKPT" ;;
  content_refine) WINNER_CKPT="$CONTENT_CKPT" ;;
  *) echo "unknown winner: $winner" >&2; exit 1 ;;
esac
strict_champion="$($PY -c 'import json,sys; print(json.load(open(sys.argv[1]))["strict_champion"] or "")' "$SELECTION")"
echo "[$(date -Is)] selected diagnostic winner=$winner strict_champion=${strict_champion:-none} checkpoint=$WINNER_CKPT"

EVAL_FREEZE="$OUT/selection.freeze.json"
"$PY" "$SOURCE/tools/eval/freeze_ood_selection.py" \
  --selection "$SELECTION" \
  --winner refine_winner \
  --model "refine_winner:$WINNER_CKPT" \
  --model "weatherbridge_detail:$DETAIL_CKPT" \
  --model "flow_spectral:$FLOW_CKPT" \
  --model "weatherdcae_14m_6yr:$DCAE_CKPT" \
  --output "$EVAL_FREEZE"

for year in 2020 2021; do
  YEAR_OUT="$OUT/$year"
  RMSE_OUT="$YEAR_OUT/rmse_acc"
  mkdir -p "$RMSE_OUT"
  if [[ "$year" == 2020 ]]; then
    # Candidate and matched-DCAE artifacts are produced by the selection gate.
    EVAL_MODELS="weatherbridge_detail:$DETAIL_CKPT,flow_spectral:$FLOW_CKPT"
    WINNER_WINDOW="$RMSE_OUT/window_metrics/$winner.npz"
    NEED_RMSE_EVAL=0
    for artifact in \
      "$RMSE_OUT/weatherbridge_detail.json" \
      "$RMSE_OUT/flow_spectral.json" \
      "$RMSE_OUT/window_metrics/weatherbridge_detail.npz" \
      "$RMSE_OUT/window_metrics/flow_spectral.npz"; do
      [[ -s "$artifact" ]] || NEED_RMSE_EVAL=1
    done
  else
    EVAL_MODELS="refine_winner:$WINNER_CKPT,weatherbridge_detail:$DETAIL_CKPT,flow_spectral:$FLOW_CKPT,weatherdcae_14m_6yr:$DCAE_CKPT"
    WINNER_WINDOW="$RMSE_OUT/window_metrics/refine_winner.npz"
    NEED_RMSE_EVAL=0
    for artifact in \
      "$RMSE_OUT/refine_winner.json" \
      "$RMSE_OUT/weatherbridge_detail.json" \
      "$RMSE_OUT/flow_spectral.json" \
      "$RMSE_OUT/weatherdcae_14m_6yr.json" \
      "$RMSE_OUT/window_metrics/refine_winner.npz" \
      "$RMSE_OUT/window_metrics/weatherbridge_detail.npz" \
      "$RMSE_OUT/window_metrics/flow_spectral.npz" \
      "$RMSE_OUT/window_metrics/weatherdcae_14m_6yr.npz"; do
      [[ -s "$artifact" ]] || NEED_RMSE_EVAL=1
    done
  fi
  if (( NEED_RMSE_EVAL )); then
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
        --models "$EVAL_MODELS" \
        --out-dir "$RMSE_OUT" \
        --paper-tag "universal_latent_refine_6h_${year}" \
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
    touch "$RMSE_OUT/.complete"
  fi

  WINDOW="$RMSE_OUT/window_metrics"
  "$PY" "$SOURCE/tools/eval/paired_block_bootstrap.py" \
    --left "refine_winner:$WINNER_WINDOW" \
    --right "weatherbridge_detail:$WINDOW/weatherbridge_detail.npz" \
    --right "flow_spectral:$WINDOW/flow_spectral.npz" \
    --right "weatherdcae_14m_6yr:$WINDOW/weatherdcae_14m_6yr.npz" \
    --taus 1,2,3,4,5 \
    --block-days 7 \
    --draws 5000 \
    --seed 2027 \
    --cellwise \
    --out-json "$RMSE_OUT/paired_rmse.json"
  for metric in acc temporal_curvature physical; do
    "$PY" "$SOURCE/tools/eval/paired_aux_block_bootstrap.py" \
      --left "refine_winner:$WINNER_WINDOW" \
      --right "weatherbridge_detail:$WINDOW/weatherbridge_detail.npz" \
      --right "flow_spectral:$WINDOW/flow_spectral.npz" \
      --right "weatherdcae_14m_6yr:$WINDOW/weatherdcae_14m_6yr.npz" \
      --metric "$metric" \
      --draws 5000 \
      --seed 2027 \
      --out-json "$RMSE_OUT/paired_${metric}.json"
  done

  "$PY" -u "$SOURCE/tools/eval/eval_anchor_exchange_consistency.py" \
    --memmap-dir "$DATA" \
    --test-year "$year" \
    --stats-path "$DATA_ROOT/json_stats_0p5.nc" \
    --surface-stats-path "$DATA_ROOT/surface_stats_0p5.json" \
    --static-path "$DATA_ROOT/static_features_0p5.pt" \
    --models "refine_winner:$WINNER_CKPT,weatherbridge_detail:$DETAIL_CKPT,flow_spectral:$FLOW_CKPT,weatherdcae_14m_6yr:$DCAE_CKPT" \
    --max-tau-hours 6 \
    --samples-per-date 2 \
    --eval-days-per-month 2 \
    --batch-size 2 \
    --num-workers 2 \
    --keep-n-channels 24 \
    --out-json "$YEAR_OUT/anchor_exchange.json" \
    --device cuda:0

  SPECTRA="$YEAR_OUT/spectra"
  mkdir -p "$SPECTRA"
  for spec in \
    "refine_winner:$WINNER_CKPT" \
    "weatherbridge_detail:$DETAIL_CKPT" \
    "flow_spectral:$FLOW_CKPT" \
    "weatherdcae_14m_6yr:$DCAE_CKPT"; do
    name="${spec%%:*}"
    checkpoint="${spec#*:}"
    "$PY" -u "$SOURCE/tools/eval/sh_energy_spectra_12h.py" \
      --memmap-dir "$DATA" \
      --test-year "$year" \
      --stats-path "$DATA_ROOT/json_stats_0p5.nc" \
      --surface-stats-path "$DATA_ROOT/surface_stats_0p5.json" \
      --static-path "$DATA_ROOT/static_features_0p5.pt" \
      --ckpt "$checkpoint" \
      --model-name "$name" \
      --model-kind capmatched \
      --out-dir "$SPECTRA" \
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
      --left "refine_winner:$SPECTRA/refine_winner_tau${tau}.npz" \
      --right "weatherbridge_detail:$SPECTRA/weatherbridge_detail_tau${tau}.npz" \
      --right "flow_spectral:$SPECTRA/flow_spectral_tau${tau}.npz" \
      --right "weatherdcae_14m_6yr:$SPECTRA/weatherdcae_14m_6yr_tau${tau}.npz" \
      --channels all \
      --block-days 7 \
      --draws 5000 \
      --seed 2027 \
      --cellwise \
      --out-json "$SPECTRA/paired_refine_vs_references_tau${tau}.json"
  done

  REGION_OUT="$YEAR_OUT/region_season"
  if [[ ! -e "$REGION_OUT/.complete" ]]; then
    (
      cd "$SOURCE"
      "$PY" -u tools/eval/region_season_12h_eval.py \
        --memmap-dir "$DATA" \
        --test-year "$year" \
        --stats-path "$DATA_ROOT/json_stats_0p5.nc" \
        --surface-stats-path "$DATA_ROOT/surface_stats_0p5.json" \
        --static-path "$DATA_ROOT/static_features_0p5.pt" \
        --models "refine_winner:$WINNER_CKPT,weatherbridge_detail:$DETAIL_CKPT,flow_spectral:$FLOW_CKPT,weatherdcae_14m_6yr:$DCAE_CKPT" \
        --out-dir "$REGION_OUT" \
        --batch-size 2 \
        --num-workers 2 \
        --samples-per-date 2 \
        --eval-days-per-month 8 \
        --max-tau-hours 6 \
        --eval-hours 2,4 \
        --keep-n-channels 24 \
        --device cuda:0
    )
    touch "$REGION_OUT/.complete"
  fi
done

"$PY" "$SOURCE/tools/eval/summarize_spectral_dominance.py" \
  --root "$OUT" \
  --reference weatherdcae_14m_6yr \
  --out-json "$OUT/spectral_dominance_vs_weatherdcae14_6yr.json"

"$PY" "$SOURCE/tools/eval/summarize_anchor_exchange_consistency.py" \
  --selection "$EVAL_FREEZE" \
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
  --selection "$EVAL_FREEZE" \
  --root-2020 "$OUT/2020/region_season" \
  --root-2021 "$OUT/2021/region_season" \
  --models refine_winner,weatherbridge_detail,flow_spectral,weatherdcae_14m_6yr \
  --block-days 7 \
  --draws 5000 \
  --seed 2027 \
  --output "$OUT/region_season_generalization.json"

"$PY" -u "$SOURCE/tools/eval/benchmark_capmatched_inference.py" \
  --model "refine_winner:$WINNER_CKPT" \
  --model "weatherbridge_detail:$DETAIL_CKPT" \
  --model "weatherdcae_14m_6yr:$DCAE_CKPT" \
  --static-path "$DATA_ROOT/static_features_0p5.pt" \
  --batch-size 1 \
  --warmup 3 \
  --iterations 5 \
  --repeats 3 \
  --tau-values 0.1666667,0.3333333,0.5,0.6666667,0.8333333 \
  --out-json "$OUT/inference_cost.json"

touch "$OUT/.complete"
echo "[$(date -Is)] Universal-Latent-Refine postrun complete"
