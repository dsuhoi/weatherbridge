#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."

LOG_ROOT="${LOG_ROOT:-/home/jovyan/shares/SR006.nfs2/dsuhoi/capmatched_logs}"
METRICS_ROOT="${METRICS_ROOT:-/home/jovyan/shares/SR006.nfs2/dsuhoi/weather_time_interpolation_runtime/metrics}"
PYTHON_BIN="${PYTHON_BIN:-/home/jovyan/.mlspace/envs/ai_scientist/bin/python}"
EXTRA_PYTHONPATH="${EXTRA_PYTHONPATH:-/home/jovyan/shares/SR006.nfs2/dsuhoi/pylibs_mamba:/home/jovyan/shares/SR006.nfs2/dsuhoi/wti_shim}"
CLIMATOLOGY="${CLIMATOLOGY:-/workspace-SR006.nfs2/weather_data/time_interpolation_0p5/climatology_1990-2019_0p5_canonical_v2.zarr}"
CLIMATOLOGY_CACHE="${CLIMATOLOGY_CACHE:-/home/jovyan/shares/SR006.nfs2/dsuhoi/climatology_eval_cache_24ch}"
MIN_FREE_MIB="${MIN_FREE_MIB:-48000}"
MAX_UTIL="${MAX_UTIL:-5}"
SLEEP_SEC="${SLEEP_SEC:-120}"
GPU_CANDIDATES="${GPU_CANDIDATES:-0 1}"
export PYTHONPATH="$EXTRA_PYTHONPATH:$PWD:${PYTHONPATH:-}"

FINAL_REPORT="$METRICS_ROOT/compact_flow_screen_6h/four_epoch_selection.json"
SCREEN_COMPLETE="$LOG_ROOT/compact_flow_short_budget.complete"
FLOW_CHECKPOINT="$LOG_ROOT/exp_flow_pp3_135_14m_6h_s202707_protocol_v2/epoch=3-step=6568.ckpt"
QUEUE_LOG="$LOG_ROOT/compact_flow_short_budget_eval.queue.log"
TERMINAL_MARKER="$LOG_ROOT/compact_flow_short_budget_eval.terminal"
COMPLETE_MARKER="$LOG_ROOT/compact_flow_short_budget_eval.complete"

mkdir -p "$LOG_ROOT" "$CLIMATOLOGY_CACHE"
exec 9>"$LOG_ROOT/compact_flow_short_budget_eval.queue.lock"
if ! flock -n 9; then
  echo "[$(date -Is)] compact short-budget evaluation already active" \
    | tee -a "$QUEUE_LOG"
  exit 0
fi
rm -f "$TERMINAL_MARKER" "$COMPLETE_MARKER"
trap 'touch "$TERMINAL_MARKER"' ERR

while [[ ! -e "$SCREEN_COMPLETE" || ! -s "$FINAL_REPORT" ]]; do
  echo "[$(date -Is)] wait compact short-budget selection" \
    | tee -a "$QUEUE_LOG"
  sleep "$SLEEP_SEC"
done

selected="$("$PYTHON_BIN" -c '
import json, sys
report = json.load(open(sys.argv[1]))
assert report["epoch"] == 3
assert report["reference"] == "weatherbridge_ref"
print(report["selected"] or "")
' "$FINAL_REPORT")"
if [[ -z "$selected" ]]; then
  echo "[$(date -Is)] no short-budget model beat Flow at four epochs" \
    | tee -a "$QUEUE_LOG"
  touch "$COMPLETE_MARKER"
  trap - ERR
  exit 0
fi

case "$selected" in
  upr_lite_implicit_global)
    winner_checkpoint="$LOG_ROOT/exp_upr_lite_implicit_global_24ch_6h_2014_19_sparse135_lr1e4_b8_s202707/epoch=3-step=6568.ckpt"
    ;;
  flow_compact_vp3)
    winner_checkpoint="$LOG_ROOT/exp_flow_compact_vp3_hf_135_3m_6h_s202707_protocol_v2/epoch=3-step=6568.ckpt"
    ;;
  flow_compact_vp3_m)
    winner_checkpoint="$LOG_ROOT/exp_flow_compact_vp3_m_hf_135_5m_6h_s202707_protocol_v1/epoch=3-step=6568.ckpt"
    ;;
  *)
    echo "unknown selected architecture: $selected" >&2
    exit 2
    ;;
esac
for checkpoint in "$winner_checkpoint" "$FLOW_CHECKPOINT"; do
  if [[ ! -s "$checkpoint" ]]; then
    echo "missing frozen evaluation checkpoint: $checkpoint" >&2
    exit 2
  fi
done

gpu=""
lock_fd=""
while [[ -z "$gpu" ]]; do
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
      gpu="$candidate"
      lock_fd="$candidate_fd"
      break
    fi
    flock -u "$candidate_fd"
    exec {candidate_fd}>&-
  done
  if [[ -z "$gpu" ]]; then
    sleep "$SLEEP_SEC"
  fi
done
export CUDA_VISIBLE_DEVICES="$gpu"
echo "[$(date -Is)] evaluate selected=$selected gpu=$gpu" \
  | tee -a "$QUEUE_LOG"

for year in 2020 2021; do
  out_dir="metrics/compact_flow_screen_6h_${year}"
  mkdir -p "$out_dir"
  need_field_eval=0
  for artifact_spec in \
    "compact_short_winner:$winner_checkpoint" \
    "flow_4ep_ref:$FLOW_CHECKPOINT"; do
    name="${artifact_spec%%:*}"
    checkpoint="${artifact_spec#*:}"
    if ! "$PYTHON_BIN" tools/eval/eval_artifact_status.py \
      "$out_dir/$name.json" \
      --checkpoint "$checkpoint" \
      --required-taus 1,2,3,4,5 \
      --acc-mode enabled \
      --require-physical-metrics \
      --require-temporal-metrics \
      --quiet; then
      need_field_eval=1
    fi
  done
  if [[ "$need_field_eval" -eq 1 ]]; then
    "$PYTHON_BIN" -u tools/eval/batch_eval_12h_memmap.py \
      --memmap-dir /tmp/wb2_0p5_cache \
      --test-year "$year" \
      --climatology "$CLIMATOLOGY" \
      --climatology-cache-dir "$CLIMATOLOGY_CACHE" \
      --lazy-climatology \
      --stats-path data/json_stats_0p5.nc \
      --surface-stats-path data/surface_stats_0p5.json \
      --static-path data/static_features_0p5.pt \
      --models "compact_short_winner:$winner_checkpoint:,flow_4ep_ref:$FLOW_CHECKPOINT:" \
      --out-dir "$out_dir" \
      --paper-tag "compact_flow_screen_6h_${year}" \
      --batch-size 2 \
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
      --save-temporal-metrics \
      2>&1 | tee -a "$QUEUE_LOG"
  fi

  for name in compact_short_winner flow_4ep_ref; do
    checkpoint="$winner_checkpoint"
    if [[ "$name" == flow_4ep_ref ]]; then
      checkpoint="$FLOW_CHECKPOINT"
    fi
    "$PYTHON_BIN" -u tools/eval/sh_energy_spectra_12h.py \
      --memmap-dir /tmp/wb2_0p5_cache \
      --test-year "$year" \
      --stats-path data/json_stats_0p5.nc \
      --surface-stats-path data/surface_stats_0p5.json \
      --static-path data/static_features_0p5.pt \
      --ckpt "$checkpoint" \
      --model-name "$name" \
      --model-kind capmatched \
      --out-dir "metrics/compact_flow_spectra_6h_${year}" \
      --taus 1,2,3,4,5 \
      --channels all \
      --lmax 359 \
      --keep-n-channels 24 \
      --batch-size 2 \
      --samples-per-date 2 \
      --eval-days-per-month 2 \
      --hf-ell-min 180 \
      --max-tau-hours 6 \
      --skip-existing \
      2>&1 | tee -a "$QUEUE_LOG"
  done

  "$PYTHON_BIN" tools/eval/paired_block_bootstrap.py \
    --left "compact_short_winner:$out_dir/window_metrics/compact_short_winner.npz" \
    --right "flow_4ep_ref:$out_dir/window_metrics/flow_4ep_ref.npz" \
    --taus 1,2,3,4,5 \
    --block-days 7 \
    --draws 5000 \
    --seed 2027 \
    --out-json "$out_dir/paired_rmse_vs_flow_4ep.json" \
    2>&1 | tee -a "$QUEUE_LOG"

  for tau in 2 4; do
    "$PYTHON_BIN" tools/eval/spectral_block_bootstrap.py \
      --left "compact_short_winner:metrics/compact_flow_spectra_6h_${year}/compact_short_winner_tau${tau}.npz" \
      --right "flow_4ep_ref:metrics/compact_flow_spectra_6h_${year}/flow_4ep_ref_tau${tau}.npz" \
      --channels all \
      --block-days 7 \
      --draws 5000 \
      --seed 2027 \
      --out-json "metrics/compact_flow_spectra_6h_${year}/paired_tau${tau}.json" \
      2>&1 | tee -a "$QUEUE_LOG"
  done
done

flock -u "$lock_fd"
exec {lock_fd}>&-
touch "$COMPLETE_MARKER"
trap - ERR
echo "[$(date -Is)] compact short-budget evaluation complete" \
  | tee -a "$QUEUE_LOG"
