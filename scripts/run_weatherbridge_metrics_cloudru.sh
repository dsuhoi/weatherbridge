#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/jovyan/shares/SR006.nfs2/dsuhoi/weather_time_interpolation_runtime}"
PYTHON_BIN="${PYTHON_BIN:-/home/jovyan/.mlspace/envs/ai_scientist/bin/python}"
EXTRA_PYTHONPATH="${EXTRA_PYTHONPATH:-/home/jovyan/shares/SR006.nfs2/dsuhoi/pylibs_mamba:/home/jovyan/shares/SR006.nfs2/dsuhoi/wti_shim}"
GPU="${GPU:-0}"
LOG_DIR="${LOG_DIR:-/home/jovyan/shares/SR006.nfs2/dsuhoi/capmatched_logs}"

FLOW_6H_CKPT="${FLOW_6H_CKPT:-/home/jovyan/shares/SR006.nfs2/dsuhoi/capmatched_logs/exp_flow_pp3_135_14m_6h/last.ckpt}"
FLOW_12H_CKPT="${FLOW_12H_CKPT:-/home/jovyan/shares/SR006.nfs2/dsuhoi/capmatched_logs/exp_flow_pp3_12h_2017_19_held468_lr1e4_sp2_protocol_v3/last.ckpt}"
CLIMATOLOGY="${CLIMATOLOGY:-/workspace-SR006.nfs2/weather_data/time_interpolation_0p5/climatology_1990-2019_0p5_canonical_v2.zarr}"
CLIMATOLOGY_CACHE="${CLIMATOLOGY_CACHE:-/home/jovyan/shares/SR006.nfs2/dsuhoi/climatology_eval_cache_24ch}"

mkdir -p "$LOG_DIR" "$REPO_ROOT/metrics" "$REPO_ROOT/weights"
cd "$REPO_ROOT"
export CUDA_VISIBLE_DEVICES="$GPU"
export PYTHONPATH="$EXTRA_PYTHONPATH:$REPO_ROOT:${PYTHONPATH:-}"

exec >"$LOG_DIR/weatherbridge_missing_metrics.log" 2>&1
echo "[$(date -Is)] WeatherBridge metrics started"

if [[ ! -s weights/weatherbridge_14m_6h_bare.pt ]]; then
  "$PYTHON_BIN" -u tools/train/capmatched_to_bare.py \
    --ckpt "$FLOW_6H_CKPT" \
    --arch flow_pp3 \
    --out weights/weatherbridge_14m_6h_bare.pt
fi

if [[ ! -s weights/weatherbridge_14m_12h_bare.pt ]]; then
  "$PYTHON_BIN" -u tools/train/capmatched_to_bare.py \
    --ckpt "$FLOW_12H_CKPT" \
    --arch flow_pp3 \
    --out weights/weatherbridge_14m_12h_bare.pt
fi

if [[ ! -s metrics/acc_6h_2020_paper_leaderboard/weatherbridge_pp3.json ]]; then
  "$PYTHON_BIN" -u tools/eval/batch_eval_12h_memmap.py \
    --memmap-dir /tmp/wb2_0p5_cache \
    --test-year 2020 \
    --climatology "$CLIMATOLOGY" \
    --lazy-climatology \
    --climatology-cache-dir "$CLIMATOLOGY_CACHE" \
    --models "weatherbridge_pp3:$FLOW_6H_CKPT" \
    --out-dir metrics/acc_6h_2020_paper_leaderboard \
    --paper-tag 6h_2020 \
    --batch-size 1 \
    --num-workers 0 \
    --samples-per-date 4 \
    --eval-days-per-month 4 \
    --max-tau-hours 6 \
    --eval-hours 1,2,3,4,5 \
    --seen-tau 1,2,3,4,5 \
    --unseen-tau "" \
    --keep-n-channels 24 \
    --device cuda:0
fi

if [[ ! -s metrics/eval_12h_2020_ep10/WeatherBridge_PP3_3yr_12h.json ]]; then
  "$PYTHON_BIN" -u tools/eval/batch_eval_12h_memmap.py \
    --memmap-dir /tmp/wb2_0p5_cache \
    --test-year 2020 \
    --climatology "$CLIMATOLOGY" \
    --lazy-climatology \
    --climatology-cache-dir "$CLIMATOLOGY_CACHE" \
    --models "WeatherBridge_PP3_3yr_12h:$FLOW_12H_CKPT" \
    --out-dir metrics/eval_12h_2020_ep10 \
    --paper-tag 12h_2020 \
    --batch-size 1 \
    --num-workers 0 \
    --samples-per-date 2 \
    --eval-days-per-month 4 \
    --max-tau-hours 12 \
    --eval-hours 1,2,3,4,5,6,7,8,9,10,11 \
    --seen-tau 1,2,3,5,7,9,10,11 \
    --unseen-tau 4,6,8 \
    --keep-n-channels 24 \
    --device cuda:0
fi

"$PYTHON_BIN" -u tools/eval/normalize_cloudpipe_to_paperbase.py \
  --in-dir metrics/eval_12h_2020_ep10 \
  --out-dir metrics/eval_12h_2020_ep10_normalized \
  --horizon 12h

if [[ ! -s metrics/sh_spectra_12h_ep10_24ch/WeatherBridge_3yr_12h_tau8.npz ]]; then
  "$PYTHON_BIN" -u tools/eval/sh_energy_spectra_12h.py \
    --memmap-dir /tmp/wb2_0p5_cache \
    --test-year 2020 \
    --ckpt "$FLOW_12H_CKPT" \
    --model-name WeatherBridge_3yr_12h \
    --model-kind capmatched \
    --keep-n-channels 24 \
    --max-tau-hours 12 \
    --taus 2,3,5,8 \
    --channels all \
    --out-dir metrics/sh_spectra_12h_ep10_24ch \
    --batch-size 1 \
    --samples-per-date 2 \
    --eval-days-per-month 2 \
    --device cuda:0 \
    --skip-existing
fi

echo "[$(date -Is)] WeatherBridge metrics complete"
