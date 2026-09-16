#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/jovyan/.mlspace/envs/ai_scientist/bin/python}"
EXTRA_PYTHONPATH="${EXTRA_PYTHONPATH:-/home/jovyan/shares/SR006.nfs2/dsuhoi/pylibs_mamba:/home/jovyan/shares/SR006.nfs2/dsuhoi/wti_shim}"
GPU="${GPU:-0}"
MAX_ACTIVE_BYTES="${MAX_ACTIVE_BYTES:-450000000000}"
MIN_FREE_MIB="${MIN_FREE_MIB:-60000}"
POLL_SECONDS="${POLL_SECONDS:-300}"
LOG_DIR="${LOG_DIR:-/home/jovyan/shares/SR006.nfs2/dsuhoi/capmatched_logs}"

FLOW_6H_CKPT="${FLOW_6H_CKPT:-/home/jovyan/shares/SR006.nfs2/dsuhoi/capmatched_logs/exp_flow_pp3_135_14m_6h/last.ckpt}"
FLOW_12H_CKPT="${FLOW_12H_CKPT:-/home/jovyan/shares/SR006.nfs2/dsuhoi/capmatched_logs/exp_flow_pp3_12h_2017_19_held468_lr1e4_sp2_protocol_v3/last.ckpt}"
DCAE_6H_CKPT="${DCAE_6H_CKPT:-/home/jovyan/dsuhoi/weather_time_interpolation/logs/exp_dcae_noskip_deep_14M_64_128_256_l3_3yr_fibo/epoch=7-step=8760.ckpt}"
CLIMATOLOGY="${CLIMATOLOGY:-/workspace-SR006.nfs2/weather_data/time_interpolation_0p5/climatology_1990-2019_0p5_canonical_v2.zarr}"
CLIMATOLOGY_CACHE="${CLIMATOLOGY_CACHE:-/home/jovyan/shares/SR006.nfs2/dsuhoi/climatology_eval_cache_24ch}"

mkdir -p "$LOG_DIR"
QUEUE_LOG="$LOG_DIR/missing_weatherbridge_paper_evals.queue.log"
RUN_LOG="$LOG_DIR/missing_weatherbridge_paper_evals.log"
LOCK_FILE="$LOG_DIR/missing_weatherbridge_paper_evals.lock"

exec 9>"$LOCK_FILE"
if ! flock -n 9; then
  echo "queue already active" | tee -a "$QUEUE_LOG"
  exit 0
fi

cd "$REPO_ROOT"
export CUDA_VISIBLE_DEVICES="$GPU"
export PYTHONPATH="$EXTRA_PYTHONPATH:$REPO_ROOT:${PYTHONPATH:-}"

resource_snapshot() {
  local anon active_file kernel active_bytes free_mib
  anon="$(awk '$1 == "anon" {print $2}' /sys/fs/cgroup/memory.stat)"
  active_file="$(awk '$1 == "active_file" {print $2}' /sys/fs/cgroup/memory.stat)"
  kernel="$(awk '$1 == "kernel" {print $2}' /sys/fs/cgroup/memory.stat)"
  active_bytes=$((anon + active_file + kernel))
  free_mib="$(
    nvidia-smi \
      --query-gpu=memory.free \
      --format=csv,noheader,nounits \
      -i "$GPU" | tr -d ' '
  )"
  echo "$active_bytes $free_mib"
}

echo "[$(date -Is)] queue started" | tee -a "$QUEUE_LOG"
while true; do
  read -r active_bytes free_mib < <(resource_snapshot)
  echo "[$(date -Is)] active_bytes=$active_bytes gpu_free_mib=$free_mib" \
    | tee -a "$QUEUE_LOG"
  if (( active_bytes < MAX_ACTIVE_BYTES && free_mib >= MIN_FREE_MIB )); then
    break
  fi
  sleep "$POLL_SECONDS"
done

{
  echo "[$(date -Is)] resources ready"

  "$PYTHON_BIN" -u tools/eval/smoke_canonical_weights.py \
    --name WeatherDCAE-14M \
    --horizon 6 \
    --checkpoint "$DCAE_6H_CKPT"

  if [[ ! -s metrics/sh_spectra_6h_unified_v3/weatherdcae_14m_tau2.npz ||
        ! -s metrics/sh_spectra_6h_unified_v3/weatherdcae_14m_tau3.npz ]]; then
    "$PYTHON_BIN" -u tools/eval/sh_energy_spectra_12h.py \
      --memmap-dir /tmp/wb2_0p5_cache \
      --test-year 2020 \
      --ckpt "$DCAE_6H_CKPT" \
      --model-name weatherdcae_14m \
      --model-kind hermite \
      --keep-n-channels 24 \
      --max-tau-hours 6 \
      --taus 2,3 \
      --channels t2m,mslp,u10,v10,T850 \
      --out-dir metrics/sh_spectra_6h_unified_v3 \
      --batch-size 1 \
      --samples-per-date 2 \
      --eval-days-per-month 2 \
      --device cuda:0
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

  if [[ -s metrics/eval_12h_2020_ep10/WeatherBridge_PP3_3yr_12h.json ]]; then
    "$PYTHON_BIN" -u tools/eval/normalize_cloudpipe_to_paperbase.py \
      --in-dir metrics/eval_12h_2020_ep10 \
      --out-dir metrics/eval_12h_2020_ep10_normalized \
      --horizon 12h
  fi

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

  if [[ ! -s weights/weatherdcae_14m_6h_bare.pt ]]; then
    "$PYTHON_BIN" -u tools/train/dcae_to_bare.py \
      --ckpt "$DCAE_6H_CKPT" \
      --out weights/weatherdcae_14m_6h_bare.pt
  fi

  echo "[$(date -Is)] all missing evaluations complete"
} >"$RUN_LOG" 2>&1
