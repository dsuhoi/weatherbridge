#!/usr/bin/env bash
# Canonical spectral-energy, shape, and coherence evaluation for the npj manuscript.
set -euo pipefail

RUNTIME=${RUNTIME:-/home/jovyan/shares/SR006.nfs2/dsuhoi/weather_time_interpolation_runtime}
LEGACY=${LEGACY:-/home/jovyan/dsuhoi/weather_time_interpolation}
LOG_ROOT=${LOG_ROOT:-/home/jovyan/shares/SR006.nfs2/dsuhoi/capmatched_logs}
PY=${PY:-/home/jovyan/.mlspace/envs/ai_scientist/bin/python}
GPU_CANDIDATES=${GPU_CANDIDATES:-"0 1"}
MIN_FREE_MIB=${MIN_FREE_MIB:-60000}
MAX_UTIL=${MAX_UTIL:-5}
POLL_SECONDS=${POLL_SECONDS:-120}
CHANNELS=${CHANNELS:-all}
LMAX=${LMAX:-359}
HF_ELL_MIN=${HF_ELL_MIN:-180}
EVAL_DAYS_PER_MONTH=${EVAL_DAYS_PER_MONTH:-2}

cd "$RUNTIME"
export PYTHONPATH="$RUNTIME:/home/jovyan/shares/SR006.nfs2/dsuhoi/pylibs_mamba:/home/jovyan/shares/SR006.nfs2/dsuhoi/wti_shim"
mkdir -p metrics/journal_spectra logs/runner
LOG=logs/runner/journal_spectra_queue.log
MARKER=metrics/journal_spectra/.complete
FIELD_MARKER=metrics/journal_unified/.final_eval_complete
exec 8>"$LOG_ROOT/.journal_spectra_queue.lock"
if ! flock -n 8; then
  echo "[$(date -Is)] journal spectra queue already active" | tee -a "$LOG"
  exit 0
fi

while ! "$PY" - "$FIELD_MARKER" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
if not path.is_file():
    raise SystemExit(1)
payload = json.loads(path.read_text())
raise SystemExit(0 if payload.get("status") == "complete" else 1)
PY
do
  echo "[$(date -Is)] wait for canonical full-year evaluation" | tee -a "$LOG"
  sleep "$POLL_SECONDS"
done

exec 9>/tmp/weatherbridge_journal_eval_6h.lock
flock 9
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

run_spectra() {
  local horizon="$1"
  local samples_per_date="$2"
  local taus="$3"
  local name="$4"
  local checkpoint="$5"
  local kind="$6"
  local envs="${7:-}"
  local out="metrics/journal_spectra/${horizon}h_2020"

  mkdir -p "$out"
  echo "[$(date -Is)] spectra ${name} horizon=${horizon}" | tee -a "$LOG"
  "$PY" -u tools/eval/sh_energy_spectra_12h.py \
    --memmap-dir /tmp/wb2_0p5_cache \
    --test-year 2020 \
    --stats-path data/json_stats_0p5.nc \
    --surface-stats-path data/surface_stats_0p5.json \
    --static-path data/static_features_0p5.pt \
    --ckpt "$checkpoint" \
    --model-name "$name" \
    --model-kind "$kind" \
    --envs "$envs" \
    --out-dir "$out" \
    --taus "$taus" \
    --channels "$CHANNELS" \
    --lmax "$LMAX" --hf-ell-min "$HF_ELL_MIN" \
    --keep-n-channels 24 \
    --batch-size 2 \
    --samples-per-date "$samples_per_date" \
    --eval-days-per-month "$EVAL_DAYS_PER_MONTH" \
    --max-tau-hours "$horizon" \
    --skip-existing 2>&1 | tee -a "$LOG"
}

SIX_TAUS=1,2,3,4,5
run_spectra 6 4 "$SIX_TAUS" weatherbridge \
  "$LOG_ROOT/exp_flow_pp3_135_14m_6h/last.ckpt" capmatched
run_spectra 6 4 "$SIX_TAUS" weatherdcae_14m \
  "$LOG_ROOT/exp_weatherdcae_14m_6h_skip_ablation_noskip_s202707_v2/last.ckpt" hermite
run_spectra 6 4 "$SIX_TAUS" weatherdcae_skip \
  "$LOG_ROOT/exp_weatherdcae_14m_6h_skip_ablation_skip_s202707_v2/last.ckpt" hermite
run_spectra 6 4 "$SIX_TAUS" fuxi \
  "$LEGACY/logs/exp_fuxi_24ch_6yr_v2/epoch=7-step=17520.ckpt" hermite
run_spectra 6 4 "$SIX_TAUS" modafno \
  "$LEGACY/logs/exp_modafno_24ch_6yr/epoch=7-step=35032.ckpt" hermite \
  "MODAFNO_INP_H=360 MODAFNO_INP_W=720 MODAFNO_NATIVE_H=360 MODAFNO_NATIVE_W=720"
run_spectra 6 4 "$SIX_TAUS" sdyff \
  "$LEGACY/logs/exp_sdyff_24ch_6yr/epoch=7-step=35032.ckpt" hermite \
  "SDYFF_NLAT=360 SDYFF_NLON=720 SDYFF_LAT_CROP=0"
run_spectra 6 4 "$SIX_TAUS" atm_vfi \
  "$LOG_ROOT/exp_atmvfi_24ch_6h_2014_19_sparse135_lr1e4_matched/last.ckpt" atmvfi
run_spectra 6 4 "$SIX_TAUS" linear analytic bilinear

TWELVE_TAUS=1,2,3,4,5,6,7,8,9,10,11
run_spectra 12 2 "$TWELVE_TAUS" weatherbridge \
  "$LOG_ROOT/exp_flow_pp3_12h_2017_19_held468_lr1e4_sp2_protocol_v3/last.ckpt" capmatched
run_spectra 12 2 "$TWELVE_TAUS" weatherdcae_14m \
  "$LOG_ROOT/exp_weatherdcae_14m_24ch_12h_2017_19_lr1e4_protocol_v3/last.ckpt" hermite
run_spectra 12 2 "$TWELVE_TAUS" fuxi \
  "$LOG_ROOT/exp_fuxi_24ch_12h_2017_19_lr1e4/epoch=9-step=5470.ckpt" hermite
run_spectra 12 2 "$TWELVE_TAUS" modafno \
  "$LEGACY/logs/_12h_migrated/modafno_12h.ckpt" hermite \
  "MODAFNO_INP_H=360 MODAFNO_INP_W=720 MODAFNO_NATIVE_H=360 MODAFNO_NATIVE_W=720"
run_spectra 12 2 "$TWELVE_TAUS" sdyff \
  "$LEGACY/logs/_12h_migrated/sdyff_12h.ckpt" hermite \
  "SDYFF_NLAT=360 SDYFF_NLON=720 SDYFF_LAT_CROP=0"
run_spectra 12 2 "$TWELVE_TAUS" atm_vfi \
  "$LOG_ROOT/exp_atmvfi_24ch_12h_2017_19_oddskip_lr1e4_protocol_v3/last.ckpt" atmvfi
run_spectra 12 2 "$TWELVE_TAUS" linear analytic bilinear

spectral_stats() {
  local horizon="$1"
  local tau="$2"
  local root="metrics/journal_spectra/${horizon}h_2020"
  "$PY" tools/eval/spectral_block_bootstrap.py \
    --left "weatherbridge:${root}/weatherbridge_tau${tau}.npz" \
    --right "weatherdcae_14m:${root}/weatherdcae_14m_tau${tau}.npz" \
    --right "fuxi:${root}/fuxi_tau${tau}.npz" \
    --right "modafno:${root}/modafno_tau${tau}.npz" \
    --right "sdyff:${root}/sdyff_tau${tau}.npz" \
    --right "atm_vfi:${root}/atm_vfi_tau${tau}.npz" \
    --right "linear:${root}/linear_tau${tau}.npz" \
    --channels "$CHANNELS" \
    --block-days 7 --draws 5000 --seed 2027 \
    --out-json "${root}/weatherbridge_pairwise_tau${tau}.json" | tee -a "$LOG"
}

for tau in 1 2 3 4 5; do
  spectral_stats 6 "$tau"
done
for tau in 1 2 3 4 5 6 7 8 9 10 11; do
  spectral_stats 12 "$tau"
done

"$PY" - "$MARKER" "$GPU" "$LMAX" "$HF_ELL_MIN" "$CHANNELS" <<'PY'
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

path = Path(sys.argv[1])
payload = {
    "schema_version": 1,
    "status": "complete",
    "completed_at": datetime.now(timezone.utc).isoformat(),
    "gpu": int(sys.argv[2]),
    "lmax": int(sys.argv[3]),
    "hf_ell_min": int(sys.argv[4]),
    "channels": sys.argv[5],
}
temporary = path.with_suffix(path.suffix + ".tmp")
temporary.write_text(json.dumps(payload, sort_keys=True) + "\n")
os.replace(temporary, path)
PY
echo "[$(date -Is)] spectral evaluation complete" | tee -a "$LOG"
flock -u "$GPU_LOCK_FD"
