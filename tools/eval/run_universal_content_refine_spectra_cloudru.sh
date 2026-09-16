#!/usr/bin/env bash
set -euo pipefail

WAIT_PID="${1:?usage: $0 CONTENT_TRAIN_PID}"
GPU="${GPU:-0}"
PY="${PY:-/home/jovyan/.mlspace/envs/ai_scientist/bin/python}"
RUNTIME="${RUNTIME:-/home/jovyan/shares/SR006.nfs2/dsuhoi/weather_time_interpolation_runtime}"
SOURCE="${SOURCE:-/home/jovyan/shares/SR006.nfs2/dsuhoi/weatherbridge_universal_content_refine_v2_eval_source}"
LOG_ROOT="${LOG_ROOT:-/home/jovyan/shares/SR006.nfs2/dsuhoi/capmatched_logs}"
DATA="${DATA:-/tmp/wb2_0p5_cache}"
DATA_ROOT="${DATA_ROOT:-/home/jovyan/dsuhoi/weather_time_interpolation/data}"
CONTENT_RUN="$LOG_ROOT/exp_flow_universal_content_refine_14m_6h_s202707_v1_bs4"
CONTENT_CKPT="$CONTENT_RUN/last.ckpt"
PRUNED="$CONTENT_RUN.pruned.json"
REF_ROOT="$RUNTIME/metrics/universal_latent_refine_6h"
OUT="$RUNTIME/metrics/universal_content_refine_6h"
LOG="$LOG_ROOT/universal_content_refine_spectra.log"

mkdir -p "$OUT"
exec > >(tee -a "$LOG") 2>&1

echo "[$(date -Is)] waiting for Content-Refine training pid=$WAIT_PID"
while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 120; done
if [[ -s "$PRUNED" ]]; then
  echo "[$(date -Is)] Content-Refine was pruned; skipping spectra"
  exit 0
fi

"$PY" - "$CONTENT_CKPT" <<'PY'
import sys
import torch

checkpoint = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
assert int(checkpoint.get("epoch", -1)) >= 7
assert int(checkpoint.get("global_step", -1)) >= 13136
assert checkpoint.get("hyper_parameters", {}).get("arch") == (
    "flow_universal_content_refine"
)
PY

echo "[$(date -Is)] waiting for frozen Refine/DCAE spectral references"
while [[ ! -e "$REF_ROOT/.complete" ]]; do sleep 120; done

exec 7>"$LOG_ROOT/.universal_refine_postrun_gpu${GPU}.lock"
flock 7
export CUDA_VISIBLE_DEVICES="$GPU"
export PYTHONPATH="$SOURCE:$RUNTIME:${PYTHONPATH:-}"

for year in 2020 2021; do
  SPECTRA="$OUT/$year/spectra"
  REF_SPECTRA="$REF_ROOT/$year/spectra"
  mkdir -p "$SPECTRA"
  "$PY" -u "$SOURCE/tools/eval/sh_energy_spectra_12h.py" \
    --memmap-dir "$DATA" \
    --test-year "$year" \
    --stats-path "$DATA_ROOT/json_stats_0p5.nc" \
    --surface-stats-path "$DATA_ROOT/surface_stats_0p5.json" \
    --static-path "$DATA_ROOT/static_features_0p5.pt" \
    --ckpt "$CONTENT_CKPT" \
    --model-name content_refine \
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

  for tau in 1 2 3 4 5; do
    "$PY" "$SOURCE/tools/eval/spectral_block_bootstrap.py" \
      --left "content_refine:$SPECTRA/content_refine_tau${tau}.npz" \
      --right "refine_winner:$REF_SPECTRA/refine_winner_tau${tau}.npz" \
      --right "weatherdcae_14m_6yr:$REF_SPECTRA/weatherdcae_14m_6yr_tau${tau}.npz" \
      --channels all \
      --block-days 7 \
      --draws 5000 \
      --seed 2027 \
      --cellwise \
      --out-json "$SPECTRA/paired_content_vs_references_tau${tau}.json"
  done
done

"$PY" "$SOURCE/tools/eval/summarize_spectral_dominance.py" \
  --root "$OUT" \
  --pattern "{year}/spectra/paired_content_vs_references_tau{tau}.json" \
  --reference weatherdcae_14m_6yr \
  --out-json "$OUT/spectral_dominance_vs_weatherdcae14_6yr.json"

touch "$OUT/.complete"
echo "[$(date -Is)] Content-Refine spectra complete"
