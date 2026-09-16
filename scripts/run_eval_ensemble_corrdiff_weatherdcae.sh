#!/usr/bin/env bash
# Run on fibo after corrdiff-fm-weatherdcae-3yr container exits successfully.
#
# DEPRECATED (2026-06-12) — Hydra equivalent:
#   python eval.py eval=crps_ensemble \
#       eval.fm_ckpt=<FM_CKPT> eval.base_ckpt=<BASE_CKPT> \
#       eval.n_ensemble=16 eval.model_name=corrdiff_fm_weatherdcae \
#       eval.out_dir=metrics/crps_0p5_2020
#
# Defaults: N=16 ensemble, bs=2 (lower than non-ensemble due to memory).
# ETA on B300: ~3-5h.
set -euo pipefail

LOG_DIR=/workspace/code/wti/logs/exp_corrdiff_fm_weatherdcae_3yr_fibo

# Pick the latest top-val ckpt; fall back to last.ckpt.
FM_CKPT=$(ls -t ${LOG_DIR}/*imp*.ckpt 2>/dev/null | head -1 || echo "${LOG_DIR}/last.ckpt")
BASE_CKPT=/workspace/code/wti/logs/weatherdcae_noskip_24ch_6yr_ep8.ckpt

if [[ ! -f "$(echo $FM_CKPT | sed 's|/workspace/code/wti|/home/d.sukhorukov/weather_time_interpolation|')" ]]; then
  echo "FM_CKPT not found: $FM_CKPT — aborting."
  exit 1
fi

# Use GPU 1 by default (after corrdiff training is done on GPU 0). Pass GPU=2 to override.
GPU="${GPU:-1}"

docker run -d --rm --gpus "device=${GPU}" --shm-size 32g --name eval-corrdiff-weatherdcae-ens16 \
  -v /home/d.sukhorukov/weather_time_interpolation:/workspace/code/wti \
  -v /tmp/wti_cache:/tmp/wb2_0p5_cache \
  wti-train:v1 bash -c "cd /workspace/code/wti && python3 tools/eval/eval_ensemble_crps.py \
    --mode corrdiff_fm \
    --fm_ckpt ${FM_CKPT} \
    --base_ckpt ${BASE_CKPT} \
    --n_ensemble 16 \
    --batch_size 2 \
    --num_workers 4 \
    --samples_per_date 4 \
    --out_rmse metrics/eval_6h_2020_paper_leaderboard/corrdiff_fm_weatherdcae_24ch_3yr_ep10.json \
    --out_crps metrics/crps_0p5_2020/corrdiff_fm_weatherdcae.json \
    --model_name corrdiff_fm_weatherdcae 2>&1 | tee logs/runner/eval_corrdiff_weatherdcae_ens16.log"

echo "launched: eval-corrdiff-weatherdcae-ens16 on GPU ${GPU}"
echo "FM ckpt: ${FM_CKPT}"
echo "base ckpt: ${BASE_CKPT}"
echo "tail -f /home/d.sukhorukov/weather_time_interpolation/logs/runner/eval_corrdiff_weatherdcae_ens16.log"
