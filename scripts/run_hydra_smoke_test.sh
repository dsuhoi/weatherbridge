#!/usr/bin/env bash
# Run the Hydra smoke test suite inside a fresh wti-train:v1 docker on fibo.
#
# Usage:
#   bash scripts/run_hydra_smoke_test.sh [GPU_INDEX=1]
#
# Mounts:
#   -v /home/d.sukhorukov/weather_time_interpolation:/workspace/code/wti
#   -v /tmp/wti_cache:/tmp/wb2_0p5_cache
#
# Output:
#   /tmp/hydra_smoke_test_results.log  (on the host)
#
# The container is ephemeral (--rm) and gets a single PID-tracked process.
set -euo pipefail

GPU="${1:-1}"
LOG="/tmp/hydra_smoke_test_results.log"

echo "=== launching Hydra smoke tests on GPU ${GPU} ===" | tee "$LOG"
echo "$(date)" | tee -a "$LOG"

docker run --rm --gpus "device=${GPU}" \
  -v /home/d.sukhorukov/weather_time_interpolation:/workspace/code/wti \
  -v /tmp/wti_cache:/tmp/wb2_0p5_cache \
  -w /workspace/code/wti \
  wti-train:v1 bash -lc "
    set -eo pipefail
    pip install --quiet hydra-core==1.3.2 'omegaconf>=2.3,<2.4'
    pytest tests/test_hydra_data_loaders.py \
           tests/test_hydra_model_builds.py \
           tests/test_hydra_eval_loads.py \
           tests/test_hydra_train_one_step.py \
           -v --tb=short
  " 2>&1 | tee -a "$LOG"

echo "$(date) DONE" | tee -a "$LOG"
echo "results in $LOG"
