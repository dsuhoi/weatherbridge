#!/usr/bin/env bash
# Install the canonical-grid numerical sampler only after the running
# architecture x objective evaluation has sealed its source snapshot.
set -euo pipefail

SOURCE=${SOURCE:-/home/jovyan/shares/SR006.nfs2/dsuhoi/weather_time_interpolation_spectral_v6}
RUNTIME=${RUNTIME:-/home/jovyan/shares/SR006.nfs2/dsuhoi/weather_time_interpolation_runtime}
STAGE=${STAGE:-$RUNTIME/staging/numerical_baselines_v3}
FACTOR_MARKER=${FACTOR_MARKER:-$RUNTIME/metrics/loss_architecture_factorial_v1/.complete}
POLL_SECONDS=${POLL_SECONDS:-120}

while [[ ! -s "$FACTOR_MARKER" ]]; do
  echo "[$(date -Is)] wait loss x architecture completion"
  sleep "$POLL_SECONDS"
done

for relative in \
  tools/baselines/numerical_baselines.py \
  weather_time_interp/eval_runner.py \
  scripts/run_full_year_numerical_baselines_cloudru.sh; do
  source_file="$STAGE/$relative"
  target_file="$SOURCE/$relative"
  [[ -s "$source_file" ]] || { echo "missing staged file: $source_file" >&2; exit 2; }
  install -m 0644 "$source_file" "$target_file"
done
chmod +x "$SOURCE/scripts/run_full_year_numerical_baselines_cloudru.sh"

cd "$SOURCE"
python -m py_compile \
  tools/baselines/numerical_baselines.py \
  weather_time_interp/eval_runner.py
bash -n scripts/run_full_year_numerical_baselines_cloudru.sh
exec bash scripts/run_full_year_numerical_baselines_cloudru.sh
