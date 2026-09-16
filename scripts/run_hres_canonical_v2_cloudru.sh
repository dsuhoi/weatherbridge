#!/usr/bin/env bash
set -euo pipefail

SOURCE=${SOURCE:-/home/jovyan/shares/SR006.nfs2/dsuhoi/weather_time_interpolation_spectral_v6}
PY=${PY:-/home/jovyan/.mlspace/envs/ai_scientist/bin/python}
OUT=${OUT:-/home/jovyan/shares/SR006.nfs2/dsuhoi/memmap_hres_2021_canonical_v3}
LOG_ROOT=${LOG_ROOT:-/home/jovyan/shares/SR006.nfs2/dsuhoi/capmatched_logs}
LOG=${LOG:-$LOG_ROOT/hres_2021_canonical_v3.log}
MIN_FREE_KIB=${MIN_FREE_KIB:-75000000}

cd "$SOURCE"
mkdir -p "$OUT" "$LOG_ROOT"
exec 8>"$OUT/build.lock"
if ! flock -n 8; then
  echo "[$(date -Is)] canonical HRES build already active" | tee -a "$LOG"
  exit 0
fi

free_kib=$(df --output=avail "$OUT" | tail -n 1 | tr -d ' ')
if [[ ! -s "$OUT/forecast_archive_manifest.json" && "$free_kib" -lt "$MIN_FREE_KIB" ]]; then
  echo "need at least $MIN_FREE_KIB KiB free, found $free_kib" >&2
  exit 2
fi

INIT_TIMES=(
  2021-01-01T00 2021-01-21T00 2021-02-16T00 2021-03-06T00
  2021-04-01T00 2021-04-21T00 2021-05-16T00 2021-06-06T00
  2021-07-01T00 2021-07-21T00 2021-08-16T00 2021-09-06T00
  2021-10-01T00 2021-10-21T00 2021-11-16T00 2021-12-06T00
)
# WB2 contains rare isolated native-cell omissions. The v3 builder retains the
# fixed seasonal schedule, averages a coarse block only when at least three of
# four native values are finite, and records every affected value in sidecars.
INIT_CSV=$(IFS=,; echo "${INIT_TIMES[*]}")
export PYTHONPATH="$SOURCE:${PYTHONPATH:-}"

"$PY" -u tools/data/build_hres_forecast_memmap.py \
  --out-dir "$OUT" \
  --init-times "$INIT_CSV" 2>&1 | tee -a "$LOG"

test -s "$OUT/forecast_archive_manifest.json"
count=$(find "$OUT" -maxdepth 1 -type f -name 'init_*.bin' | wc -l)
[[ "$count" -eq 16 ]]
echo "[$(date -Is)] canonical HRES build complete" | tee -a "$LOG"
