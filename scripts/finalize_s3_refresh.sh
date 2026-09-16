#!/usr/bin/env bash
# Finalize S3 refresh: after refresh_s3_dataset.sh completes successfully and v2 zarrs
# are verified, this script:
#   1. Validates that all expected v2 zarr stores exist and open cleanly with xarray
#   2. Renames legacy /time_interpolation → /time_interpolation_legacy_pre_v2 (safety net)
#   3. Renames /time_interpolation_v2 → /time_interpolation (canonical path)
#
# Run only after Phase B's logs/runner/refresh_s3.log shows "Phase B finished".
set -eo pipefail
source /home/user/conda/etc/profile.d/conda.sh
conda activate /home/jovyan/.mlspace/envs/ai_scientist
cd "$(dirname "$0")/.."

LEGACY=/mnt/s3/bucket8tb/weather_data/time_interpolation
NEW=/mnt/s3/bucket8tb/weather_data/time_interpolation_v2
BACKUP=/mnt/s3/bucket8tb/weather_data/time_interpolation_legacy_pre_v2

YEARS_ALL="2012 2013 2014 2015 2016 2017 2018 2019 2020"

echo "=== Validate v2 dataset before cutover ==="
date

if [[ ! -d "$NEW" ]]; then
  echo "FATAL: $NEW missing. Phase B did not finish?" 1>&2
  exit 1
fi

# Verify each year's zarr opens with expected schema
python -u <<'PY' 2>&1
import sys, xarray as xr
NEW = "/mnt/s3/bucket8tb/weather_data/time_interpolation_v2"
EXPECTED_LEVELS = sorted([1000, 925, 850, 700])  # compare against ascending sort
EXPECTED_PL_VARS = {"t","u","v","q","z"}
EXPECTED_SURF_BASE = {"t2m","u10","v10","mslp","sst","tcc","tcwv"}
problems = []
for y in [2012,2013,2014,2015,2016,2017,2018,2019,2020]:
    pl_path = f"{NEW}/zarr_{y}.zarr"
    sf_path = f"{NEW}/surface_{y}.zarr"
    try:
        ds = xr.open_zarr(pl_path, consolidated=True)
        levels = sorted([int(l) for l in ds.level.values])
        if levels != EXPECTED_LEVELS:
            problems.append(f"PL {y}: levels {levels} != {EXPECTED_LEVELS}")
        missing = EXPECTED_PL_VARS - set(ds.data_vars)
        if missing:
            problems.append(f"PL {y}: missing vars {missing}")
        T = ds.sizes["time"]
        if T < 8000:
            problems.append(f"PL {y}: only {T} time samples (year incomplete)")
    except Exception as e:
        problems.append(f"PL {y}: open FAILED — {type(e).__name__}: {e}")
    try:
        ds = xr.open_zarr(sf_path, consolidated=True)
        miss = EXPECTED_SURF_BASE - set(ds.data_vars)
        # tisr: required only for 2018, must NOT be in others
        if y == 2018:
            if "tisr" not in ds.data_vars:
                problems.append(f"surface 2018: missing tisr (audit)")
        else:
            if "tisr" in ds.data_vars:
                problems.append(f"surface {y}: contains tisr (should be removed)")
        if miss:
            problems.append(f"surface {y}: missing vars {miss}")
    except Exception as e:
        problems.append(f"surface {y}: open FAILED — {type(e).__name__}: {e}")

if problems:
    print("VALIDATION FAILED:")
    for p in problems: print(" -", p)
    sys.exit(2)
print("[ok] v2 dataset validation passed for all 9 years.")
PY

echo
echo "=== Cutover: legacy → backup, v2 → canonical ==="
date

if [[ -e "$BACKUP" ]]; then
  echo "FATAL: $BACKUP already exists. Refusing to clobber." 1>&2
  exit 3
fi

mv "$LEGACY" "$BACKUP"
echo "[ok] $LEGACY  →  $BACKUP"

mv "$NEW" "$LEGACY"
echo "[ok] $NEW  →  $LEGACY"

echo
echo "=== Final state ==="
ls -la /mnt/s3/bucket8tb/weather_data/
du -sh "$LEGACY" "$BACKUP"

echo
echo "=== Done at $(date) ==="
echo "After verifying training works, you can manually delete $BACKUP:"
echo "  rm -rf $BACKUP"
