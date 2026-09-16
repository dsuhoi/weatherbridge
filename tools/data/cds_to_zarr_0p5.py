#!/usr/bin/env python3
"""Combine monthly CDS NCs → per-year Zarrs + integrity verification.

Stage 2 after download_cds_0p5_parallel.py.

For each year:
  1. Concat 12 monthly NCs along time axis
  2. Rename CDS short names → our short names (msl→mslp; rest keep)
  3. Write per-year Zarr (zarr_YYYY.zarr + surface_YYYY.zarr) with Zstd+bitshuffle

Integrity checks per year:
  - time axis has exactly 8760 or 8784 timesteps (no gaps)
  - all expected vars present
  - spatial shape == (361, 720)
  - sample value ranges sane (T in [180, 330] K, etc.)
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import numcodecs
import numpy as np
import xarray as xr

NC_DIR = Path("/workspace-SR006.nfs2/weather_data/time_interpolation_0p5/nc_tmp")
ZARR_DIR = Path("/workspace-SR006.nfs2/weather_data/time_interpolation_0p5")

EXPECTED_PL_VARS = ["t", "u", "v", "q", "z"]
EXPECTED_SURFACE_VARS = ["t2m", "u10", "v10", "msl", "sst", "tcc"]
RENAME_SURFACE = {"msl": "mslp"}  # match our pipeline convention
EXPECTED_LEVELS = [1000, 925, 850, 700]
EXPECTED_LAT = 361
EXPECTED_LON = 720

# Sanity ranges per variable (post-conversion)
SANE_RANGES = {
    "t":    (150, 340),    # K, atmosphere
    "u":    (-120, 120),   # m/s
    "v":    (-120, 120),
    "q":    (0, 0.04),     # kg/kg
    "z":    (-2e3, 6e4),   # m²/s²
    "t2m":  (180, 330),
    "u10":  (-100, 100),
    "v10":  (-100, 100),
    "mslp": (85000, 110000),  # Pa
    "sst":  (270, 310),     # K (NaN over land OK)
    "tcc":  (0, 1.001),     # fraction
}


def setup_logger():
    logger = logging.getLogger("cds2zarr")
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    h = logging.StreamHandler(sys.stdout)
    h.setFormatter(logging.Formatter("[%(asctime)s] %(message)s", "%H:%M:%S"))
    logger.addHandler(h)
    return logger


def open_year_concat(year, category, log):
    """Open all chunks (per-month, per-10day-chunk) for one year/category."""
    pattern = f"{year}_*_c*_{category}_0p5.nc"
    files = sorted(NC_DIR.glob(pattern))
    # Each year should have ~36 chunks (~3 chunks/month × 12 months)
    expected = 36
    if len(files) < 30:
        log.warning(f"  {year}/{category}: expected ~{expected} chunks, found {len(files)}")
    if len(files) == 0:
        return None
    log.info(f"  concat {len(files)} chunks for {year}/{category}")
    ds_list = []
    for f in files:
        try:
            ds_list.append(xr.open_dataset(str(f), engine="netcdf4"))
        except Exception as e:
            log.error(f"  failed to open {f.name}: {e!r}")
    if not ds_list:
        return None
    ds = xr.concat(ds_list, dim="valid_time")
    ds = ds.rename({"valid_time": "time"})
    # Sort by time (chunks may overlap or be out of order)
    ds = ds.sortby("time")
    # Drop duplicates if any
    _, unique_idx = np.unique(ds.time.values, return_index=True)
    ds = ds.isel(time=sorted(unique_idx))
    for d in list(ds.dims):
        if ds.sizes[d] == 1 and d not in ("time", "latitude", "longitude", "pressure_level"):
            ds = ds.squeeze(d, drop=True)
    return ds


def verify_dataset(ds, year, category, log):
    """Integrity check: shape, time count, var ranges."""
    issues = []
    expected_t = 8784 if year % 4 == 0 else 8760
    actual_t = ds.sizes.get("time", 0)
    if actual_t != expected_t:
        issues.append(f"time={actual_t}, expected {expected_t}")
    if ds.sizes.get("latitude", 0) != EXPECTED_LAT:
        issues.append(f"lat={ds.sizes.get('latitude')}, expected {EXPECTED_LAT}")
    if ds.sizes.get("longitude", 0) != EXPECTED_LON:
        issues.append(f"lon={ds.sizes.get('longitude')}, expected {EXPECTED_LON}")
    if category == "pl":
        if "pressure_level" in ds.coords:
            lvls = sorted(int(L) for L in ds.pressure_level.values)
            if lvls != sorted(EXPECTED_LEVELS):
                issues.append(f"levels={lvls}, expected {EXPECTED_LEVELS}")
        expected = EXPECTED_PL_VARS
    else:
        expected = EXPECTED_SURFACE_VARS
    missing = [v for v in expected if v not in ds.data_vars]
    if missing:
        issues.append(f"missing vars: {missing}")

    # Sample value check on first timestep
    if "time" in ds.dims and ds.sizes["time"] > 0:
        t0 = ds.isel(time=0)
        for v in ds.data_vars:
            try:
                arr = t0[v].values
                vmin, vmax = float(np.nanmin(arr)), float(np.nanmax(arr))
                lo, hi = SANE_RANGES.get(v, (-1e9, 1e9))
                if not (lo <= vmin and vmax <= hi):
                    issues.append(f"{v} out of sane range: [{vmin:.2f}, {vmax:.2f}] vs [{lo}, {hi}]")
            except Exception as e:
                issues.append(f"{v} verify err: {e!r}")
    if issues:
        log.error(f"  {year}/{category} INTEGRITY ISSUES:")
        for i in issues:
            log.error(f"    - {i}")
        return False
    log.info(f"  {year}/{category} INTEGRITY OK (T={actual_t}, lat={EXPECTED_LAT}, lon={EXPECTED_LON})")
    return True


def write_zarr(ds, out_path, log):
    if out_path.exists():
        try:
            check = xr.open_zarr(str(out_path), consolidated=True)
            T = check.sizes.get("time", 0)
            check.close()
            log.info(f"  Zarr exists with time={T} — overwriting")
            import shutil
            shutil.rmtree(out_path)
        except Exception:
            import shutil
            shutil.rmtree(out_path)

    compressor = numcodecs.Blosc(cname="zstd", clevel=5, shuffle=numcodecs.Blosc.BITSHUFFLE)
    encoding = {}
    for v in ds.data_vars:
        dims = ds[v].dims
        chunks = []
        for d in dims:
            if d == "time":
                chunks.append(min(744, ds.sizes[d]))
            else:
                chunks.append(ds.sizes[d])
        encoding[v] = {"compressor": compressor, "chunks": tuple(chunks)}

    log.info(f"  writing {out_path}...")
    t0 = time.time()
    import dask
    with dask.config.set(scheduler="threads"):
        ds.chunk({"time": 744}).to_zarr(str(out_path), mode="w", consolidated=True,
                                          zarr_version=2, encoding=encoding)
    dt = time.time() - t0
    disk = sum(p.stat().st_size for p in out_path.rglob("*") if p.is_file()) / 1024**3
    log.info(f"  DONE {out_path.name} in {dt/60:.1f}min, {disk:.2f}GiB")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--years", type=int, nargs="+",
                        default=[2014, 2015, 2016, 2017, 2018, 2019, 2020])
    args = parser.parse_args()

    log = setup_logger()
    log.info(f"Years: {args.years}")
    log.info(f"NC dir: {NC_DIR}, Zarr dir: {ZARR_DIR}")

    all_ok = True
    for year in args.years:
        log.info(f"=== YEAR {year} ===")
        for category, prefix in [("pl", "zarr"), ("surface", "surface")]:
            ds = open_year_concat(year, category, log)
            if ds is None:
                log.error(f"  no NCs for {year}/{category}")
                all_ok = False
                continue
            if category == "surface" and "msl" in ds.data_vars:
                ds = ds.rename(RENAME_SURFACE)
            ok = verify_dataset(ds, year, category, log)
            if not ok:
                all_ok = False
                log.error(f"  SKIP write for {year}/{category} due to integrity issues")
                continue
            out_path = ZARR_DIR / f"{prefix}_{year}.zarr"
            write_zarr(ds, out_path, log)

    if all_ok:
        log.info("=== ALL YEARS PASSED INTEGRITY + WRITTEN ===")
    else:
        log.error("=== SOME YEARS HAD ISSUES ===")
        sys.exit(1)


if __name__ == "__main__":
    main()
