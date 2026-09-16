#!/usr/bin/env python3
"""Download surface (single-level) variables from WeatherBench2 ERA5 for given years.

Surface vars added on top of existing 5-channel pressure-level dataset:
- 2m_temperature (t2m)
- 10m_u_component_of_wind (u10)
- 10m_v_component_of_wind (v10)
- mean_sea_level_pressure (mslp)
- toa_incident_solar_radiation (tisr) — diurnal signal
- total_column_water_vapour (tcwv) — atmospheric humidity column

Output: per-year zarr files at <target>/surface_<year>.zarr with all 6 vars.

Usage:
    python tools/data/download_surface_vars.py \
        --target /tmp/zarrs --years 2016 2017 2018 2019
"""
import argparse
import time
from pathlib import Path

import numcodecs
import numpy as np
import xarray as xr
import zarr


SURFACE_VARS = {
    "2m_temperature": "t2m",
    "10m_u_component_of_wind": "u10",
    "10m_v_component_of_wind": "v10",
    "mean_sea_level_pressure": "mslp",
    "sea_surface_temperature": "sst",
    "total_cloud_cover": "tcc",
    "total_column_water_vapour": "tcwv",
    "toa_incident_solar_radiation": "tisr",   # included only when --include-tisr
}
ALWAYS_VARS = ["t2m", "u10", "v10", "mslp", "sst", "tcc", "tcwv"]
TISR_VAR = "tisr"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", default="https://storage.googleapis.com/weatherbench2/datasets/era5/1959-2022-1h-360x181_equiangular_with_poles_conservative.zarr")
    p.add_argument("--target", default="/tmp/zarrs")
    p.add_argument("--years", type=int, nargs="+", default=[2016, 2017, 2018, 2019])
    p.add_argument("--chunk-time", type=int, default=744)  # ~31 days at 1h
    p.add_argument("--clevel", type=int, default=5)
    p.add_argument("--overwrite", action="store_true", default=False)
    p.add_argument("--include-tisr", action="store_true", default=False,
                   help="Include TISR (TOA incident solar radiation) in output. "
                        "Default: False (we compute TISR analytically from datetime).")
    args = p.parse_args()
    # Build active var list dynamically
    active_short = list(ALWAYS_VARS)
    if args.include_tisr:
        active_short.append(TISR_VAR)
    short_set = set(active_short)
    src_vars_local = [k for k, v in SURFACE_VARS.items() if v in short_set]
    rename_map_local = {k: SURFACE_VARS[k] for k in src_vars_local}

    target = Path(args.target)
    target.mkdir(parents=True, exist_ok=True)

    print(f"Opening source zarr: {args.source}", flush=True)
    # Retry with backoff: gs:// can timeout on metadata read.
    import time as _t
    last_err = None
    for attempt in range(5):
        try:
            src = xr.open_zarr(
                args.source,
                consolidated=True,
                storage_options={"timeout": 60.0, "asynchronous": False},
            )
            break
        except Exception as e:
            last_err = e
            wait = 10 * (2 ** attempt)
            print(f"  attempt {attempt+1}/5 failed: {type(e).__name__}, retrying in {wait}s", flush=True)
            _t.sleep(wait)
    else:
        raise RuntimeError(f"Failed to open source zarr after 5 retries: {last_err}")
    print(f"  source variables: {len(src.data_vars)} total", flush=True)

    src_vars = src_vars_local
    rename_map = rename_map_local
    print(f"  active surface vars (output): {sorted(rename_map.values())}", flush=True)

    compressor = numcodecs.Blosc(cname="zstd", clevel=args.clevel, shuffle=numcodecs.Blosc.BITSHUFFLE)

    for year in args.years:
        out_path = target / f"surface_{year}.zarr"
        if out_path.exists() and not args.overwrite:
            print(f"[{year}] {out_path} exists, skipping (use --overwrite to rebuild)", flush=True)
            continue

        t0 = time.time()
        print(f"[{year}] selecting time range and {len(src_vars)} vars...", flush=True)
        # Use isel-based slicing — sel() can fail on non-monotonic time index.
        import pandas as pd
        times = pd.to_datetime(src.time.values)
        mask = (times >= pd.Timestamp(f"{year}-01-01")) & (times <= pd.Timestamp(f"{year}-12-31T23:59"))
        idx = np.where(mask)[0]
        ds_year = src[src_vars].isel(time=idx)
        ds_year = ds_year.rename(rename_map)
        # Reorder dims to (time, latitude, longitude) for consistency
        ds_year = ds_year.transpose("time", "latitude", "longitude")
        # Rechunk via dask to match target chunks (avoid alignment errors).
        ds_year = ds_year.chunk({"time": args.chunk_time, "latitude": -1, "longitude": -1})

        T = ds_year.sizes["time"]
        H = ds_year.sizes["latitude"]
        W = ds_year.sizes["longitude"]
        print(f"[{year}] target shape per var: ({T}, {H}, {W})", flush=True)

        # Encoding: chunk by time only (so all spatial loaded in one shot)
        encoding = {}
        for new_name in rename_map.values():
            encoding[new_name] = {
                "chunks": (args.chunk_time, H, W),
                "compressor": compressor,
            }

        print(f"[{year}] writing to {out_path}...", flush=True)
        # Remove old (potentially empty) zarr dir if present.
        if out_path.exists():
            import shutil
            shutil.rmtree(out_path)
        ds_year.to_zarr(str(out_path), mode="w", consolidated=True, encoding=encoding,
                        safe_chunks=False)
        print(f"[{year}] done in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
