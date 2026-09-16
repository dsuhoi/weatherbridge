#!/usr/bin/env python3
"""Download surface vars from 0.25° WB2 source and coarsen 2× → 0.5°.

The existing download_surface_vars.py opens a 1° (360×181) zarr which mismatches
our 0.5° (720×360) PL pipeline. This script reads the 0.25° source and applies
mean coarsening (factor 2) to produce a 720×360 surface zarr per year.

Usage:
    python tools/data/download_surface_0p5.py --years 2021 \
        --target /workspace-SR006.nfs2/weather_data/time_interpolation_0p5
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import xarray as xr

# Same source as download_wb2_full_0p5.py (0.25° = 1440×721)
DEFAULT_SOURCE = "gs://weatherbench2/datasets/era5/1959-2023_01_10-full_37-1h-0p25deg-chunk-1.zarr"

SURFACE_VARS_MAP = {
    "2m_temperature": "t2m",
    "10m_u_component_of_wind": "u10",
    "10m_v_component_of_wind": "v10",
    "mean_sea_level_pressure": "mslp",
    "sea_surface_temperature": "sst",
    "total_cloud_cover": "tcc",
    "total_column_water_vapour": "tcwv",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=int, nargs="+", required=True)
    ap.add_argument("--target", default="/workspace-SR006.nfs2/weather_data/time_interpolation_0p5")
    ap.add_argument("--source", default=DEFAULT_SOURCE)
    args = ap.parse_args()
    target = Path(args.target); target.mkdir(parents=True, exist_ok=True)
    print(f"opening 0.25° source: {args.source}")
    # Force anonymous GCS auth — otherwise gcsfs tries AWS metadata endpoint and hangs.
    storage_opts = {"token": "anon"} if args.source.startswith("gs://") else {}
    ds = xr.open_zarr(args.source, consolidated=True, chunks={"time": 24},
                      storage_options=storage_opts)
    print(f"  source dims: lat={ds.sizes['latitude']}, lon={ds.sizes['longitude']}, time={ds.sizes['time']}")
    # Verify it's 0.25°
    assert ds.sizes['latitude'] == 721 and ds.sizes['longitude'] == 1440, \
        f"expected 721×1440 (0.25°), got {ds.sizes['latitude']}×{ds.sizes['longitude']}"

    avail = [v for v in SURFACE_VARS_MAP if v in ds.data_vars]
    print(f"  surface vars present: {[SURFACE_VARS_MAP[v] for v in avail]}")

    for year in args.years:
        out = target / f"surface_{year}.zarr"
        if out.exists():
            print(f"  [skip] {out} already exists")
            continue
        t0 = time.time()
        print(f"\n=== YEAR {year} ===")
        ds_year = ds.sel(time=str(year))
        T = ds_year.sizes['time']
        print(f"  T={T}", flush=True)
        # Coarsen each variable per-month, write monthly chunks → robust to dask hangs
        import zarr as zarrlib
        store = zarrlib.DirectoryStore(str(out))
        root = zarrlib.open(store, mode="w")
        # Coordinates
        lat_out = np.linspace(89.75, -89.75, 360, dtype=np.float32)
        lon_out = np.linspace(0.0, 359.5, 720, dtype=np.float32)
        root.create_dataset("latitude", data=lat_out, chunks=(360,))
        root.create_dataset("longitude", data=lon_out, chunks=(720,))
        time_vals = ds_year.time.values
        root.create_dataset("time", data=time_vals.view(np.int64), chunks=(744,))
        root["time"].attrs["units"] = "nanoseconds since 1970-01-01"
        root["latitude"].attrs["units"] = "degrees_north"
        root["longitude"].attrs["units"] = "degrees_east"
        # Iterate vars and time-chunks
        chunk_size = 744  # ~1 month
        for v_long in avail:
            v_short = SURFACE_VARS_MAP[v_long]
            arr = root.create_dataset(
                v_short, shape=(T, 360, 720), chunks=(chunk_size, 360, 720),
                dtype=np.float32, fill_value=np.nan,
            )
            arr.attrs["_ARRAY_DIMENSIONS"] = ["time", "latitude", "longitude"]
            tv = time.time()
            for c0 in range(0, T, chunk_size):
                c1 = min(c0 + chunk_size, T)
                # Read 0.25° chunk and coarsen via numpy
                src = ds_year[v_long].isel(time=slice(c0, c1)).values  # (Tc, 721, 1440)
                # Trim odd pole row at lat=−90 (index 720), coarsen 2x
                src_trim = src[:, :720, :]  # (Tc, 720, 1440)
                # Coarsen: reshape + mean
                out_chunk = src_trim.reshape(c1 - c0, 360, 2, 720, 2).mean(axis=(2, 4))
                arr[c0:c1] = out_chunk.astype(np.float32)
                print(f"    [{v_short}] chunk {c0}:{c1} done ({(time.time()-tv):.1f}s elapsed)", flush=True)
            print(f"  DONE {v_short}: {(time.time()-tv)/60:.1f} min", flush=True)
        for k in ("latitude", "longitude"):
            root[k].attrs["_ARRAY_DIMENSIONS"] = [k]
        root["time"].attrs["_ARRAY_DIMENSIONS"] = ["time"]
        zarrlib.consolidate_metadata(store)
        print(f"  done {year} in {(time.time()-t0)/60:.1f} min")

    print("\n=== ALL DONE ===")


if __name__ == "__main__":
    main()
