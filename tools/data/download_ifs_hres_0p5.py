#!/usr/bin/env python3
"""Download IFS HRES + GFS analysis/init-time fields from WeatherBench2.

Both sources are operational forecast systems with init-time states available
at 0/6/12/18 UTC. We grab the init-time analyses (step=0) and coarsen to 0.5°
to match our ERA5 pipeline grid. Used as cross-source OOD test for trained
DC-AE Skip / FuXi / S-DYff / ModAFNO models.

WeatherBench2 URLs (verified 2026-05):
  HRES T0:  gs://weatherbench2/datasets/hres_t0/2016-2022-6h-1440x721.zarr
  HRES:     gs://weatherbench2/datasets/hres/2016-2022-0p25-init-times-evaluation-conservative.zarr
  IFS-ENS:  gs://weatherbench2/datasets/ens/2016-2022-0p25-init-times-evaluation-conservative.zarr
  GFS:      gs://weatherbench2/datasets/gfs/2016-2022-0p25-init-times.zarr

We use HRES T0 (init-time analysis-like) which mirrors ERA5 structure.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import xarray as xr
import zarr as zarrlib


DEFAULT_URLS = {
    "hres_t0": "gs://weatherbench2/datasets/hres_t0/2016-2022-6h-1440x721.zarr",
    "hres":    "gs://weatherbench2/datasets/hres/2016-2022-0p25-init-times-evaluation-conservative.zarr",
    "gfs":     "gs://weatherbench2/datasets/gfs/2016-2022-0p25-init-times.zarr",
}

PL_VARS_MAP = {
    "temperature": "t",
    "u_component_of_wind": "u",
    "v_component_of_wind": "v",
    "specific_humidity": "q",
    "geopotential": "z",
}
PL_LEVELS = [1000, 925, 850, 700]

SURFACE_VARS_MAP = {
    "2m_temperature": "t2m",
    "10m_u_component_of_wind": "u10",
    "10m_v_component_of_wind": "v10",
    "mean_sea_level_pressure": "mslp",
    "sea_surface_temperature": "sst",
    "total_cloud_cover": "tcc",
    "total_column_water_vapour": "tcwv",
}


def coarsen_2x(arr_3d: np.ndarray) -> np.ndarray:
    """arr (T, 721, 1440) → (T, 360, 720). Trims pole row (idx 720)."""
    src = arr_3d[:, :720, :]
    return src.reshape(arr_3d.shape[0], 360, 2, 720, 2).mean(axis=(2, 4))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=list(DEFAULT_URLS), required=True)
    ap.add_argument("--years", type=int, nargs="+", required=True)
    ap.add_argument("--target", default="/workspace-SR006.nfs2/weather_data/time_interpolation_0p5")
    ap.add_argument("--url", default=None,
                    help="Override source URL (otherwise picks DEFAULT_URLS[source])")
    args = ap.parse_args()

    src_url = args.url or DEFAULT_URLS[args.source]
    print(f"opening {args.source}: {src_url}")
    ds = xr.open_zarr(src_url, consolidated=True, chunks={"time": 24})
    print(f"  source dims: {dict(ds.dims)}")
    print(f"  source vars (first 10): {list(ds.data_vars)[:10]}")

    target = Path(args.target); target.mkdir(parents=True, exist_ok=True)

    # Validate it's 0.25° native
    if "latitude" in ds.dims and ds.sizes["latitude"] == 721:
        coarsen = True
    elif "latitude" in ds.dims and ds.sizes["latitude"] == 360:
        coarsen = False  # already 0.5°
        print("  source already at 0.5° — skip coarsening")
    else:
        raise ValueError(f"unexpected lat dim {ds.sizes.get('latitude')}")

    for year in args.years:
        # ===== PL =====
        pl_out = target / f"{args.source}_zarr_{year}.zarr"
        if pl_out.exists():
            print(f"  [skip PL] {pl_out} exists")
        else:
            t0 = time.time()
            print(f"\n=== {args.source.upper()} PL {year} ===")
            ds_y = ds.sel(time=str(year))
            T = ds_y.sizes["time"]
            print(f"  T={T}")
            # Find level indices for PL_LEVELS
            level_coord = ds_y["level"] if "level" in ds_y.coords else ds_y["levels"]
            level_arr = level_coord.values.astype(int)
            lvl_idx = [int(np.where(level_arr == L)[0][0]) for L in PL_LEVELS]
            print(f"  level indices: {lvl_idx} for {PL_LEVELS}")

            store = zarrlib.DirectoryStore(str(pl_out))
            root = zarrlib.open(store, mode="w")
            lat_out = np.linspace(89.75, -89.75, 360, dtype=np.float32) if coarsen else ds_y["latitude"].values.astype(np.float32)
            lon_out = np.linspace(0.0, 359.5, 720, dtype=np.float32) if coarsen else ds_y["longitude"].values.astype(np.float32)
            root.create_dataset("latitude", data=lat_out, chunks=(360,))
            root.create_dataset("longitude", data=lon_out, chunks=(720,))
            root.create_dataset("level", data=np.array(PL_LEVELS, dtype=np.int32), chunks=(4,))
            root.create_dataset("time", data=ds_y.time.values.view(np.int64), chunks=(744,))
            for k, dims in (("latitude", ["latitude"]), ("longitude", ["longitude"]),
                            ("level", ["level"]), ("time", ["time"])):
                root[k].attrs["_ARRAY_DIMENSIONS"] = dims
            root["latitude"].attrs["units"] = "degrees_north"
            root["longitude"].attrs["units"] = "degrees_east"
            root["time"].attrs["units"] = "nanoseconds since 1970-01-01"
            chunk = 744
            for v_long, v_short in PL_VARS_MAP.items():
                if v_long not in ds_y.data_vars:
                    print(f"  [skip-var] {v_long}")
                    continue
                arr = root.create_dataset(
                    v_short, shape=(T, 4, 360, 720),
                    chunks=(chunk, 4, 360, 720), dtype=np.float32, fill_value=np.nan,
                )
                arr.attrs["_ARRAY_DIMENSIONS"] = ["time", "level", "latitude", "longitude"]
                tv = time.time()
                for c0 in range(0, T, chunk):
                    c1 = min(c0 + chunk, T)
                    src = ds_y[v_long].isel(time=slice(c0, c1), level=lvl_idx).values  # (Tc, 4, lat, lon)
                    if coarsen:
                        # coarsen each level
                        coarse = np.stack([coarsen_2x(src[:, l]) for l in range(4)], axis=1)
                    else:
                        coarse = src
                    arr[c0:c1] = coarse.astype(np.float32)
                    print(f"    [{v_short}] chunk {c0}:{c1} done ({(time.time()-tv):.1f}s elapsed)", flush=True)
                print(f"  DONE {v_short}: {(time.time()-tv)/60:.1f} min", flush=True)
            zarrlib.consolidate_metadata(store)
            print(f"  PL {args.source} {year} done in {(time.time()-t0)/60:.1f} min")

        # ===== Surface =====
        surf_out = target / f"{args.source}_surface_{year}.zarr"
        if surf_out.exists():
            print(f"  [skip surface] {surf_out} exists")
            continue
        t0 = time.time()
        print(f"\n=== {args.source.upper()} surface {year} ===")
        ds_y = ds.sel(time=str(year))
        T = ds_y.sizes["time"]
        store = zarrlib.DirectoryStore(str(surf_out))
        root = zarrlib.open(store, mode="w")
        lat_out = np.linspace(89.75, -89.75, 360, dtype=np.float32) if coarsen else ds_y["latitude"].values.astype(np.float32)
        lon_out = np.linspace(0.0, 359.5, 720, dtype=np.float32) if coarsen else ds_y["longitude"].values.astype(np.float32)
        root.create_dataset("latitude", data=lat_out, chunks=(360,))
        root.create_dataset("longitude", data=lon_out, chunks=(720,))
        root.create_dataset("time", data=ds_y.time.values.view(np.int64), chunks=(744,))
        for k, dims in (("latitude", ["latitude"]), ("longitude", ["longitude"]), ("time", ["time"])):
            root[k].attrs["_ARRAY_DIMENSIONS"] = dims
        chunk = 744
        for v_long, v_short in SURFACE_VARS_MAP.items():
            if v_long not in ds_y.data_vars:
                print(f"  [skip-var] surface {v_long}")
                continue
            arr = root.create_dataset(
                v_short, shape=(T, 360, 720),
                chunks=(chunk, 360, 720), dtype=np.float32, fill_value=np.nan,
            )
            arr.attrs["_ARRAY_DIMENSIONS"] = ["time", "latitude", "longitude"]
            tv = time.time()
            for c0 in range(0, T, chunk):
                c1 = min(c0 + chunk, T)
                src = ds_y[v_long].isel(time=slice(c0, c1)).values
                coarse = coarsen_2x(src) if coarsen else src
                arr[c0:c1] = coarse.astype(np.float32)
                print(f"    [{v_short}] chunk {c0}:{c1} done ({(time.time()-tv):.1f}s elapsed)", flush=True)
            print(f"  DONE {v_short}: {(time.time()-tv)/60:.1f} min", flush=True)
        zarrlib.consolidate_metadata(store)
        print(f"  Surface {args.source} {year} done in {(time.time()-t0)/60:.1f} min")

    print("\n=== ALL DONE ===")


if __name__ == "__main__":
    main()
