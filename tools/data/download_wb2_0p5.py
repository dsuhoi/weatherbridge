#!/usr/bin/env python3
"""WeatherBench 2 ERA5 → 0.5° yearly Zarr (coarsen 2× from native 0.25°).

Source: gs://weatherbench2/datasets/era5/1959-2022-1h-1440x721.zarr (0.25° 1h)
Target: NFS workspace /workspace-SR006.nfs2/weather_data/time_interpolation_0p5/

Variables (16 channels total):
  PL @ [1000, 850]: temperature, u/v_wind, specific_humidity, geopotential  → 10 ch
  Surface: 2m_T, 10m_U/V, mslp, sst, tcc                                    → 6 ch
  (TISR computed analytically inside dataset.py — not downloaded)

Coarsening: xarray.coarsen(lat=2, lon=2, boundary="trim").mean()
  → 0.5° = 361×720 grid (was 0.25° = 721×1440)
  Conservative averaging (standard WB2 regridding).

Outputs per year:
  zarr_YYYY.zarr     — PL vars (time, level, latitude, longitude)
  surface_YYYY.zarr  — surface vars (time, latitude, longitude)

Resumable: skips year if output already exists with correct time size.
"""

import argparse
import logging
import shutil
import sys
import time
import traceback
from pathlib import Path

import fsspec
import numcodecs
import pandas as pd
import xarray as xr

SRC_URL = (
    "https://storage.googleapis.com/weatherbench2/datasets/era5/"
    "1959-2022-1h-1440x721.zarr"
)

PL_VARS = {
    "temperature": "t",
    "u_component_of_wind": "u",
    "v_component_of_wind": "v",
    "specific_humidity": "q",
    "geopotential": "z",
}
SURFACE_VARS = {
    "2m_temperature": "t2m",
    "10m_u_component_of_wind": "u10",
    "10m_v_component_of_wind": "v10",
    "mean_sea_level_pressure": "mslp",
    "sea_surface_temperature": "sst",
    "total_cloud_cover": "tcc",
}
PRESSURE_LEVELS = [1000, 925, 850, 700]  # match 1° pipeline
COARSEN_FACTOR = 2


def setup_logger():
    logger = logging.getLogger("wb2_0p5")
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    h = logging.StreamHandler(sys.stdout)
    h.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s", "%H:%M:%S"))
    logger.addHandler(h)
    return logger


def open_source(log):
    """Open WB2 Zarr via gcsfs anonymous access (works for public buckets)."""
    import gcsfs
    bucket_path = "weatherbench2/datasets/era5/1959-2022-1h-1440x721.zarr"
    log.info(f"opening source: gcs://{bucket_path}")
    fs = gcsfs.GCSFileSystem(token="anon")
    mapper = fs.get_mapper(bucket_path)
    try:
        ds = xr.open_zarr(mapper, consolidated=True)
    except KeyError as e:
        log.warning(f"consolidated metadata missing ({e!r}); trying consolidated=False")
        ds = xr.open_zarr(mapper, consolidated=False)
    log.info(f"source dims: {dict(ds.sizes)}")
    log.info(f"source vars (first 8): {list(ds.data_vars)[:8]}")
    return ds


def coarsen_ds(ds, factor=COARSEN_FACTOR):
    # skipna=True: critical for SST (NaN over land); averages only valid pixels
    # in each 2x2 cell. boundary="trim" drops <1° at lat poles for 721→360.
    return ds.coarsen(latitude=factor, longitude=factor, boundary="trim").mean(skipna=True)


def prepare_year_pl(ds_src, year, log):
    """Select PL vars, slice pressure levels & year, coarsen to 0.5°."""
    pl_keys = list(PL_VARS.keys())
    missing = [v for v in pl_keys if v not in ds_src.data_vars]
    if missing:
        raise ValueError(f"missing PL vars in source: {missing}")
    ds = ds_src[pl_keys]
    ds = ds.sel(level=PRESSURE_LEVELS)
    yr0 = pd.Timestamp(year=year, month=1, day=1, hour=0)
    yr1 = pd.Timestamp(year=year, month=12, day=31, hour=23)
    ds = ds.sel(time=slice(yr0, yr1))
    ds = coarsen_ds(ds)
    ds = ds.rename(PL_VARS)
    target_dims = ["time", "level", "latitude", "longitude"]
    for v in ds.data_vars:
        if all(d in ds[v].dims for d in target_dims):
            ds[v] = ds[v].transpose(*target_dims)
    log.info(f"  PL prepared: dims={dict(ds.sizes)}, vars={list(ds.data_vars)}")
    return ds


def prepare_year_surface(ds_src, year, log):
    """Select surface vars, slice year, coarsen to 0.5°."""
    surf_keys = list(SURFACE_VARS.keys())
    available = [v for v in surf_keys if v in ds_src.data_vars]
    missing = [v for v in surf_keys if v not in ds_src.data_vars]
    if missing:
        log.warning(f"  missing surface vars in source (will skip): {missing}")
    ds = ds_src[available]
    yr0 = pd.Timestamp(year=year, month=1, day=1, hour=0)
    yr1 = pd.Timestamp(year=year, month=12, day=31, hour=23)
    ds = ds.sel(time=slice(yr0, yr1))
    ds = coarsen_ds(ds)
    rename_map = {k: v for k, v in SURFACE_VARS.items() if k in ds.data_vars}
    ds = ds.rename(rename_map)
    target_dims = ["time", "latitude", "longitude"]
    for v in ds.data_vars:
        if all(d in ds[v].dims for d in target_dims):
            ds[v] = ds[v].transpose(*target_dims)
    log.info(f"  surface prepared: dims={dict(ds.sizes)}, vars={list(ds.data_vars)}")
    return ds


def build_encoding(ds):
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
    return encoding


def is_complete(out_path, expected_T, log):
    if not out_path.exists():
        return False
    try:
        d = xr.open_zarr(str(out_path), consolidated=True)
        T = d.sizes.get("time", 0)
        d.close()
        if T == expected_T:
            log.info(f"  already complete: {out_path} (time={T})")
            return True
        log.warning(f"  partial: time={T}, expected={expected_T} — will overwrite")
        return False
    except Exception as e:
        log.warning(f"  unreadable: {e!r} — will overwrite")
        return False


def export_with_retry(ds, out_path, log, max_retries=5, backoff=10.0):
    expected_T = ds.sizes["time"]
    if is_complete(out_path, expected_T, log):
        return
    if out_path.exists():
        log.info(f"  removing {out_path}")
        shutil.rmtree(out_path)

    encoding = build_encoding(ds)
    import dask
    for attempt in range(1, max_retries + 1):
        t0 = time.time()
        try:
            log.info(f"  writing {out_path} (attempt {attempt}/{max_retries})...")
            with dask.config.set(scheduler="threads"):
                ds.to_zarr(str(out_path), mode="w", consolidated=True,
                           zarr_version=2, encoding=encoding)
            dt = time.time() - t0
            disk = sum(p.stat().st_size for p in out_path.rglob("*") if p.is_file())
            disk_gb = disk / 1024**3
            log.info(f"  DONE {out_path.name} in {dt/60:.1f} min, {disk_gb:.2f} GiB")
            return
        except Exception as e:
            log.error(f"  attempt {attempt} FAILED: {type(e).__name__}: {str(e)[:300]}")
            log.debug(traceback.format_exc())
            if attempt >= max_retries:
                raise
            sleep_s = backoff * (2 ** (attempt - 1))
            log.warning(f"  sleeping {sleep_s:.1f}s before retry")
            try:
                if out_path.exists():
                    shutil.rmtree(out_path)
            except Exception:
                pass
            time.sleep(sleep_s)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-dir", default="/workspace-SR006.nfs2/weather_data/time_interpolation_0p5")
    parser.add_argument("--years", type=int, nargs="+", default=[2014, 2015, 2016, 2017, 2018, 2019, 2020])
    parser.add_argument("--skip-surface", action="store_true")
    parser.add_argument("--skip-pl", action="store_true")
    args = parser.parse_args()

    log = setup_logger()
    target = Path(args.target_dir)
    target.mkdir(parents=True, exist_ok=True)

    log.info(f"target: {target}")
    log.info(f"years: {args.years}")
    log.info(f"PL levels: {PRESSURE_LEVELS}, vars: {list(PL_VARS.values())}")
    log.info(f"Surface vars: {list(SURFACE_VARS.values())}")

    ds_src = open_source(log)

    for year in args.years:
        log.info(f"=== YEAR {year} ===")
        if not args.skip_pl:
            ds_pl = prepare_year_pl(ds_src, year, log)
            export_with_retry(ds_pl, target / f"zarr_{year}.zarr", log)
        if not args.skip_surface:
            ds_surf = prepare_year_surface(ds_src, year, log)
            export_with_retry(ds_surf, target / f"surface_{year}.zarr", log)

    log.info("=== ALL YEARS DONE ===")


if __name__ == "__main__":
    main()
