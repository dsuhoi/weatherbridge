#!/usr/bin/env python3
"""WB2 chunked 0.25° → coarsen 2× → 0.5° yearly Zarr.

Source: gs://weatherbench2/datasets/era5/1959-2023_01_10-full_37-1h-0p25deg-chunk-1.zarr
  - 1h cadence, 0.25° (721×1440) native
  - All 11 variables we need (T, U, V, Q, Z + 6 surface)
  - Per-variable Zarr Array layout

Output: /workspace-SR006.nfs2/weather_data/time_interpolation_0p5/
  - zarr_YYYY.zarr (PL vars, time × level × lat × lon)
  - surface_YYYY.zarr (time × lat × lon)

Validated: coarsen-2× vs CDS native 0.5° → 0.2-0.5 K RMSE on T (1.6% of std).
"""

import argparse
import logging
import shutil
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import gcsfs
import numcodecs
import numpy as np
import xarray as xr
import zarr

ROOT = "weatherbench2/datasets/era5/1959-2023_01_10-full_37-1h-0p25deg-chunk-1.zarr"
OUTPUT_DIR = Path("/workspace-SR006.nfs2/weather_data/time_interpolation_0p5")

PL_VARS = {  # WB2 long → our short
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
TARGET_LEVELS = [1000, 925, 850, 700]
# Standard ECMWF 37 levels (ascending — verify against actual)
ECMWF_LEVELS = np.array([1, 2, 3, 5, 7, 10, 20, 30, 50, 70, 100, 125, 150, 175, 200,
                          225, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750,
                          775, 800, 825, 850, 875, 900, 925, 950, 975, 1000])
BASE_EPOCH = np.datetime64("1959-01-01T00:00:00")


def setup_logger():
    logger = logging.getLogger("wb2_dl")
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    h = logging.StreamHandler(sys.stdout)
    h.setFormatter(logging.Formatter("[%(asctime)s] %(message)s", "%H:%M:%S"))
    logger.addHandler(h)
    return logger


def coarsen_numpy(data, axis_h=-2, axis_w=-1):
    """Coarsen 2× via numpy mean (skipna). Works on (..., H, W) arrays.
    Handles odd H by trimming last row."""
    shape = data.shape
    H, W = shape[axis_h], shape[axis_w]
    # Trim to even
    if H % 2 != 0:
        slicer = [slice(None)] * data.ndim
        slicer[axis_h] = slice(None, H - 1)
        data = data[tuple(slicer)]
        H = H - 1
    if W % 2 != 0:
        slicer = [slice(None)] * data.ndim
        slicer[axis_w] = slice(None, W - 1)
        data = data[tuple(slicer)]
        W = W - 1
    # Reshape and mean over factor-2 axes
    new_shape = list(data.shape[:axis_h]) if axis_h >= 0 else list(data.shape[:axis_h])
    # For simplicity, assume shape is (..., H, W), works for both (T, L, H, W) and (T, H, W)
    new_shape = data.shape[:-2] + (H // 2, 2, W // 2, 2)
    return np.nanmean(data.reshape(new_shape), axis=(-3, -1))


def year_time_indices(year):
    """Compute hour indices into the WB2 time axis (561264 hours from 1959-01-01 00:00) for one year."""
    start = np.datetime64(f"{year}-01-01T00:00:00")
    end = np.datetime64(f"{year+1}-01-01T00:00:00")
    t0 = int((start - BASE_EPOCH) / np.timedelta64(1, "h"))
    t1 = int((end - BASE_EPOCH) / np.timedelta64(1, "h"))
    return t0, t1


def is_zarr_complete(path, expected_T, log):
    if not path.exists():
        return False
    try:
        d = xr.open_zarr(str(path), consolidated=True)
        T = d.sizes.get("time", 0)
        d.close()
        if T == expected_T:
            log.info(f"  {path.name}: complete (T={T})")
            return True
        log.warning(f"  {path.name}: partial (T={T}, expected {expected_T})")
        return False
    except Exception as e:
        log.warning(f"  {path.name}: unreadable ({e!r})")
        return False


def download_year_pl(fs, year, output_path, log):
    """Download one year of PL vars from WB2 0.25°, coarsen to 0.5°, save Zarr."""
    t0, t1 = year_time_indices(year)
    expected_T = t1 - t0
    log.info(f"  PL year {year}: time indices [{t0}..{t1}), T={expected_T}")

    if is_zarr_complete(output_path, expected_T, log):
        return

    level_idx = [int(np.where(ECMWF_LEVELS == L)[0][0]) for L in TARGET_LEVELS]
    log.info(f"  level indices: {level_idx} for {TARGET_LEVELS}")

    if output_path.exists():
        log.info(f"  removing partial {output_path}")
        shutil.rmtree(output_path)

    log.info(f"  reading + coarsening {len(PL_VARS)} PL vars (8 chunk-threads × 2 vars concurrent)...")
    t_start = time.time()

    def read_chunk_pl(args):
        arr, ti, ti_end, level_idx = args
        t_start = time.time()
        # WB2 storage chunk = (1, 37, 721, 1440) — one hour fetches ALL 37 levels.
        # Reading ALL 37 levels vs 4 levels = SAME GCS bandwidth → read full + numpy subset.
        slab_full = arr.oindex[ti:ti_end, :, :, :]  # (chunk_T, 37, 721, 1440), one GCS round-trip per time-chunk
        slab = slab_full[:, level_idx, :, :]  # numpy subset → (chunk_T, 4, 721, 1440)
        slab_05 = coarsen_numpy(slab, axis_h=-2, axis_w=-1).astype(np.float32)
        dt = time.time() - t_start
        mb = slab_full.nbytes / 1024**2  # actual GCS bandwidth (full chunk read)
        return ti, slab_05, dt, mb

    data_vars = {}
    chunk_T = 744
    chunk_ranges = [(ti, min(ti + chunk_T, t1)) for ti in range(t0, t1, chunk_T)]
    log.info(f"  {len(chunk_ranges)} chunks per var × {len(PL_VARS)} vars")

    def download_one_var(wb_name, short):
        log.info(f"    START {wb_name} → {short}")
        arr = zarr.open(fs.get_mapper(f"{ROOT}/{wb_name}"), mode="r")
        chunks_by_ti = {}
        t_var_start = time.time()
        n_done = 0
        with ThreadPoolExecutor(max_workers=8, thread_name_prefix=f"pl_{short}") as pool:
            futures = [pool.submit(read_chunk_pl, (arr, ti, ti_end, level_idx))
                       for ti, ti_end in chunk_ranges]
            for fut in as_completed(futures):
                ti, slab_05, dt, mb = fut.result()
                chunks_by_ti[ti] = slab_05
                n_done += 1
                log.info(f"      {short} chunk {ti}: {dt:.0f}s for {mb:.0f}MB ({mb/dt:.1f}MB/s) [{n_done}/{len(chunk_ranges)}]")
        full = np.concatenate([chunks_by_ti[ti] for ti, _ in chunk_ranges], axis=0)
        dt_var = time.time() - t_var_start
        log.info(f"    DONE {short}: shape={full.shape}, mean={float(np.nanmean(full)):.2f}, {dt_var/60:.1f} min")
        return short, full

    # Concurrent across 2 vars (each with 8 chunk-threads). Total parallelism = 16 reads.
    with ThreadPoolExecutor(max_workers=2, thread_name_prefix="pl_var") as outer:
        futures = [outer.submit(download_one_var, wb_name, short)
                   for wb_name, short in PL_VARS.items()]
        for fut in as_completed(futures):
            short, full = fut.result()
            data_vars[short] = (["time", "level", "latitude", "longitude"], full)

    # Build coords
    times = BASE_EPOCH + np.arange(t0, t1, dtype="timedelta64[h]")
    levels = np.array(TARGET_LEVELS, dtype=np.int32)
    lats = np.linspace(89.75, -89.75, 360, dtype=np.float32)  # 0.5° lat (after coarsen)
    lons = np.linspace(0, 359.5, 720, dtype=np.float32)
    ds = xr.Dataset(data_vars, coords={"time": times, "level": levels,
                                         "latitude": lats, "longitude": lons})
    log.info(f"  wrote dataset built: dims={dict(ds.sizes)}")

    # Write
    compressor = numcodecs.Blosc(cname="zstd", clevel=5, shuffle=numcodecs.Blosc.BITSHUFFLE)
    # Chunk time=168 (1 week) keeps each chunk under 2 GB Zstd codec limit.
    # PL chunk: 168 × 4 levels × 360 × 720 × 4B = 696 MB < 2 GB ✓
    encoding = {v: {"compressor": compressor, "chunks": (min(168, ds.sizes["time"]),
                                                          ds.sizes["level"],
                                                          ds.sizes["latitude"],
                                                          ds.sizes["longitude"])}
                for v in ds.data_vars}
    log.info(f"  writing {output_path}...")
    import dask
    with dask.config.set(scheduler="threads"):
        ds.chunk({"time": 168}).to_zarr(str(output_path), mode="w", consolidated=True,
                                          zarr_version=2, encoding=encoding)
    dt = time.time() - t_start
    disk = sum(p.stat().st_size for p in output_path.rglob("*") if p.is_file()) / 1024**3
    log.info(f"  DONE PL {year}: {dt/60:.1f} min, {disk:.2f} GiB")


def download_year_surface(fs, year, output_path, log):
    t0, t1 = year_time_indices(year)
    expected_T = t1 - t0
    log.info(f"  Surface year {year}: T={expected_T}")
    if is_zarr_complete(output_path, expected_T, log):
        return
    if output_path.exists():
        shutil.rmtree(output_path)

    log.info(f"  reading + coarsening {len(SURFACE_VARS)} surface vars (8 threads/var)...")
    t_start = time.time()

    def read_chunk_surf(args):
        arr, ti, ti_end = args
        t_start = time.time()
        slab = arr[ti:ti_end, :, :]  # (chunk_T, 721, 1440)
        slab_05 = coarsen_numpy(slab).astype(np.float32)
        dt = time.time() - t_start
        mb = slab.nbytes / 1024**2
        return ti, slab_05, dt, mb

    data_vars = {}
    chunk_T = 744
    chunk_ranges = [(ti, min(ti + chunk_T, t1)) for ti in range(t0, t1, chunk_T)]

    def download_one_surf(wb_name, short):
        log.info(f"    START {wb_name} → {short}")
        arr = zarr.open(fs.get_mapper(f"{ROOT}/{wb_name}"), mode="r")
        chunks_by_ti = {}
        t_var_start = time.time()
        n_done = 0
        with ThreadPoolExecutor(max_workers=8, thread_name_prefix=f"surf_{short}") as pool:
            futures = [pool.submit(read_chunk_surf, (arr, ti, ti_end))
                       for ti, ti_end in chunk_ranges]
            for fut in as_completed(futures):
                ti, slab_05, dt, mb = fut.result()
                chunks_by_ti[ti] = slab_05
                n_done += 1
                log.info(f"      {short} chunk {ti}: {dt:.0f}s for {mb:.0f}MB ({mb/dt:.1f}MB/s) [{n_done}/{len(chunk_ranges)}]")
        full = np.concatenate([chunks_by_ti[ti] for ti, _ in chunk_ranges], axis=0)
        dt_var = time.time() - t_var_start
        log.info(f"    DONE {short}: shape={full.shape}, mean={float(np.nanmean(full)):.2f}, {dt_var/60:.1f} min")
        return short, full

    with ThreadPoolExecutor(max_workers=3, thread_name_prefix="surf_var") as outer:
        futures = [outer.submit(download_one_surf, wb_name, short)
                   for wb_name, short in SURFACE_VARS.items()]
        for fut in as_completed(futures):
            short, full = fut.result()
            data_vars[short] = (["time", "latitude", "longitude"], full)

    times = BASE_EPOCH + np.arange(t0, t1, dtype="timedelta64[h]")
    lats = np.linspace(89.75, -89.75, 360, dtype=np.float32)
    lons = np.linspace(0, 359.5, 720, dtype=np.float32)
    ds = xr.Dataset(data_vars, coords={"time": times, "latitude": lats, "longitude": lons})

    compressor = numcodecs.Blosc(cname="zstd", clevel=5, shuffle=numcodecs.Blosc.BITSHUFFLE)
    encoding = {v: {"compressor": compressor, "chunks": (min(744, ds.sizes["time"]),
                                                          ds.sizes["latitude"],
                                                          ds.sizes["longitude"])}
                for v in ds.data_vars}
    import dask
    with dask.config.set(scheduler="threads"):
        ds.chunk({"time": 744}).to_zarr(str(output_path), mode="w", consolidated=True,
                                          zarr_version=2, encoding=encoding)
    dt = time.time() - t_start
    disk = sum(p.stat().st_size for p in output_path.rglob("*") if p.is_file()) / 1024**3
    log.info(f"  DONE Surface {year}: {dt/60:.1f} min, {disk:.2f} GiB")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--years", type=int, nargs="+",
                        default=[2014, 2015, 2016, 2017, 2018, 2019, 2020])
    parser.add_argument("--skip-pl", action="store_true")
    parser.add_argument("--skip-surface", action="store_true")
    args = parser.parse_args()

    log = setup_logger()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fs = gcsfs.GCSFileSystem(token="anon")

    log.info(f"Years: {args.years}")
    log.info(f"Output: {OUTPUT_DIR}")

    for year in args.years:
        log.info(f"=== YEAR {year} ===")
        if not args.skip_pl:
            download_year_pl(fs, year, OUTPUT_DIR / f"zarr_{year}.zarr", log)
        if not args.skip_surface:
            download_year_surface(fs, year, OUTPUT_DIR / f"surface_{year}.zarr", log)

    log.info("=== ALL DONE ===")


if __name__ == "__main__":
    main()
