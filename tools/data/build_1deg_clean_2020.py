#!/usr/bin/env python3
"""Rebuild 1° 2020 (clean schema) via streaming chunks.

V2: chunked processing to avoid 400+ GiB peak RAM that caused the earlier hang.
Each month (~744h) loaded → coarsen → interp → write incrementally.

Output: /workspace-SR006.nfs2/weather_data/time_interpolation_1deg_clean/
  zarr_2020.zarr     (t, u, v, q, z × [1000, 925, 850, 700], 181×360)
  surface_2020.zarr  (mslp, sst, t2m, tcc, tcwv, u10, v10 — 181×360)
"""

import gcsfs
import logging
import numcodecs
import numpy as np
import shutil
import sys
import time
import xarray as xr
import zarr
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

SRC_0P5 = Path("/workspace-SR006.nfs2/weather_data/time_interpolation_0p5")
ARCO_ROOT = "weatherbench2/datasets/era5/1959-2023_01_10-full_37-1h-0p25deg-chunk-1.zarr"
OUT_DIR = Path("/workspace-SR006.nfs2/weather_data/time_interpolation_1deg_clean")
YEAR = 2020
EXPECTED_T = 8784
BASE_EPOCH = np.datetime64("1959-01-01T00:00:00")
CHUNK_T = 744  # ~1 month

LAT_180 = np.linspace(89.5, -89.5, 180, dtype=np.float32)  # after 2× coarsen
LON_360_PRE = np.linspace(0.25, 359.75, 360, dtype=np.float32)  # after 2× coarsen of 720
LAT_DST = np.linspace(90.0, -90.0, 181, dtype=np.float32)  # target 1° grid w/ poles
LON_DST = np.linspace(0.0, 359.0, 360, dtype=np.float32)  # target 1° grid


def setup_logger():
    log = logging.getLogger("build1deg_v2")
    if log.handlers:
        return log
    log.setLevel(logging.INFO)
    h = logging.StreamHandler(sys.stdout)
    h.setFormatter(logging.Formatter("[%(asctime)s] %(message)s", "%H:%M:%S"))
    log.addHandler(h)
    return log


def coarsen_2x_lat_lon(arr):
    """arr shape: (..., 360, 720) → (..., 180, 360) via 2x nanmean."""
    H, W = arr.shape[-2], arr.shape[-1]
    new_shape = arr.shape[:-2] + (H // 2, 2, W // 2, 2)
    return np.nanmean(arr.reshape(new_shape), axis=(-3, -1))


def regrid_180_to_181(arr_180):
    """Interp from 180-lat cell-centers (89.75..-89.75) to 181 integer-degree (90..-90).

    Returns array with lat dim = 181, same trailing dims.
    Uses numpy-based linear interp per lat column (vectorized over other dims).
    """
    # arr_180 shape: (..., 180, 360)
    # We want output: (..., 181, 360)
    # Use np.interp per lon column? Faster: use xarray interp.
    other_dims = arr_180.shape[:-2]
    n_lon = arr_180.shape[-1]
    out = np.zeros(other_dims + (181, n_lon), dtype=np.float32)
    # Build coords
    for j in range(n_lon):
        # Per lon column: arr_180[..., :, j] is (..., 180) → interp to 181
        # Use np.interp (1D). But we need to handle ndim properly.
        # For full vectorization, use scipy.interpolate.interp1d once.
        pass
    # Simpler: use np.interp per (other, lon) slice. Vectorize via xarray (small).
    import xarray as _xr
    n_lat = arr_180.shape[-2]
    assert n_lat == 180, f"regrid_180_to_181 expects lat=180, got {n_lat}"
    # After 2x coarsen of 720 lon, we get 360. Coords match.
    lon_coords = LON_DST if n_lon == 360 else LON_360_PRE  # both are 360-wide here
    da = _xr.DataArray(arr_180, dims=list("abcd"[:arr_180.ndim - 2]) + ["latitude", "longitude"],
                       coords={"latitude": LAT_180, "longitude": lon_coords[:n_lon]})
    da = da.interp(latitude=LAT_DST, method="linear")
    arr_181 = da.values
    # Fill NaN at polar extrapolation
    if np.isnan(arr_181).any():
        arr_181 = np.nan_to_num(arr_181, nan=float(np.nanmean(arr_181)))
    return arr_181.astype(np.float32)


def year_time_indices(year):
    start = np.datetime64(f"{year}-01-01T00:00:00")
    end = np.datetime64(f"{year+1}-01-01T00:00:00")
    return int((start - BASE_EPOCH) / np.timedelta64(1, "h")), int((end - BASE_EPOCH) / np.timedelta64(1, "h"))


def build_pl(log):
    """Stream-coarsen 0.5° PL → 1° (181×360), 1 month chunks."""
    src = SRC_0P5 / f"zarr_{YEAR}.zarr"
    out = OUT_DIR / f"zarr_{YEAR}.zarr"
    if out.exists():
        shutil.rmtree(out)

    log.info(f"  opening {src}")
    ds = xr.open_zarr(str(src), consolidated=True)
    levels = list(map(int, ds.level.values))
    assert levels == [1000, 925, 850, 700], f"unexpected levels {levels}"
    assert ds.sizes["time"] == EXPECTED_T

    pl_vars = ["t", "u", "v", "q", "z"]
    # Iterate in time chunks; aggregate into full arrays
    chunks_per_var = {v: [] for v in pl_vars}
    n_chunks = (EXPECTED_T + CHUNK_T - 1) // CHUNK_T
    t_start = time.time()
    for ci in range(n_chunks):
        ti0 = ci * CHUNK_T
        ti1 = min(ti0 + CHUNK_T, EXPECTED_T)
        t_c = time.time()
        for v in pl_vars:
            slab = ds[v].isel(time=slice(ti0, ti1)).values  # (T_c, 4, 360, 720)
            coarsened = coarsen_2x_lat_lon(slab)  # (T_c, 4, 180, 360)
            regridded = regrid_180_to_181(coarsened)  # (T_c, 4, 181, 360)
            chunks_per_var[v].append(regridded.astype(np.float32))
        log.info(f"    chunk {ci+1}/{n_chunks} (t={ti0}..{ti1}): {time.time()-t_c:.1f}s")

    # Concatenate
    data_vars = {}
    for v in pl_vars:
        full = np.concatenate(chunks_per_var[v], axis=0)
        data_vars[v] = (["time", "level", "latitude", "longitude"], full)
        log.info(f"  {v}: shape={full.shape}, mean={float(np.nanmean(full)):.3g}")

    times = BASE_EPOCH + np.arange(*year_time_indices(YEAR), dtype="timedelta64[h]")
    levels_arr = np.array(levels, dtype=np.int32)
    coords = {"time": times, "level": levels_arr, "latitude": LAT_DST, "longitude": LON_DST}
    ds_out = xr.Dataset(data_vars, coords=coords)

    compressor = numcodecs.Blosc(cname="zstd", clevel=5, shuffle=numcodecs.Blosc.BITSHUFFLE)
    encoding = {v: {"compressor": compressor,
                    "chunks": (168, 4, 181, 360)} for v in pl_vars}
    log.info(f"  writing {out}")
    ds_out.chunk({"time": 168}).to_zarr(str(out), mode="w", consolidated=True,
                                          zarr_version=2, encoding=encoding)
    disk = sum(p.stat().st_size for p in out.rglob("*") if p.is_file()) / 1024**3
    log.info(f"  DONE PL: {disk:.2f} GiB in {(time.time()-t_start)/60:.1f} min")


def build_surface_partial(log):
    """Stream-coarsen 0.5° surface (6 vars) → 1° 181×360."""
    src = SRC_0P5 / f"surface_{YEAR}.zarr"
    out_partial = OUT_DIR / f"surface_{YEAR}_partial.zarr"
    if out_partial.exists():
        shutil.rmtree(out_partial)
    log.info(f"  opening {src}")
    ds = xr.open_zarr(str(src), consolidated=True)
    surf_vars = ["mslp", "sst", "t2m", "tcc", "u10", "v10"]
    chunks_per_var = {v: [] for v in surf_vars}
    n_chunks = (EXPECTED_T + CHUNK_T - 1) // CHUNK_T
    t_start = time.time()
    for ci in range(n_chunks):
        ti0 = ci * CHUNK_T
        ti1 = min(ti0 + CHUNK_T, EXPECTED_T)
        t_c = time.time()
        for v in surf_vars:
            slab = ds[v].isel(time=slice(ti0, ti1)).values  # (T_c, 360, 720)
            coarsened = coarsen_2x_lat_lon(slab)
            regridded = regrid_180_to_181(coarsened)
            chunks_per_var[v].append(regridded.astype(np.float32))
        log.info(f"    surf chunk {ci+1}/{n_chunks}: {time.time()-t_c:.1f}s")

    data_vars = {}
    for v in surf_vars:
        full = np.concatenate(chunks_per_var[v], axis=0)
        data_vars[v] = (["time", "latitude", "longitude"], full)
        log.info(f"  {v}: shape={full.shape}, mean={float(np.nanmean(full)):.3g}")

    times = BASE_EPOCH + np.arange(*year_time_indices(YEAR), dtype="timedelta64[h]")
    coords = {"time": times, "latitude": LAT_DST, "longitude": LON_DST}
    ds_out = xr.Dataset(data_vars, coords=coords)

    compressor = numcodecs.Blosc(cname="zstd", clevel=5, shuffle=numcodecs.Blosc.BITSHUFFLE)
    encoding = {v: {"compressor": compressor, "chunks": (744, 181, 360)} for v in surf_vars}
    log.info(f"  writing {out_partial}")
    ds_out.chunk({"time": 744}).to_zarr(str(out_partial), mode="w", consolidated=True,
                                          zarr_version=2, encoding=encoding)
    disk = sum(p.stat().st_size for p in out_partial.rglob("*") if p.is_file()) / 1024**3
    log.info(f"  DONE surface partial (6 vars): {disk:.2f} GiB in {(time.time()-t_start)/60:.1f} min")


def download_tcwv_1deg(log):
    """Stream tcwv from ARCO 0.25° → 1° (181×360)."""
    log.info("  opening ARCO tcwv...")
    fs = gcsfs.GCSFileSystem(token="anon")
    arr = zarr.open(fs.get_mapper(f"{ARCO_ROOT}/total_column_water_vapour"), mode="r")
    t0, t1 = year_time_indices(YEAR)
    log.info(f"  ARCO shape: {arr.shape}, fetching t=[{t0}..{t1})")

    chunk_ranges = [(ti, min(ti + CHUNK_T, t1)) for ti in range(t0, t1, CHUNK_T)]

    def fetch(args):
        ti, ti_end = args
        t_s = time.time()
        slab = arr[ti:ti_end, :, :]  # (T_c, 721, 1440)
        slab = slab[:, :720, :]  # drop odd polar row to get 720 lat
        # Coarsen 4× → (T_c, 180, 360)
        new_shape = slab.shape[:-2] + (180, 4, 360, 4)
        coarsened = np.nanmean(slab.reshape(new_shape), axis=(-3, -1)).astype(np.float32)
        # Pad to 181 lat
        regridded = regrid_180_to_181(coarsened)
        dt = time.time() - t_s
        return ti, regridded, dt, slab.nbytes / 1024**2

    pieces = {}
    n_done = 0
    with ThreadPoolExecutor(max_workers=4, thread_name_prefix="tcwv") as pool:
        futs = [pool.submit(fetch, (ti, ti_end)) for ti, ti_end in chunk_ranges]
        for fut in as_completed(futs):
            ti, arr_chunk, dt, mb = fut.result()
            pieces[ti] = arr_chunk
            n_done += 1
            log.info(f"    tcwv chunk t={ti}: {dt:.0f}s for {mb:.0f}MB [{n_done}/{len(chunk_ranges)}]")
    full = np.concatenate([pieces[ti] for ti, _ in chunk_ranges], axis=0)
    log.info(f"  tcwv full: {full.shape}, mean={float(np.nanmean(full)):.3g}")
    return full


def merge_surface(log):
    partial = OUT_DIR / f"surface_{YEAR}_partial.zarr"
    final = OUT_DIR / f"surface_{YEAR}.zarr"
    if final.exists():
        shutil.rmtree(final)
    log.info("  loading partial...")
    ds_partial = xr.open_zarr(str(partial), consolidated=True).load()

    tcwv = download_tcwv_1deg(log)
    assert tcwv.shape == (ds_partial.sizes["time"], 181, 360), f"tcwv shape mismatch: {tcwv.shape}"
    ds_partial["tcwv"] = (["time", "latitude", "longitude"], tcwv)
    log.info(f"  merged vars: {list(ds_partial.data_vars)}")

    compressor = numcodecs.Blosc(cname="zstd", clevel=5, shuffle=numcodecs.Blosc.BITSHUFFLE)
    encoding = {v: {"compressor": compressor, "chunks": (744, 181, 360)}
                for v in ds_partial.data_vars}
    ds_partial.chunk({"time": 744}).to_zarr(str(final), mode="w", consolidated=True,
                                             zarr_version=2, encoding=encoding)
    disk = sum(p.stat().st_size for p in final.rglob("*") if p.is_file()) / 1024**3
    log.info(f"  DONE merged surface (7 vars): {disk:.2f} GiB at {final}")
    shutil.rmtree(partial)


def main():
    log = setup_logger()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t_global = time.time()

    log.info("=== STEP A: PL (chunked) ===")
    build_pl(log)
    log.info(f"  STEP A done in {(time.time()-t_global)/60:.1f} min")

    log.info("\n=== STEP B: Surface partial (6 vars chunked) ===")
    t_b = time.time()
    build_surface_partial(log)
    log.info(f"  STEP B done in {(time.time()-t_b)/60:.1f} min")

    log.info("\n=== STEP C: tcwv from ARCO + merge ===")
    t_c = time.time()
    merge_surface(log)
    log.info(f"  STEP C done in {(time.time()-t_c)/60:.1f} min")

    log.info(f"\n=== ALL DONE in {(time.time()-t_global)/60:.1f} min ===")


if __name__ == "__main__":
    main()
