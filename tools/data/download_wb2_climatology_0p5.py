#!/usr/bin/env python3
"""Download the legacy-labelled WB2 2x2 block-average climatology staging Zarr.

This downloader preserves the historical payload construction so existing
archives can be audited byte-for-byte. Its nominal 0.5-degree coordinate
labels are not the true block centres. Run
``tools/data/canonicalize_wb2_climatology.py`` immediately afterwards; ACC
evaluation deliberately rejects this staging store.

Source: gs://weatherbench2/datasets/era5-hourly-climatology/1990-2019_6h_1440x721.zarr
  - Dims: hour (4: 0/6/12/18), dayofyear (366), level (13), lat (721), lon (1440)
  - 6-hourly grid (we interpolate to 1h at eval time)

Output: /workspace-SR006.nfs2/weather_data/time_interpolation_0p5/
        climatology_1990-2019_0p5_legacy_labels.zarr
  Single Zarr with:
    - PL: (hour, dayofyear, level=4, latitude=360, longitude=720) for t,u,v,q,z
    - Surface: (hour, dayofyear, latitude=360, longitude=720) for t2m,u10,v10,mslp,sst,tcc

Used for ACC: ACC = corr(pred - clim, target - clim) per channel.
"""

import logging
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import gcsfs
import numcodecs
import numpy as np
import xarray as xr
import zarr

ROOT = "weatherbench2/datasets/era5-hourly-climatology/1990-2019_6h_1440x721.zarr"
OUTPUT_PATH = Path("/workspace-SR006.nfs2/weather_data/time_interpolation_0p5/"
                   "climatology_1990-2019_0p5_legacy_labels.zarr")

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
TARGET_LEVELS = [1000, 925, 850, 700]


def setup_logger():
    logger = logging.getLogger("wb2_clim")
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    h = logging.StreamHandler(sys.stdout)
    h.setFormatter(logging.Formatter("[%(asctime)s] %(message)s", "%H:%M:%S"))
    logger.addHandler(h)
    return logger


def coarsen_numpy(data):
    """Coarsen 2x via numpy mean over last two axes. Trims odd H/W."""
    H, W = data.shape[-2], data.shape[-1]
    if H % 2:
        data = data[..., :-1, :]
        H -= 1
    if W % 2:
        data = data[..., :, :-1]
        W -= 1
    new_shape = data.shape[:-2] + (H // 2, 2, W // 2, 2)
    return np.nanmean(data.reshape(new_shape), axis=(-3, -1))


def download_pl(fs, log):
    log.info("PL: probing source levels...")
    ds_probe = xr.open_zarr(fs.get_mapper(ROOT), consolidated=True)
    src_levels = ds_probe["level"].values
    log.info(f"  source levels: {list(src_levels)}")
    level_idx = [int(np.where(src_levels == L)[0][0]) for L in TARGET_LEVELS]
    log.info(f"  level indices for {TARGET_LEVELS}: {level_idx}")
    ds_probe.close()

    def fetch_var(wb_name, short):
        log.info(f"    START {wb_name} -> {short}")
        t_start = time.time()
        arr = zarr.open(fs.get_mapper(f"{ROOT}/{wb_name}"), mode="r")
        full_levels = arr[:]  # (4, 366, 13, 721, 1440) ~6 GB
        slab = full_levels[:, :, level_idx, :, :]
        coarsened = coarsen_numpy(slab).astype(np.float32)
        dt = time.time() - t_start
        gib = full_levels.nbytes / 1024**3
        log.info(f"    DONE {short}: shape={coarsened.shape}, "
                 f"mean={float(np.nanmean(coarsened)):.3g}, "
                 f"{gib:.2f} GiB raw, {dt:.0f}s")
        return short, coarsened

    pl_data = {}
    with ThreadPoolExecutor(max_workers=3, thread_name_prefix="clim_pl") as pool:
        futs = [pool.submit(fetch_var, n, s) for n, s in PL_VARS.items()]
        for fut in as_completed(futs):
            short, arr = fut.result()
            pl_data[short] = arr
    return pl_data


def download_surface(fs, log):
    log.info("Surface: starting...")

    def fetch_var(wb_name, short):
        log.info(f"    START {wb_name} -> {short}")
        t_start = time.time()
        arr = zarr.open(fs.get_mapper(f"{ROOT}/{wb_name}"), mode="r")
        slab = arr[:]  # (4, 366, 721, 1440) ~1.5 GB
        coarsened = coarsen_numpy(slab).astype(np.float32)
        dt = time.time() - t_start
        gib = slab.nbytes / 1024**3
        log.info(f"    DONE {short}: shape={coarsened.shape}, "
                 f"mean={float(np.nanmean(coarsened)):.3g}, "
                 f"{gib:.2f} GiB raw, {dt:.0f}s")
        return short, coarsened

    surf_data = {}
    with ThreadPoolExecutor(max_workers=3, thread_name_prefix="clim_surf") as pool:
        futs = [pool.submit(fetch_var, n, s) for n, s in SURFACE_VARS.items()]
        for fut in as_completed(futs):
            short, arr = fut.result()
            surf_data[short] = arr
    return surf_data


def main():
    log = setup_logger()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    if OUTPUT_PATH.exists():
        log.info(f"removing existing {OUTPUT_PATH}")
        shutil.rmtree(OUTPUT_PATH)

    fs = gcsfs.GCSFileSystem(token="anon")
    t_global = time.time()

    pl_data = download_pl(fs, log)
    surf_data = download_surface(fs, log)

    log.info("Building combined Dataset...")
    hours = np.array([0, 6, 12, 18], dtype=np.int32)
    doy = np.arange(1, 367, dtype=np.int32)
    levels = np.array(TARGET_LEVELS, dtype=np.int32)
    lats = np.linspace(89.75, -89.75, 360, dtype=np.float32)
    lons = np.linspace(0, 359.5, 720, dtype=np.float32)

    data_vars = {}
    for short, arr in pl_data.items():
        data_vars[short] = (["hour", "dayofyear", "level", "latitude", "longitude"], arr)
    for short, arr in surf_data.items():
        data_vars[short] = (["hour", "dayofyear", "latitude", "longitude"], arr)

    ds = xr.Dataset(
        data_vars,
        coords={"hour": hours, "dayofyear": doy, "level": levels,
                "latitude": lats, "longitude": lons},
    )
    ds.attrs["source"] = f"gs://{ROOT}"
    ds.attrs["climatology_window"] = "1990-2019"
    ds.attrs["coarsen_factor"] = 2
    ds.attrs["coordinate_status"] = "legacy_nominal_labels_not_block_centres"
    log.info(f"  Dataset built: dims={dict(ds.sizes)}")

    compressor = numcodecs.Blosc(cname="zstd", clevel=5, shuffle=numcodecs.Blosc.BITSHUFFLE)
    # Numcodecs Blosc has a hard 2 GiB per-chunk codec buffer limit.
    # Split dayofyear=366 into 4x ~92-day chunks:
    #   PL chunk = 4 * 92 * 4 * 360 * 720 * 4B = 380 MiB  (safe)
    #   Surf chunk = 4 * 92 * 360 * 720 * 4B = 95 MiB
    encoding = {}
    for v in pl_data:
        encoding[v] = {"compressor": compressor, "chunks": (4, 92, 4, 360, 720)}
    for v in surf_data:
        encoding[v] = {"compressor": compressor, "chunks": (4, 92, 360, 720)}

    log.info(f"writing {OUTPUT_PATH}...")
    import dask
    with dask.config.set(scheduler="threads"):
        ds.chunk({"dayofyear": 92}).to_zarr(str(OUTPUT_PATH), mode="w",
                                              consolidated=True, zarr_version=2,
                                              encoding=encoding)

    dt = time.time() - t_global
    disk = sum(p.stat().st_size for p in OUTPUT_PATH.rglob("*") if p.is_file()) / 1024**3
    log.info(f"=== ALL DONE: {dt/60:.1f} min, {disk:.2f} GiB on disk ===")


if __name__ == "__main__":
    main()
