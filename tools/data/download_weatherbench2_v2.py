#!/usr/bin/env python3
"""WeatherBench2 ERA5 → yearly Zarr v2 (chunk-aligned bulk writes via xarray+dask).

Differences from v1:
- One write per year, not per day (no append-storm).
- chunk-aligned: chunks=(744, level, lat, lon) — month-long compressed blocks.
- Forced Blosc-Zstd compressor + bitshuffle (better ratio for float32 climate data).
- Retry with exponential back-off on transient I/O errors (S3/NFS).
- Resumable: if zarr_{year}.zarr exists and is well-formed, skip the year.
- HTTPS source (gcsfs metadata API blocked from cloud.ru).
"""

import argparse
import json
import logging
import shutil
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, List, Optional

import fsspec
import numcodecs
import pandas as pd
import xarray as xr
import zarr


def setup_logger(level: str = "INFO") -> logging.Logger:
    logger = logging.getLogger("wb2_v2")
    if logger.handlers:
        return logger
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    h = logging.StreamHandler(sys.stdout)
    fmt = logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s", "%H:%M:%S")
    h.setFormatter(fmt)
    logger.addHandler(h)
    return logger


def load_cfg(path: str) -> Dict:
    cfg = json.loads(Path(path).read_text(encoding="utf-8"))
    for k in ["source_zarr", "target_dir", "years"]:
        if k not in cfg:
            raise ValueError(f"Missing field: {k}")
    cfg.setdefault("source_variables", ["temperature", "u_component_of_wind",
                                        "v_component_of_wind", "specific_humidity",
                                        "geopotential"])
    cfg.setdefault("rename_vars", {"temperature": "t", "u_component_of_wind": "u",
                                   "v_component_of_wind": "v",
                                   "specific_humidity": "q", "geopotential": "z"})
    cfg.setdefault("pressure_levels", [1000, 950, 900, 850])
    cfg.setdefault("step_hours", 1)
    cfg.setdefault("chunk_out", {"time": 744, "level": -1, "latitude": -1, "longitude": -1})
    cfg.setdefault("compressor", {"cname": "zstd", "clevel": 5, "shuffle": "bitshuffle"})
    cfg.setdefault("storage_options", {})
    cfg.setdefault("max_retries", 5)
    cfg.setdefault("retry_initial_backoff_s", 5.0)
    cfg.setdefault("dask_scheduler", "threads")  # threads | synchronous
    cfg.setdefault("force_overwrite", False)
    return cfg


def open_source(cfg: Dict, log: logging.Logger) -> xr.Dataset:
    src = cfg["source_zarr"]
    storage_opts = cfg.get("storage_options", {})
    log.info(f"opening source: {src}")
    if src.startswith("gs://"):
        # Convert to https — gcsfs JSON metadata API is blocked from cloud.ru,
        # but raw HTTPS GETs to storage.googleapis.com work.
        src_https = src.replace("gs://", "https://storage.googleapis.com/", 1)
        log.warning(f"rewriting gs:// → {src_https}")
        src = src_https
    if src.startswith("http"):
        mapper = fsspec.get_mapper(src)
        ds = xr.open_zarr(mapper, consolidated=True)
    else:
        ds = xr.open_zarr(src, consolidated=True, storage_options=storage_opts)
    return ds


def normalize_names(ds: xr.Dataset) -> xr.Dataset:
    ren = {}
    if "lat" in ds.coords or "lat" in ds.dims:
        ren["lat"] = "latitude"
    if "lon" in ds.coords or "lon" in ds.dims:
        ren["lon"] = "longitude"
    for cand in ("isobaricInhPa", "pressure_level", "plev"):
        if cand in ds.coords or cand in ds.dims:
            ren[cand] = "level"
            break
    if ren:
        ds = ds.rename(ren)
    if "time" not in ds.coords:
        raise ValueError("source has no 'time' coord")
    ds = ds.sortby("time")
    if "level" in ds.coords:
        ds = ds.sortby("level")
    return ds


def prepare_year(ds_src: xr.Dataset, cfg: Dict, year: int) -> xr.Dataset:
    src_vars = list(cfg["source_variables"])
    missing = [v for v in src_vars if v not in ds_src.data_vars]
    if missing:
        raise ValueError(f"missing source vars: {missing}")
    ds = ds_src[src_vars]

    rn = {k: v for k, v in cfg.get("rename_vars", {}).items() if k in ds.data_vars}
    if rn:
        ds = ds.rename(rn)

    if "level" in ds.coords and cfg.get("pressure_levels"):
        ds = ds.sel(level=cfg["pressure_levels"])

    yr0 = pd.Timestamp(year=year, month=1, day=1, hour=0)
    yr1 = pd.Timestamp(year=year, month=12, day=31, hour=23)
    ds = ds.sel(time=slice(yr0, yr1))

    step = int(cfg.get("step_hours", 1))
    if step > 1:
        ds = ds.isel(time=slice(0, None, step))

    target_dims = ["time", "level", "latitude", "longitude"]
    for v in ds.data_vars:
        if all(d in ds[v].dims for d in target_dims):
            ds[v] = ds[v].transpose(*target_dims)

    return ds


def build_chunks(ds: xr.Dataset, cfg: Dict) -> Dict[str, int]:
    """Return chunk dict for ds.chunk() — uses dim sizes for -1/none."""
    chunk_out = cfg.get("chunk_out", {})
    chunks = {}
    for d in ds.dims:
        req = chunk_out.get(d)
        if req is None or (isinstance(req, int) and req <= 0):
            chunks[d] = ds.sizes[d]
        else:
            chunks[d] = min(int(req), ds.sizes[d])
    return chunks


def build_compressor(cfg: Dict) -> numcodecs.abc.Codec:
    c = cfg.get("compressor", {})
    cname = c.get("cname", "zstd")
    clevel = int(c.get("clevel", 5))
    shuf = c.get("shuffle", "bitshuffle")
    shuffle_map = {"shuffle": numcodecs.Blosc.SHUFFLE,
                   "bitshuffle": numcodecs.Blosc.BITSHUFFLE,
                   "noshuffle": numcodecs.Blosc.NOSHUFFLE}
    if isinstance(shuf, str):
        shuffle = shuffle_map.get(shuf.lower(), numcodecs.Blosc.BITSHUFFLE)
    else:
        shuffle = int(shuf)
    return numcodecs.Blosc(cname=cname, clevel=clevel, shuffle=shuffle)


def build_encoding(ds: xr.Dataset, chunks: Dict[str, int],
                   compressor: numcodecs.abc.Codec) -> Dict:
    encoding = {}
    for v in ds.data_vars:
        var_chunks = tuple(chunks[d] for d in ds[v].dims)
        encoding[v] = {"compressor": compressor, "chunks": var_chunks}
    return encoding


def is_complete_zarr(out_path: Path, expected_time_size: int, log: logging.Logger) -> bool:
    if not out_path.exists():
        return False
    try:
        ds_check = xr.open_zarr(str(out_path), consolidated=True)
        T = ds_check.sizes.get("time", 0)
        ds_check.close()
        if T == expected_time_size:
            log.info(f"  already complete: {out_path} (time={T})")
            return True
        log.warning(f"  partial zarr at {out_path}: time={T}, expected {expected_time_size} — will overwrite")
        return False
    except Exception as e:
        log.warning(f"  zarr at {out_path} unreadable ({e!r}) — will overwrite")
        return False


def export_year_with_retry(ds_src: xr.Dataset, cfg: Dict, year: int,
                           out_dir: Path, log: logging.Logger) -> None:
    ds_year = prepare_year(ds_src, cfg, year)
    expected_T = ds_year.sizes["time"]

    out_path = out_dir / f"zarr_{year}.zarr"

    if is_complete_zarr(out_path, expected_T, log) and not cfg.get("force_overwrite"):
        return

    if out_path.exists():
        log.info(f"  removing existing {out_path}")
        shutil.rmtree(out_path)

    chunks = build_chunks(ds_year, cfg)
    log.info(f"  chunk plan: {chunks}")

    ds_chunked = ds_year.chunk(chunks)
    compressor = build_compressor(cfg)
    encoding = build_encoding(ds_chunked, chunks, compressor)
    log.info(f"  encoding[{list(encoding)[0]}]: chunks={encoding[list(encoding)[0]]['chunks']}, "
             f"compressor={compressor}")
    log.info(f"  total time={expected_T}h, vars={list(ds_chunked.data_vars)}, "
             f"raw_size_GiB={(expected_T * ds_chunked.sizes.get('level',1) * ds_chunked.sizes.get('latitude',1) * ds_chunked.sizes.get('longitude',1) * 4 * len(ds_chunked.data_vars)) / 1024**3:.1f}")

    max_retries = int(cfg.get("max_retries", 5))
    backoff = float(cfg.get("retry_initial_backoff_s", 5.0))

    import dask
    sched = cfg.get("dask_scheduler", "threads")

    for attempt in range(1, max_retries + 1):
        t0 = time.time()
        try:
            log.info(f"  [year {year}] attempt {attempt}/{max_retries}, scheduler={sched}")
            with dask.config.set(scheduler=sched):
                ds_chunked.to_zarr(
                    str(out_path),
                    mode="w",
                    consolidated=True,
                    zarr_version=2,
                    encoding=encoding,
                )
            dt = time.time() - t0
            try:
                disk = sum(p.stat().st_size for p in out_path.rglob("*") if p.is_file())
                disk_gb = disk / 1024**3
            except Exception:
                disk_gb = float("nan")
            log.info(f"  [year {year}] DONE in {dt/60:.1f}min, on-disk={disk_gb:.2f} GiB, ratio={disk_gb*1024/(expected_T * 5 * 4 * 181 * 360 * 4 / 1024**2):.2f}x compression vs raw")
            return
        except Exception as e:
            tb = traceback.format_exc()
            log.error(f"  [year {year}] attempt {attempt} FAILED after {(time.time()-t0)/60:.1f}min: "
                      f"{type(e).__name__}: {str(e)[:300]}")
            log.debug(tb)
            if attempt >= max_retries:
                raise
            sleep_s = backoff * (2 ** (attempt - 1))
            log.warning(f"  [year {year}] sleeping {sleep_s:.1f}s before retry, "
                        f"cleaning partial output")
            try:
                if out_path.exists():
                    shutil.rmtree(out_path)
            except Exception as ce:
                log.warning(f"  cleanup failed: {ce}")
            time.sleep(sleep_s)


def main():
    parser = argparse.ArgumentParser(description="WeatherBench2 ERA5 v2 exporter")
    parser.add_argument("--config", required=True)
    parser.add_argument("--year", type=int, default=None,
                        help="Override years list with one year (debug)")
    args = parser.parse_args()

    cfg = load_cfg(args.config)
    log = setup_logger("INFO")

    out_dir = Path(cfg["target_dir"]).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    log.info(f"target_dir = {out_dir}")

    ds_src = open_source(cfg, log)
    ds_src = normalize_names(ds_src)
    log.info(f"source dims: {dict(ds_src.sizes)}")

    years = [int(args.year)] if args.year is not None else [int(y) for y in cfg["years"]]
    log.info(f"years to export: {years}")

    failed = []
    for y in years:
        try:
            export_year_with_retry(ds_src, cfg, y, out_dir, log)
        except Exception as e:
            log.error(f"year {y} FAILED permanently: {type(e).__name__}: {e}")
            failed.append(y)

    if failed:
        log.error(f"failed years: {failed}")
        sys.exit(1)
    log.info("all years done")


if __name__ == "__main__":
    main()
