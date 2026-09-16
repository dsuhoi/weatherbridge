#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ERA5 / CDS → Zarr downloader with monthly blocking, JSON config, robust coord handling,
logging/progress, validation, and append-to-Zarr.

Requirements:
  pip install cdsapi xarray netcdf4 zarr pandas dask tqdm

Example config.json:
{
  "start_time": "2020-01-01T00",
  "end_time": "2020-12-31T23",
  "step": 6,
  "area": "60/-20/30/40",
  "zarr_out": "era5_2020_custom_6h.zarr",

  "surface": {
    "dataset": "reanalysis-era5-single-levels",
    "product_type": "reanalysis",
    "variables": [
      "2m_temperature",
      "10m_u_component_of_wind",
      "10m_v_component_of_wind",
      "mean_sea_level_pressure"
    ],
    "format": "netcdf"
  },

  "upperair": {
    "dataset": "reanalysis-era5-pressure-levels",
    "product_type": "reanalysis",
    "variables": [
      "geopotential",
      "u_component_of_wind",
      "v_component_of_wind",
      "temperature",
      "relative_humidity"
    ],
    "levels": [1000, 925, 850, 700, 500, 300, 200, 100],
    "level_param": "pressure_level",
    "format": "netcdf"
  },

  "tmp_dir": null,
  "chunks": { "time": 240, "lat": 180, "lon": 180, "level": null },
  "compressor": { "cname": "zstd", "clevel": 5, "shuffle": 2 },

  "log_level": "INFO",
  "dry_run": false,
  "save_plan": "plan_2020.csv",
  "verify_each_block": true,
  "show_sizes": true,
  "strict_time": false
}

Launch:
  python era5_downloader.py --config config.json

Notes:
- Supports arbitrary CDS dataset names for surface (single-level-like) and upper-air (levelled) groups.
- Robust coordinate normalization: time/valid_time, latitude/longitude (or lat/lon), level aliases.
- Uses tz-naive UTC times internally to avoid .sel() KeyError.
- If strict_time=false, will reindex time to the exact expected range (introducing NaNs on gaps) instead of raising.
"""

import sys
import json
import time
import tempfile
import logging
from logging import Logger
from pathlib import Path
from typing import Optional, List, Tuple, Dict
from contextlib import contextmanager

import pandas as pd
import xarray as xr

from tqdm.auto import tqdm

try:
    import cdsapi
except Exception:
    cdsapi = None  # allow static checks / file generation without cdsapi installed


# =========================
# Logging helpers
# =========================


def setup_logger(level: str = "INFO") -> Logger:
    logger = logging.getLogger("era5_to_zarr")
    if logger.handlers:
        return logger
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    h = logging.StreamHandler(sys.stdout)
    fmt = logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s", "%H:%M:%S")
    h.setFormatter(fmt)
    logger.addHandler(h)
    return logger


@contextmanager
def section(logger: Logger, msg: str):
    logger.info(f"▶ {msg}")
    t0 = time.time()
    try:
        yield
    finally:
        dt = time.time() - t0
        logger.info(f"✔ {msg} — {dt:.1f}s")


# =========================
# Time utilities
# =========================


def parse_iso_dt(s: str) -> pd.Timestamp:
    ts = pd.to_datetime(s, utc=True)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    return ts


def monthly_blocks(
    start: pd.Timestamp, end: pd.Timestamp
) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
    start_m = pd.Timestamp(start.year, start.month, 1, tz="UTC")
    end_m_plus = pd.Timestamp(end.year, end.month, 1, tz="UTC") + pd.offsets.MonthEnd(1)
    blocks = []
    cur = start_m
    while cur < end_m_plus:
        month_start = cur
        month_end = (cur + pd.offsets.MonthEnd(1)).normalize() + pd.Timedelta(
            hours=23, minutes=59, seconds=59
        )
        block_start = max(month_start, start)
        block_end = min(month_end, end)
        if block_start <= block_end:
            blocks.append((block_start, block_end))
        cur = cur + pd.offsets.MonthBegin(1)
    return blocks


def hours_list_in_block(
    block_start: pd.Timestamp, block_end: pd.Timestamp, step_h: int
) -> List[str]:
    rng = pd.date_range(block_start, block_end, freq=f"{step_h}h", inclusive="both")
    hhmm = sorted(set([ts.strftime("%H:%M") for ts in rng]))
    return hhmm


def days_list_in_block(block_start: pd.Timestamp, block_end: pd.Timestamp) -> List[str]:
    days = pd.date_range(
        block_start.normalize(), block_end.normalize(), freq="D", inclusive="both"
    )
    return [f"{d.day:02d}" for d in days]


def to_utc_naive_range(
    start: pd.Timestamp, end: pd.Timestamp, step_h: int
) -> pd.DatetimeIndex:
    rng = pd.date_range(start, end, freq=f"{step_h}h", inclusive="both")  # tz-aware UTC
    return pd.DatetimeIndex(rng.tz_convert("UTC").tz_localize(None))


# =========================
# CDS helpers
# =========================


def cds_client() -> "cdsapi.Client":
    if cdsapi is None:
        raise RuntimeError("cdsapi is not installed. Please `pip install cdsapi`.")
    return cdsapi.Client(
        url="https://cds.climate.copernicus.eu/api",
        key="26f3f3e0-1292-420f-8bb6-6d5ced9aec80",
        quiet=True,
        verify=True,
        timeout=300,
    )


def with_retries(fn, retries=5, base_sleep=15, max_sleep=180):
    for i in range(retries):
        try:
            return fn()
        except Exception as e:
            if i == retries - 1:
                raise
            sleep_s = min(max_sleep, base_sleep * (2**i))
            print(
                f"[WARN] Attempt {i + 1}/{retries} failed: {e}. Sleeping {sleep_s}s...",
                file=sys.stderr,
            )
            time.sleep(sleep_s)


def retrieve_to_file(dataset: str, request: Dict, outfile: Path):
    c = cds_client()

    def _job():
        return c.retrieve(dataset, request)

    r = with_retries(_job)
    with_retries(lambda: r.download(str(outfile)))


# =========================
# Payload builders
# =========================


def build_payload_surface(
    years,
    months,
    days,
    times,
    variables,
    area=None,
    grid=None,
    product_type="reanalysis",
    format_="netcdf",
) -> dict:
    payload = {
        "format": format_,
        "variable": variables,
        "year": years,
        "month": months,
        "day": days,
        "time": times,
    }
    if product_type:
        payload["product_type"] = product_type
    if area:
        payload["area"] = area  # "N/W/S/E"
    if grid:
        payload["grid"] = grid  # [lat_step, lon_step], e.g. [1.5, 1.5]
    return payload


def build_payload_upperair(
    years,
    months,
    days,
    times,
    variables,
    levels,
    level_param="pressure_level",
    area=None,
    grid=None,
    product_type="reanalysis",
    format_="netcdf",
) -> dict:
    payload = {
        "format": format_,
        "variable": variables,
        level_param: [str(L) for L in levels],
        "year": years,
        "month": months,
        "day": days,
        "time": times,
    }
    if product_type:
        payload["product_type"] = product_type
    if area:
        payload["area"] = area
    if grid:
        payload["grid"] = grid
    return payload


# =========================
# Normalization & validation
# =========================


def normalize_coords_dims(ds: xr.Dataset) -> xr.Dataset:
    """
    Normalize common coordinate/dimension names from CDS/NetCDF variants:
      - time/valid_time → time
      - latitude/lat → latitude
      - longitude/lon → longitude
      - level aliases: level, isobaricInhPa, plev, pressure_level → level
    Enforce variable dim order:
      - 3D: [time, latitude, longitude]
      - 4D: [time, level, latitude, longitude]
    Sort time ascending; sort level ascending when present.
    """
    # time
    if "valid_time" in ds.coords or "valid_time" in ds.dims:
        ds = ds.rename({"valid_time": "time"})
    # Sometimes forecast-like datasets use 'time' + 'step' to form valid_time; not handled here.

    # lat/lon
    ren = {}
    if "lat" in ds.dims or "lat" in ds.coords:
        ren["lat"] = "latitude"
    if "lon" in ds.dims or "lon" in ds.coords:
        ren["lon"] = "longitude"
    if ren:
        ds = ds.rename(ren)

    # level
    level_aliases = ["level", "isobaricInhPa", "plev", "pressure_level"]
    present = [a for a in level_aliases if a in ds.dims]
    if present:
        src = present[0]
        if src != "level":
            ds = ds.rename({src: "level"})

    # ensure sorted time
    if "time" in ds.coords:
        ds = ds.sortby("time")

    # reorder variables
    for v in ds.data_vars:
        dims = list(ds[v].dims)
        want4 = ["time", "level", "latitude", "longitude"]
        want3 = ["time", "latitude", "longitude"]
        if all(d in dims for d in want4):
            ds[v] = ds[v].transpose(*want4)
        elif all(d in dims for d in want3):
            ds[v] = ds[v].transpose(*want3)

    # sort level ascending if present
    if "level" in ds.dims:
        try:
            ds = ds.sortby("level")
        except Exception:
            pass

    return ds


def validate_block(ds: xr.Dataset, step_hours: int, logger: Optional[Logger] = None):
    lg = logger or logging.getLogger("era5_to_zarr")
    assert "time" in ds.coords, "Dataset has no 'time' coordinate"
    times = pd.DatetimeIndex(ds["time"].values)
    assert len(times) > 0, "Empty time coordinate"
    expected = pd.date_range(
        times[0], times[-1], freq=f"{step_hours}h", inclusive="both"
    )
    miss = set(expected) - set(times)
    if miss:
        lg.warning(
            f"Missing {len(miss)} timestamps in block (first 5): {sorted(list(miss))[:5]}"
        )
    # dim sanity
    for v, da in ds.data_vars.items():
        dims = set(da.dims)
        if "level" in da.dims:
            if not {"time", "level", "latitude", "longitude"}.issubset(dims):
                lg.warning(f"{v}: unexpected dims {da.dims}")
        else:
            if not {"time", "latitude", "longitude"}.issubset(dims):
                lg.warning(f"{v}: unexpected dims {da.dims}")


# =========================
# Download helpers (per block)
# =========================


def download_surface_block(
    surface_cfg, years, months, days, times, area, tmp_root, tag
):
    if not surface_cfg or not surface_cfg.get("variables"):
        return None, None
    payload = build_payload_surface(
        years,
        months,
        days,
        times,
        variables=surface_cfg["variables"],
        area=area,
        grid=surface_cfg.get("grid"),
        product_type=surface_cfg.get("product_type", "reanalysis"),
        format_=surface_cfg.get("format", "netcdf"),
    )
    out_nc = tmp_root / f"surf_{tag}.nc"
    retrieve_to_file(surface_cfg["dataset"], payload, out_nc)
    ds = xr.open_dataset(out_nc)
    # Normalize coords
    ds = normalize_coords_dims(ds)
    return ds, out_nc


def download_upperair_block(
    upperair_cfg, years, months, days, times, area, tmp_root, tag, levels_per_request=2
):
    if (
        not upperair_cfg
        or not upperair_cfg.get("variables")
        or not upperair_cfg.get("levels")
    ):
        return None, None
    
    all_levels = upperair_cfg["levels"]
    datasets = []
    files = []
    
    # Разбиваем уровни на группы
    for i in range(0, len(all_levels), levels_per_request):
        level_group = all_levels[i:i + levels_per_request]
        payload = build_payload_upperair(
            years,
            months,
            days,
            times,
            variables=upperair_cfg["variables"],
            levels=level_group,
            level_param=upperair_cfg.get("level_param", "pressure_level"),
            area=area,
            grid=upperair_cfg.get("grid"),
            product_type=upperair_cfg.get("product_type", "reanalysis"),
            format_=upperair_cfg.get("format", "netcdf"),
        )
        out_nc = tmp_root / f"ua_{tag}_lvl{i//levels_per_request}.nc"
        retrieve_to_file(upperair_cfg["dataset"], payload, out_nc)
        ds = xr.open_dataset(out_nc)
        ds = normalize_coords_dims(ds)
        datasets.append(ds)
        files.append(out_nc)
    
    # Объединяем все группы уровней
    if len(datasets) > 1:
        merged = xr.merge(datasets, compat="override", join="outer")
        # Удаляем временные файлы
        for f in files:
            try:
                Path(f).unlink()
            except:
                pass
        return merged, files[-1]  # возвращаем последний файл для отслеживания
    elif len(datasets) == 1:
        return datasets[0], files[0]
    else:
        return None, None


# =========================
# Plan / dry-run
# =========================


def build_plan(start, end, step, surface_cfg, upperair_cfg, area, split_by_month=False):
    plan = []
    blocks = monthly_blocks(start, end)
    for bstart, bend in blocks:
        days = days_list_in_block(bstart, bend)
        times = hours_list_in_block(bstart, bend, step)
        years = [f"{bstart.year:04d}"]
        months = [f"{bstart.month:02d}"]
        plan.append(
            {
                "year": years[0],
                "month": months[0],
                "n_days": len(days),
                "times": times,
                "n_times": len(
                    pd.date_range(bstart, bend, freq=f"{step}h", inclusive="both")
                ),
                "surface": bool(surface_cfg and surface_cfg.get("variables")),
                "upperair": bool(
                    upperair_cfg
                    and upperair_cfg.get("variables")
                    and upperair_cfg.get("levels")
                ),
                "area": area,
            }
        )
    return plan


def save_plan(plan, path):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.suffix.lower() == ".json":
        p.write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
    elif p.suffix.lower() == ".csv":
        import csv

        with p.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=plan[0].keys())
            w.writeheader()
            w.writerows(plan)
    else:
        raise ValueError("Plan path must end with .json or .csv")


# =========================
# Zarr writing
# =========================


def build_compressor(cfg: Optional[dict]):
    """
    cfg like {"cname": "zstd", "clevel": 5, "shuffle": 2}
    Returns zarr.Blosc compressor or None if not available/unspecified.
    """
    if not cfg:
        return None
    try:
        import zarr

        cname = cfg.get("cname", "zstd")
        clevel = int(cfg.get("clevel", 5))
        shuffle = int(cfg.get("shuffle", 2))
        return zarr.Blosc(cname=cname, clevel=clevel, shuffle=shuffle)
    except Exception:
        return None


def apply_encoding(ds: xr.Dataset, compressor):
    if compressor is None:
        return {}
    enc = {}
    for v in ds.data_vars:
        enc[v] = {"compressor": compressor}
    return enc


# =========================
# Main processing
# =========================


def process_blocks_to_zarr(
    start_time: str,
    end_time: str,
    step: int,
    surface_cfg: dict,
    upperair_cfg: dict,
    zarr_out: str,
    area: Optional[str],
    tmp_dir: Optional[str],
    chunks: Dict[str, Optional[int]],
    log_level: str = "INFO",
    dry_run: bool = False,
    save_plan_path: Optional[str] = None,
    verify_each_block: bool = True,
    show_sizes: bool = True,
    strict_time: bool = False,
    compressor_cfg: Optional[dict] = None,
    levels_per_request: int = 2,
    split_by_day: bool = False,
):
    logger = setup_logger(log_level)
    start = parse_iso_dt(start_time)
    end = parse_iso_dt(end_time)
    if end < start:
        raise ValueError("end_time < start_time")

    with section(logger, "Build request plan"):
        plan = build_plan(start, end, step, surface_cfg, upperair_cfg, area)
        total_steps = sum(p["n_times"] for p in plan)
        logger.info(f"Blocks: {len(plan)}; total timesteps ~ {total_steps}")
        if save_plan_path:
            save_plan(plan, save_plan_path)
            logger.info(f"Plan saved to {save_plan_path}")
        if dry_run:
            for p in plan[: min(5, len(plan))]:
                logger.info(f"Plan sample: {p}")
            logger.info("Dry-run finished (no downloads).")
            return

    zarr_path = Path(zarr_out).resolve()
    zarr_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_root = Path(tmp_dir) if tmp_dir else Path(tempfile.mkdtemp(prefix="era5_zarr_"))

    # Prepare chunking
    time_chunk = int(chunks.get("time", 240))
    lat_chunk = int(chunks.get("lat", chunks.get("latitude", 180)))
    lon_chunk = int(chunks.get("lon", chunks.get("longitude", 180)))
    level_chunk = chunks.get("level", None)
    compressor = build_compressor(compressor_cfg)

    first_write = True
    pbar = tqdm(plan, desc="Processing blocks", unit="block")

    for p in pbar:
        years = [p["year"]]
        months = [p["month"]]
        # reconstruct block bounds
        bstart = pd.Timestamp(int(p["year"]), int(p["month"]), 1, tz="UTC")
        bend = (bstart + pd.offsets.MonthEnd(1)).normalize() + pd.Timedelta(
            hours=23, minutes=59, seconds=59
        )
        bstart = max(bstart, start)
        bend = min(bend, end)
        
        # Если split_by_day=True, разбиваем месяц на отдельные дни
        if split_by_day:
            day_range = pd.date_range(bstart.normalize(), bend.normalize(), freq="D", inclusive="both")
        else:
            day_range = [None]  # один "блок" на весь месяц
        
        for day_start in day_range:
            if day_start is not None:
                day_end = day_start + pd.Timedelta(hours=23, minutes=59, seconds=59)
                day_start = max(day_start, bstart)
                day_end = min(day_end, bend)
                if day_end < day_start:
                    continue
                days = [f"{day_start.day:02d}"]
                times = hours_list_in_block(day_start, day_end, step)
                tag = f"{day_start.year:04d}-{day_start.month:02d}-{day_start.day:02d}"
            else:
                days = days_list_in_block(bstart, bend)
                times = hours_list_in_block(bstart, bend, step)
                tag = f"{years[0]}-{months[0]}"

            # Download
            with section(logger, f"Download {tag}"):
                ds_surf, f_surf = (None, None)
                ds_ua, f_ua = (None, None)
                if p["surface"]:
                    ds_surf, f_surf = download_surface_block(
                        surface_cfg, years, months, days, times, area, tmp_root, tag
                    )
                if p["upperair"]:
                    ds_ua, f_ua = download_upperair_block(
                        upperair_cfg, years, months, days, times, area, tmp_root, tag,
                        levels_per_request=levels_per_request
                    )

            if show_sizes:
                for kind, path in [("surface", f_surf), ("upperair", f_ua)]:
                    if path and Path(path).exists():
                        size_mb = Path(path).stat().st_size / 1e6
                        logger.info(
                            f"{kind} file: {Path(path).name} size={size_mb:.1f} MB"
                        )

            # Time filtering
            if day_start is not None:
                exact_bstart = day_start
                exact_bend = day_end
            else:
                exact_bstart = bstart
                exact_bend = bend
            exact = to_utc_naive_range(exact_bstart, exact_bend, step)
            block_ds = None

            def _select_or_reindex(ds, name):
                if ds is None:
                    return None
                # If ds['time'] might be tz-naive already; ensure it's naive
                if "time" in ds.coords:
                    # Make sure it's tz-naive
                    tvals = pd.DatetimeIndex(ds["time"].values)
                    if tvals.tz is not None:
                        ds = ds.assign_coords(
                            time=tvals.tz_convert("UTC").tz_localize(None)
                        )
                try:
                    if strict_time:
                        return ds.sel(time=exact)
                    else:
                        return ds.reindex(time=exact)
                except KeyError as e:
                    # Fallback if exact labels mismatch
                    logging.warning(f"{name}: .sel failed ({e}), falling back to reindex()")
                    return ds.reindex(time=exact)

            if ds_surf is not None:
                ds_surf = _select_or_reindex(ds_surf, "surface")
                ds_surf = normalize_coords_dims(ds_surf)
                block_ds = ds_surf

            if ds_ua is not None:
                ds_ua = _select_or_reindex(ds_ua, "upperair")
                ds_ua = normalize_coords_dims(ds_ua)
                block_ds = (
                    ds_ua
                    if block_ds is None
                    else xr.merge([block_ds, ds_ua], compat="override", join="outer")
                )

            if block_ds is None:
                logger.warning(f"{tag}: empty block, skipping")
                continue

            if verify_each_block:
                validate_block(block_ds, step, logger=logger)

            # Chunking
            chunk_map = {"time": time_chunk, "latitude": lat_chunk, "longitude": lon_chunk}
            if "level" in block_ds.dims:
                chunk_map["level"] = (
                    int(level_chunk)
                    if (level_chunk is not None)
                    else max(1, block_ds.sizes["level"])
                )
            block_ds = block_ds.chunk(
                {k: v for k, v in chunk_map.items() if k in block_ds.dims}
            )

            encoding = apply_encoding(block_ds, compressor)

            with section(logger, f"Write {tag} → Zarr"):
                if first_write:
                    block_ds.to_zarr(
                        str(zarr_path), mode="w", consolidated=True, encoding=encoding
                    )
                    first_write = False
                else:
                    block_ds.to_zarr(
                        str(zarr_path),
                        mode="a",
                        append_dim="time",
                        consolidated=True,
                        encoding=encoding,
                    )
            
            # Небольшая задержка между запросами для CDS API
            if split_by_day and day_range[-1] != day_start:
                logger.info("Waiting 10s before next request...")
                time.sleep(10)

    try:
        xr.open_zarr(str(zarr_path), consolidated=True).close()
    except Exception:
        pass
    logger.info(f"[OK] Zarr written → {zarr_path}")


# =========================
# Yearly orchestration
# =========================


def process_years_to_zarr(cfg: dict):
    logger = setup_logger(cfg["log_level"])
    start = parse_iso_dt(cfg["start_time"])
    end = parse_iso_dt(cfg["end_time"])
    split_by_year = bool(cfg.get("split_by_year", False))
    split_by_month = bool(cfg.get("split_by_month", False))
    split_by_day = bool(cfg.get("split_by_day", False))
    zarr_out_template = cfg.get("zarr_out_template")

    if not split_by_year and not split_by_month:
        process_blocks_to_zarr(
            start_time=cfg["start_time"],
            end_time=cfg["end_time"],
            step=cfg["step"],
            surface_cfg=cfg.get("surface"),
            upperair_cfg=cfg.get("upperair"),
            zarr_out=cfg["zarr_out"],
            area=cfg["area"],
            tmp_dir=cfg["tmp_dir"],
            chunks=cfg["chunks"],
            log_level=cfg["log_level"],
            dry_run=cfg["dry_run"],
            save_plan_path=cfg["save_plan"],
            verify_each_block=cfg["verify_each_block"],
            show_sizes=cfg["show_sizes"],
            strict_time=cfg["strict_time"],
            compressor_cfg=cfg.get("compressor"),
            levels_per_request=cfg.get("levels_per_request", 2),
            split_by_day=split_by_day,
        )
        return

    if not zarr_out_template:
        raise ValueError(
            "split_by_year=true requires zarr_out_template, e.g. "
            "'C:/data/era5_{year}.zarr'"
        )

    for year in range(start.year, end.year + 1):
        year_start = pd.Timestamp(year, 1, 1, 0, 0, tz="UTC")
        year_end = pd.Timestamp(year, 12, 31, 23, 0, tz="UTC")
        chunk_start = max(start, year_start)
        chunk_end = min(end, year_end)
        if chunk_end < chunk_start:
            continue

        zarr_out = str(Path(zarr_out_template.format(year=year)).resolve())
        save_plan = cfg["save_plan"]
        if save_plan:
            p = Path(save_plan)
            save_plan = str((p.parent / f"{p.stem}_{year}{p.suffix}").resolve())

        if split_by_month:
            logger.info(
                f"Year {year}: processing by months {chunk_start.isoformat()} .. {chunk_end.isoformat()} -> {zarr_out}"
            )
            for month in range(1, 13):
                month_start = pd.Timestamp(year, month, 1, 0, 0, tz="UTC")
                month_end = (month_start + pd.offsets.MonthEnd(1)).normalize() + pd.Timedelta(
                    hours=23, minutes=59, seconds=59
                )
                m_start = max(month_start, chunk_start)
                m_end = min(month_end, chunk_end)
                if m_end < m_start:
                    continue
                logger.info(f"  Month {month:02d}: {m_start.isoformat()} .. {m_end.isoformat()}")
                process_blocks_to_zarr(
                    start_time=m_start.isoformat(),
                    end_time=m_end.isoformat(),
                    step=cfg["step"],
                    surface_cfg=cfg.get("surface"),
                    upperair_cfg=cfg.get("upperair"),
                    zarr_out=zarr_out,
                    area=cfg["area"],
                    tmp_dir=cfg["tmp_dir"],
                    chunks=cfg["chunks"],
                    log_level=cfg["log_level"],
                    dry_run=cfg["dry_run"],
                    save_plan_path=save_plan if month == 1 else None,
                    verify_each_block=cfg["verify_each_block"],
                    show_sizes=cfg["show_sizes"],
                    strict_time=cfg["strict_time"],
                    compressor_cfg=cfg.get("compressor"),
                    levels_per_request=cfg.get("levels_per_request", 2),
                    split_by_day=split_by_day,
                )
        else:
            logger.info(
                f"Year {year}: {chunk_start.isoformat()} .. {chunk_end.isoformat()} -> {zarr_out}"
            )
            process_blocks_to_zarr(
                start_time=chunk_start.isoformat(),
                end_time=chunk_end.isoformat(),
                step=cfg["step"],
                surface_cfg=cfg.get("surface"),
                upperair_cfg=cfg.get("upperair"),
                zarr_out=zarr_out,
                area=cfg["area"],
                tmp_dir=cfg["tmp_dir"],
                chunks=cfg["chunks"],
                log_level=cfg["log_level"],
                dry_run=cfg["dry_run"],
                save_plan_path=save_plan,
                verify_each_block=cfg["verify_each_block"],
                show_sizes=cfg["show_sizes"],
                strict_time=cfg["strict_time"],
                compressor_cfg=cfg.get("compressor"),
                levels_per_request=cfg.get("levels_per_request", 2),
                split_by_day=split_by_day,
            )


# =========================
# Config & CLI
# =========================


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    required = ["start_time", "end_time", "step"]
    for k in required:
        if k not in cfg:
            raise ValueError(f"Missing required config field: '{k}'")

    if cfg["step"] not in [1, 6, 12, 24]:
        raise ValueError("step must be one of [1, 6, 12, 24]")

    cfg.setdefault("area", None)
    cfg.setdefault("tmp_dir", None)

    # chunks
    cfg.setdefault("chunks", {})
    cfg["chunks"].setdefault("time", 240)
    # allow both lat/lon or latitude/longitude keys in config
    if "latitude" in cfg["chunks"] and "lat" not in cfg["chunks"]:
        cfg["chunks"]["lat"] = cfg["chunks"]["latitude"]
    if "longitude" in cfg["chunks"] and "lon" not in cfg["chunks"]:
        cfg["chunks"]["lon"] = cfg["chunks"]["longitude"]
    cfg["chunks"].setdefault("lat", 180)
    cfg["chunks"].setdefault("lon", 180)
    cfg["chunks"].setdefault("level", None)

    # groups
    cfg.setdefault("surface", None)
    cfg.setdefault("upperair", None)
    if cfg["surface"] is not None:
        cfg["surface"].setdefault("grid", cfg.get("grid", None))
    if cfg["upperair"] is not None:
        cfg["upperair"].setdefault("grid", cfg.get("grid", None))

    # logging & control
    cfg.setdefault("log_level", "INFO")
    cfg.setdefault("dry_run", False)
    cfg.setdefault("save_plan", None)
    cfg.setdefault("verify_each_block", True)
    cfg.setdefault("show_sizes", True)
    cfg.setdefault("strict_time", False)

    # compression
    cfg.setdefault("compressor", None)
    cfg.setdefault("split_by_year", False)
    cfg.setdefault("zarr_out_template", None)
    cfg.setdefault("zarr_out", None)

    if cfg["split_by_year"]:
        if not cfg["zarr_out_template"]:
            raise ValueError("split_by_year=true requires zarr_out_template")
    else:
        if not cfg["zarr_out"]:
            raise ValueError("zarr_out is required when split_by_year=false")

    return cfg


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="CDS → Zarr with arbitrary surface/upper-air datasets (JSON config)."
    )
    parser.add_argument("--config", required=True, help="Path to JSON config.")
    args = parser.parse_args()

    cfg = load_config(args.config)

    process_years_to_zarr(cfg)


if __name__ == "__main__":
    main()
