#!/usr/bin/env python3
"""Export WeatherBench2 ERA5 data into yearly local Zarr stores.

Output format is compatible with dataset.py:
- zarr_YYYY.zarr per year
- output variables like t,u,v,q,z
- dims order: time, level, latitude, longitude

This version writes by day with tqdm progress so logs show day-level ETA.
"""

import argparse
import json
import shutil
import time
from pathlib import Path
from typing import Dict, List

import pandas as pd
import xarray as xr
import zarr

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    tqdm = None


def load_cfg(path: str) -> Dict:
    cfg = json.loads(Path(path).read_text(encoding="utf-8"))

    for k in ["source_zarr", "target_dir", "years"]:
        if k not in cfg:
            raise ValueError(f"Missing required field: {k}")

    cfg.setdefault("step_hours", 1)
    cfg.setdefault(
        "source_variables",
        [
            "temperature",
            "u_component_of_wind",
            "v_component_of_wind",
            "specific_humidity",
            "geopotential",
        ],
    )
    cfg.setdefault(
        "rename_vars",
        {
            "temperature": "t",
            "u_component_of_wind": "u",
            "v_component_of_wind": "v",
            "specific_humidity": "q",
            "geopotential": "z",
        },
    )
    cfg.setdefault("pressure_levels", [1000, 950, 900, 850])
    cfg.setdefault("time_coord", "time")
    cfg.setdefault("lat_name", None)
    cfg.setdefault("lon_name", None)
    cfg.setdefault("level_name", None)
    cfg.setdefault("consolidated", True)
    cfg.setdefault("storage_options", {"token": "anon"})
    cfg.setdefault("chunk_out", {"time": 24, "level": -1, "latitude": 181, "longitude": 360})

    if int(cfg["step_hours"]) < 1:
        raise ValueError("step_hours must be >= 1")

    return cfg


def normalize_names(ds: xr.Dataset, cfg: Dict) -> xr.Dataset:
    ren = {}

    time_coord = cfg.get("time_coord")
    if time_coord and time_coord != "time" and (time_coord in ds.coords or time_coord in ds.dims):
        ren[time_coord] = "time"

    lat_name = cfg.get("lat_name")
    lon_name = cfg.get("lon_name")
    level_name = cfg.get("level_name")

    if lat_name and lat_name != "latitude" and (lat_name in ds.coords or lat_name in ds.dims):
        ren[lat_name] = "latitude"
    elif "lat" in ds.coords or "lat" in ds.dims:
        ren["lat"] = "latitude"

    if lon_name and lon_name != "longitude" and (lon_name in ds.coords or lon_name in ds.dims):
        ren[lon_name] = "longitude"
    elif "lon" in ds.coords or "lon" in ds.dims:
        ren["lon"] = "longitude"

    if level_name and level_name != "level" and (level_name in ds.coords or level_name in ds.dims):
        ren[level_name] = "level"
    else:
        for cand in ["isobaricInhPa", "pressure_level", "plev"]:
            if cand in ds.coords or cand in ds.dims:
                ren[cand] = "level"
                break

    if ren:
        ds = ds.rename(ren)

    if "time" not in ds.coords:
        raise ValueError("Could not find time coordinate in source dataset")

    ds = ds.sortby("time")
    if "level" in ds.coords:
        ds = ds.sortby("level")

    return ds


def validate_vars(ds: xr.Dataset, variables: List[str]) -> None:
    missing = [v for v in variables if v not in ds.data_vars]
    if missing:
        raise ValueError(f"Variables not found in source dataset: {missing}")


def hourly_range_for_year(year: int) -> pd.DatetimeIndex:
    return pd.date_range(
        pd.Timestamp(year=year, month=1, day=1, hour=0),
        pd.Timestamp(year=year, month=12, day=31, hour=23),
        freq="1h",
        inclusive="both",
    )


def prepare_dataset_for_year(ds_src: xr.Dataset, cfg: Dict, year: int) -> xr.Dataset:
    source_variables = list(cfg["source_variables"])
    rename_vars = dict(cfg.get("rename_vars", {}))

    validate_vars(ds_src, source_variables)
    ds = ds_src[source_variables]

    if rename_vars:
        ds = ds.rename({k: v for k, v in rename_vars.items() if k in ds.data_vars})

    if "level" in ds.coords and cfg.get("pressure_levels"):
        ds = ds.sel(level=cfg["pressure_levels"])

    yr = hourly_range_for_year(year)
    ds = ds.sel(time=slice(yr[0], yr[-1]))

    step = int(cfg["step_hours"])
    if step > 1:
        ds = ds.isel(time=slice(0, None, step))

    return ds


def maybe_transpose_and_chunk(ds: xr.Dataset, cfg: Dict) -> xr.Dataset:
    target_dims = ["time", "level", "latitude", "longitude"]
    for v in ds.data_vars:
        if all(d in ds[v].dims for d in target_dims):
            ds[v] = ds[v].transpose(*target_dims)

    chunk_out = cfg.get("chunk_out", {})
    chunk_map = {k: v for k, v in chunk_out.items() if k in ds.dims and isinstance(v, int) and v > 0}
    if chunk_map:
        ds = ds.chunk(chunk_map)
    return ds


def export_year(ds_src: xr.Dataset, cfg: Dict, year: int, out_dir: Path) -> None:
    ds = prepare_dataset_for_year(ds_src, cfg, year)

    out_path = out_dir / f"zarr_{year}.zarr"
    if out_path.exists():
        print(f"[overwrite] remove existing {out_path}")
        shutil.rmtree(out_path)

    day_starts = pd.date_range(f"{year}-01-01", f"{year}-12-31", freq="1D")
    day_starts = [d for d in day_starts if d.year == year]

    print(f"[write] {out_path}")
    print(f"        vars={list(ds.data_vars)}")
    print(f"        total_time={ds.sizes.get('time')} level={ds.sizes.get('level', '-')}")
    print(f"        daily_steps={len(day_starts)}")

    start_ts = time.time()
    first = True

    iterator = day_starts
    if tqdm is not None:
        iterator = tqdm(day_starts, desc=f"year {year}", unit="day", dynamic_ncols=True)

    for d0 in iterator:
        d1 = d0 + pd.Timedelta(days=1) - pd.Timedelta(hours=1)
        ds_day = ds.sel(time=slice(d0, d1))
        if ds_day.sizes.get("time", 0) == 0:
            continue

        ds_day = maybe_transpose_and_chunk(ds_day, cfg)

        if first:
            ds_day.to_zarr(str(out_path), mode='w', consolidated=False, zarr_version=2)
            first = False
        else:
            ds_day.to_zarr(str(out_path), mode='a', append_dim='time', consolidated=False, zarr_version=2)

        if tqdm is None:
            done_days = (d0 - day_starts[0]).days + 1
            elapsed = time.time() - start_ts
            per_day = elapsed / max(done_days, 1)
            left = max(len(day_starts) - done_days, 0)
            eta_min = (per_day * left) / 60.0
            print(f"[year {year}] {done_days}/{len(day_starts)} days, elapsed={elapsed/60:.1f}m, eta={eta_min:.1f}m")

    zarr.consolidate_metadata(str(out_path))
    elapsed = time.time() - start_ts
    print(f"[year {year}] done in {elapsed/60:.1f}m")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="WeatherBench2 ERA5 exporter")
    parser.add_argument("--config", required=True, help="Path to JSON config")
    parser.add_argument(
        "--year",
        type=int,
        default=None,
        help="Export only one year (overrides cfg years list)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_cfg(args.config)

    source = cfg["source_zarr"]
    print(f"[open] {source}")
    ds_src = xr.open_zarr(
        source,
        consolidated=bool(cfg.get("consolidated", True)),
        storage_options=cfg.get("storage_options", {}),
    )
    ds_src = normalize_names(ds_src, cfg)

    out_dir = Path(cfg["target_dir"]).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.year is not None:
        years = [int(args.year)]
    else:
        years = [int(y) for y in cfg["years"]]

    for year in years:
        export_year(ds_src, cfg, year, out_dir)

    print("[done]")


if __name__ == "__main__":
    main()
