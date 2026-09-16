"""Subsample WB2 forecast zarr → per-source local zarr on cloudru shared.

Usage:
  python download_wb2_forecast.py --source hres --year 2022
  python download_wb2_forecast.py --source aurora --year 2022
"""
import argparse
import warnings
warnings.filterwarnings("ignore")
from datetime import datetime
import numpy as np
import xarray as xr

from weather_time_interp.grid import (
    WB2_BLOCK_GRID_NAME,
    wb2_block_average_latitudes,
    wb2_block_average_longitudes,
)

SRC = {
    "hres":   "gs://weatherbench2/datasets/hres/2016-2022-0012-1440x721.zarr",
    "aurora": "gs://weatherbench2/datasets/aurora/2022-1440x721.zarr",
}
PL_VARS_SRC = {
    "temperature": "T",
    "u_component_of_wind": "U",
    "v_component_of_wind": "V",
    "specific_humidity": "Q",
    "geopotential": "Z",
}
# Aurora zarr has \t suffix on 3 vars — fix at open-time
AURORA_RENAMES = {
    "geopotential\t": "geopotential",
    "specific_humidity\t": "specific_humidity",
    "temperature": "temperature",   # already clean
}
SURFACE_VARS = ["2m_temperature", "10m_u_component_of_wind",
                "10m_v_component_of_wind", "mean_sea_level_pressure"]
# Target PL levels (intersect w source availability at runtime)
TARGET_PL = [1000, 925, 850, 700]
INIT_DAYS = [1, 6, 11, 16, 21, 26]      # 6 inits / month
INIT_HOUR = 0                            # 00Z

def build_init_times(year: int) -> list[np.datetime64]:
    out = []
    for m in range(1, 13):
        for d in INIT_DAYS:
            try:
                out.append(np.datetime64(datetime(year, m, d, INIT_HOUR)))
            except ValueError:
                pass
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["hres", "aurora"], required=True)
    ap.add_argument("--year", type=int, default=2022)
    ap.add_argument("--dst-root",
        default="/home/jovyan/shares/SR006.nfs2/weather_data/time_interpolation_0p5")
    ap.add_argument("--coarsen", type=int, default=2,
        help="coarsen factor lat+lon (2 = 0.25→0.5°)")
    args = ap.parse_args()

    print(f"[open] {SRC[args.source]}")
    ds = xr.open_zarr(SRC[args.source], storage_options={"token": "anon"},
                       consolidated=True)
    if args.source == "aurora":
        ds = ds.rename({k: v for k, v in AURORA_RENAMES.items() if k in ds.data_vars})

    # Filter to target inits
    all_inits = build_init_times(args.year)
    have = np.intersect1d(ds.time.values, np.array(all_inits, dtype="datetime64[ns]"))
    print(f"[inits] wanted={len(all_inits)}, present={len(have)}")
    ds = ds.sel(time=have)

    # Levels: keep target subset that source has
    src_levels = ds.level.values.tolist()
    keep_pl = [lv for lv in TARGET_PL if lv in src_levels]
    print(f"[levels] source={src_levels}, keep={keep_pl}")
    ds = ds.sel(level=keep_pl)

    # Vars: PL + surface
    keep_vars = list(PL_VARS_SRC.keys()) + SURFACE_VARS
    keep_vars = [v for v in keep_vars if v in ds.data_vars]
    print(f"[vars] {keep_vars}")
    ds = ds[keep_vars]

    # Canonical ERA5 memmaps are north-to-south. WB2 forecast archives are
    # south-to-north, so reverse before trimming the south-pole row.
    latitude = np.asarray(ds.latitude.values, dtype=np.float64)
    longitude = np.asarray(ds.longitude.values, dtype=np.float64)
    if not np.array_equal(latitude, np.linspace(-90.0, 90.0, 721)):
        raise ValueError("unexpected forecast latitude coordinate")
    if not np.array_equal(longitude, np.arange(1440) * 0.25):
        raise ValueError("unexpected forecast longitude coordinate")
    ds = ds.isel(latitude=slice(None, None, -1))

    # Coarsen to the exact target grid (721 -> 720 -> 360 latitude rows).
    if args.coarsen > 1:
        # After reversal the trimmed row is the south pole, matching ERA5.
        if ds.sizes["latitude"] % args.coarsen != 0:
            ds = ds.isel(latitude=slice(0, -(ds.sizes["latitude"] % args.coarsen)))
        ds = ds.coarsen(latitude=args.coarsen, longitude=args.coarsen).mean()
    np.testing.assert_array_equal(
        ds.latitude.values,
        wb2_block_average_latitudes(),
    )
    np.testing.assert_array_equal(
        ds.longitude.values,
        wb2_block_average_longitudes(),
    )
    ds.attrs.update(
        source_uri=SRC[args.source],
        target_grid=WB2_BLOCK_GRID_NAME,
        latitude_order="north_to_south",
        spatial_transform=(
            "reverse_latitude_then_drop_south_pole_then_"
            "unweighted_2x2_block_average"
        ),
    )
    print(f"[dims after coarsen] {dict(ds.sizes)}")

    # fp16 to save space
    for v in ds.data_vars:
        ds[v] = ds[v].astype("float32")

    dst = f"{args.dst_root}/forecasts_{args.source}_{args.year}_0p5_subsample.zarr"
    print(f"[write] {dst}")
    delayed = ds.chunk({"time": 1, "prediction_timedelta": -1,
                        "level": -1, "latitude": -1, "longitude": -1})
    delayed.to_zarr(dst, mode="w", consolidated=True, compute=True)
    print("[done]")

if __name__ == "__main__":
    main()
