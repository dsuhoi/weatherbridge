#!/usr/bin/env python3
"""Download January 2020 at native 0.5° from CDS API for validation.

Downloads:
  PL [1000, 925, 850, 700] × {T, U, V, Q, Z}
  Surface {t2m, u10, v10, mslp, sst, tcc}
  Hourly cadence, January 2020 (744 timesteps)
  Native 0.5° grid (sized 0.5°×0.5° via grid=[0.5, 0.5] CDS param)

Output: /workspace-SR006.nfs2/weather_data/cds_validation/jan2020_0p5.nc
"""

import logging
import sys
from pathlib import Path

import cdsapi

OUTPUT_DIR = Path("/workspace-SR006.nfs2/weather_data/cds_validation")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PL_VARS = [
    "temperature", "u_component_of_wind", "v_component_of_wind",
    "specific_humidity", "geopotential",
]
SURFACE_VARS = [
    "2m_temperature", "10m_u_component_of_wind", "10m_v_component_of_wind",
    "mean_sea_level_pressure", "sea_surface_temperature", "total_cloud_cover",
]
PL_LEVELS = ["1000", "925", "850", "700"]

YEAR = "2020"
MONTH = "01"
# 10 spread-out days for stable validation while staying under CDS cost limit
DAYS = ["01", "04", "07", "10", "13", "16", "19", "22", "25", "28"]
HOURS = [f"{h:02d}:00" for h in range(24)]


def setup_logger():
    logger = logging.getLogger("cds_val")
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    h = logging.StreamHandler(sys.stdout)
    h.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s", "%H:%M:%S"))
    logger.addHandler(h)
    return logger


def download_pl(client, log):
    target = OUTPUT_DIR / f"jan2020_pl_0p5.nc"
    if target.exists() and target.stat().st_size > 1_000_000:
        log.info(f"PL exists ({target.stat().st_size/1e6:.1f}MB) — skip")
        return target
    log.info(f"requesting PL: {len(PL_VARS)} vars × {len(PL_LEVELS)} levels × Jan 2020 (744h) @ 0.5°")
    client.retrieve(
        "reanalysis-era5-pressure-levels",
        {
            "product_type": ["reanalysis"],
            "variable": PL_VARS,
            "pressure_level": PL_LEVELS,
            "year": YEAR,
            "month": MONTH,
            "day": DAYS,
            "time": HOURS,
            "grid": [0.5, 0.5],
            "format": "netcdf",
            "download_format": "unarchived",
        },
        str(target),
    )
    log.info(f"PL done: {target} ({target.stat().st_size/1e6:.1f}MB)")
    return target


def download_surface(client, log):
    target = OUTPUT_DIR / f"jan2020_surface_0p5.nc"
    if target.exists() and target.stat().st_size > 1_000_000:
        log.info(f"Surface exists ({target.stat().st_size/1e6:.1f}MB) — skip")
        return target
    log.info(f"requesting Surface: {len(SURFACE_VARS)} vars × Jan 2020 (744h) @ 0.5°")
    client.retrieve(
        "reanalysis-era5-single-levels",
        {
            "product_type": ["reanalysis"],
            "variable": SURFACE_VARS,
            "year": YEAR,
            "month": MONTH,
            "day": DAYS,
            "time": HOURS,
            "grid": [0.5, 0.5],
            "format": "netcdf",
            "download_format": "unarchived",
        },
        str(target),
    )
    log.info(f"Surface done: {target} ({target.stat().st_size/1e6:.1f}MB)")
    return target


def main():
    log = setup_logger()
    log.info("CDS API client init")
    client = cdsapi.Client()
    log.info(f"Output dir: {OUTPUT_DIR}")
    pl_path = download_pl(client, log)
    surf_path = download_surface(client, log)
    log.info("=== ALL DONE ===")
    log.info(f"  {pl_path}: {pl_path.stat().st_size/1e6:.1f}MB")
    log.info(f"  {surf_path}: {surf_path.stat().st_size/1e6:.1f}MB")


if __name__ == "__main__":
    main()
