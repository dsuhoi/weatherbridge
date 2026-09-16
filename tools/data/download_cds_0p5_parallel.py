#!/usr/bin/env python3
"""Multi-threaded CDS native 0.5° ERA5 download (1h cadence, 2014-2020).

Strategy:
  - Per-month per-category (PL/surface) requests submitted concurrently
  - ThreadPoolExecutor with max_workers=5 (CDS rate-friendly)
  - Output: NetCDF per (year, month, category) to nc_tmp/
  - Resumable: skips existing complete files
  - Logs every request status

Stage 2 (post-process, run separately):
  Combine monthly NCs → per-year Zarr (zarr/ subdir)

Variables (16 channels total, matches 1° pipeline):
  PL @ [1000, 925, 850, 700]: T, U, V, Q, Z  → 5 vars × 4 levels = 20 ch
  Surface: t2m, u10, v10, mslp, sst, tcc      → 6 ch
  Total: 26 channels (TISR computed analytically downstream)
"""

import argparse
import logging
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cdsapi

OUTPUT_DIR = Path("/workspace-SR006.nfs2/weather_data/time_interpolation_0p5")
NC_DIR = OUTPUT_DIR / "nc_tmp"
ZARR_DIR = OUTPUT_DIR / "zarr"

PL_VARS = [
    "temperature", "u_component_of_wind", "v_component_of_wind",
    "specific_humidity", "geopotential",
]
SURFACE_VARS = [
    "2m_temperature", "10m_u_component_of_wind", "10m_v_component_of_wind",
    "mean_sea_level_pressure", "sea_surface_temperature", "total_cloud_cover",
]
PL_LEVELS = ["1000", "925", "850", "700"]
HOURS = [f"{h:02d}:00" for h in range(24)]
DAYS_FULL = [f"{d:02d}" for d in range(1, 32)]


def setup_logger():
    logger = logging.getLogger("cds_dl")
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    h = logging.StreamHandler(sys.stdout)
    h.setFormatter(logging.Formatter("[%(asctime)s] [%(threadName)s] %(message)s", "%H:%M:%S"))
    logger.addHandler(h)
    return logger


def days_for_month(year, month):
    import calendar
    n_days = calendar.monthrange(year, month)[1]
    return [f"{d:02d}" for d in range(1, n_days + 1)]


def chunked_days_for_month(year, month, chunk_days=10):
    """Split month days into chunks of ~10 days each (under CDS cost limit)."""
    import calendar
    n_days = calendar.monthrange(year, month)[1]
    days = list(range(1, n_days + 1))
    chunks = []
    for i in range(0, len(days), chunk_days):
        chunks.append([f"{d:02d}" for d in days[i:i + chunk_days]])
    return chunks


def file_complete(path, expected_min_mb=10):
    return path.exists() and path.stat().st_size >= expected_min_mb * 1024 * 1024


def download_one_request(args):
    year, month, chunk_idx, days_chunk, category, log, client = args
    cat_name = "pl" if category == "pl" else "surface"
    target = NC_DIR / f"{year}_{month:02d}_c{chunk_idx}_{cat_name}_0p5.nc"
    if file_complete(target, expected_min_mb=15 if category == "pl" else 4):
        log.info(f"  SKIP {target.name} (exists, {target.stat().st_size/1e6:.0f}MB)")
        return ("skipped", year, month, category, target)
    try:
        log.info(f"  REQ  {target.name} (days {days_chunk[0]}..{days_chunk[-1]}) ...")
        t0 = time.time()
        if category == "pl":
            client.retrieve(
                "reanalysis-era5-pressure-levels",
                {
                    "product_type": ["reanalysis"],
                    "variable": PL_VARS,
                    "pressure_level": PL_LEVELS,
                    "year": str(year),
                    "month": f"{month:02d}",
                    "day": days_chunk,
                    "time": HOURS,
                    "grid": [0.5, 0.5],
                    "format": "netcdf",
                    "download_format": "unarchived",
                },
                str(target),
            )
        else:  # surface
            client.retrieve(
                "reanalysis-era5-single-levels",
                {
                    "product_type": ["reanalysis"],
                    "variable": SURFACE_VARS,
                    "year": str(year),
                    "month": f"{month:02d}",
                    "day": days_chunk,
                    "time": HOURS,
                    "grid": [0.5, 0.5],
                    "format": "netcdf",
                    "download_format": "unarchived",
                },
                str(target),
            )
        dt = time.time() - t0
        size_mb = target.stat().st_size / 1e6
        log.info(f"  DONE {target.name} in {dt/60:.1f}min, {size_mb:.0f}MB")
        return ("ok", year, month, category, target)
    except Exception as e:
        log.error(f"  FAIL {target.name}: {type(e).__name__}: {str(e)[:200]}")
        log.debug(traceback.format_exc())
        if target.exists():
            try:
                target.unlink()
            except Exception:
                pass
        return ("error", year, month, category, str(e))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--years", type=int, nargs="+",
                        default=[2014, 2015, 2016, 2017, 2018, 2019, 2020])
    parser.add_argument("--months", type=int, nargs="+", default=list(range(1, 13)))
    parser.add_argument("--workers", type=int, default=5,
                        help="Parallel CDS requests (CDS limit ~10).")
    args = parser.parse_args()

    log = setup_logger()
    NC_DIR.mkdir(parents=True, exist_ok=True)
    ZARR_DIR.mkdir(parents=True, exist_ok=True)

    log.info(f"Years: {args.years}")
    log.info(f"Months: {args.months}")
    log.info(f"Workers: {args.workers}")
    log.info(f"Output: {NC_DIR}")

    # Build request queue: (year, month, chunk_idx, days_chunk, category)
    # Each chunk = ~10 days to stay under CDS cost limit.
    requests = []
    for year in args.years:
        for month in args.months:
            chunks = chunked_days_for_month(year, month, chunk_days=10)
            for ci, days in enumerate(chunks):
                for cat in ["pl", "surface"]:
                    requests.append((year, month, ci, days, cat))
    log.info(f"Total requests: {len(requests)} (10-day chunks × {len(args.years)} years × {len(args.months)} months × 2 categories)")

    # CDS client (thread-safe — each request uses its own connection)
    client = cdsapi.Client(quiet=True, wait_until_complete=True)

    log.info(f"=== Submitting {len(requests)} requests, {args.workers} parallel ===")
    t_start = time.time()
    results = {"ok": 0, "skipped": 0, "error": 0}

    with ThreadPoolExecutor(max_workers=args.workers, thread_name_prefix="cds") as pool:
        futures = [
            pool.submit(download_one_request, (year, month, ci, days, cat, log, client))
            for year, month, ci, days, cat in requests
        ]
        for fut in as_completed(futures):
            try:
                status, year, month, category, _ = fut.result()
                results[status] = results.get(status, 0) + 1
            except Exception as e:
                log.error(f"  task error: {e!r}")
                results["error"] += 1

    dt = time.time() - t_start
    log.info(f"=== ALL DONE in {dt/60:.1f}min ===")
    log.info(f"  ok={results['ok']}, skipped={results['skipped']}, error={results['error']}")

    if results["error"] > 0:
        log.error(f"  {results['error']} failed — re-run script to retry")
        sys.exit(1)


if __name__ == "__main__":
    main()
