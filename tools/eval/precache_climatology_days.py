#!/usr/bin/env python3
"""Materialise selected evaluation days for lazy ACC climatology lookup."""
from __future__ import annotations

import argparse
from datetime import date, timedelta
from pathlib import Path

import torch

from tools.eval.batch_eval_12h_memmap import _ClimatologyLookup
from weather_time_interp.normalization import zarr_store_provenance


CHANNELS_24 = [
    "T1000", "T925", "T850", "T700",
    "U1000", "U925", "U850", "U700",
    "V1000", "V925", "V850", "V700",
    "Q1000", "Q925", "Q850", "Q700",
    "Z1000", "Z925", "Z850", "Z700",
    "t2m", "u10", "v10", "mslp",
]
DAY_PICKS = (1, 8, 15, 22)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--climatology", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--year", type=int, default=2020)
    parser.add_argument(
        "--months",
        default="",
        help="Comma-separated month numbers to materialise.",
    )
    parser.add_argument("--start-doy", type=int, default=None)
    parser.add_argument("--end-doy", type=int, default=None)
    args = parser.parse_args()

    months = sorted(
        {int(value) for value in args.months.split(",") if value.strip()}
    )
    provenance = zarr_store_provenance(args.climatology)
    lookup = _ClimatologyLookup(
        Path(args.climatology),
        CHANNELS_24,
        torch.device("cpu"),
        lazy=True,
        cache_dir=Path(args.cache_dir),
        store_identity=str(provenance["cache_identity_sha256"]),
    )
    year_start = date(args.year, 1, 1)
    if args.start_doy is not None or args.end_doy is not None:
        start = int(args.start_doy or 1)
        days_in_year = (date(args.year + 1, 1, 1) - year_start).days
        end = int(args.end_doy or days_in_year)
        days = range(start, end + 1)
    elif months:
        days = []
        for month in months:
            for day in DAY_PICKS:
                current = date(args.year, month, day)
                days.append((current - year_start).days + 1)
    else:
        parser.error("provide --months or --start-doy/--end-doy")

    for doy in days:
        current = year_start + timedelta(days=int(doy) - 1)
        field = lookup.lookup(int(doy), 0.0, args.year)
        print(
            f"{current.isoformat()} doy={int(doy):03d} "
            f"shape={tuple(field.shape)}",
            flush=True,
        )


if __name__ == "__main__":
    main()
