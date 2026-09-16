#!/usr/bin/env python3
"""Compute mean/std for surface variables across train years.

Reads /tmp/zarrs/surface_<year>.zarr files and computes per-variable
mean/std on the train years. Saves to data/surface_stats.json.

Falls back to ladcast values for known vars if data not available.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import xarray as xr


# Fallback from LaDCast (1979-2018) for variables not yet downloaded.
FALLBACK_STATS = {
    "2m_temperature": {"mean": 278.224, "std": 21.423},
    "mean_sea_level_pressure": {"mean": 100958.4, "std": 1329.3},
    "10m_u_component_of_wind": {"mean": -0.0522, "std": 5.446},
    "10m_v_component_of_wind": {"mean": 0.1889, "std": 4.673},
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="/tmp/zarrs")
    p.add_argument("--years", type=int, nargs="+", default=[2016, 2017, 2018])
    p.add_argument("--vars", nargs="+", default=["t2m", "u10", "v10", "mslp", "tisr", "tcwv"])
    p.add_argument("--out", default="/home/jovyan/dsuhoi/weather_time_interpolation/data/surface_stats.json")
    args = p.parse_args()

    data_dir = Path(args.data_dir)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Aliasing back to long names (matches what we want for the json)
    short_to_long = {
        "t2m": "2m_temperature",
        "u10": "10m_u_component_of_wind",
        "v10": "10m_v_component_of_wind",
        "mslp": "mean_sea_level_pressure",
        "tisr": "toa_incident_solar_radiation",
        "tcwv": "total_column_water_vapour",
    }

    stats = {}
    for short in args.vars:
        long_name = short_to_long.get(short, short)
        means = []
        stds = []
        ns = []
        for y in args.years:
            zarr_path = data_dir / f"surface_{y}.zarr"
            if not zarr_path.exists():
                print(f"  [{short}] year {y}: zarr not found, skipping", flush=True)
                continue
            t0 = time.time()
            ds = xr.open_zarr(zarr_path, consolidated=True)
            if short not in ds.data_vars:
                print(f"  [{short}] year {y}: var not in zarr, skipping", flush=True)
                continue
            arr = ds[short].values  # (T, H, W)
            mu = float(arr.mean())
            sd = float(arr.std())
            n = arr.size
            means.append(mu * n)
            stds.append(sd * sd * n)
            ns.append(n)
            print(f"  [{short}] year {y}: mu={mu:.4g} std={sd:.4g} n={n} ({time.time()-t0:.1f}s)", flush=True)

        if ns:
            total = sum(ns)
            mu_pool = sum(means) / total
            sd_pool = float(np.sqrt(sum(stds) / total))
            stats[long_name] = {"mean": mu_pool, "std": sd_pool}
            stats[short] = {"mean": mu_pool, "std": sd_pool}
            print(f"[{short}] pooled: mean={mu_pool:.4g} std={sd_pool:.4g}", flush=True)
        elif long_name in FALLBACK_STATS:
            stats[long_name] = FALLBACK_STATS[long_name]
            stats[short] = FALLBACK_STATS[long_name]
            print(f"[{short}] using FALLBACK: {FALLBACK_STATS[long_name]}", flush=True)
        else:
            print(f"[{short}] WARNING: no data and no fallback", flush=True)

    with open(out_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
