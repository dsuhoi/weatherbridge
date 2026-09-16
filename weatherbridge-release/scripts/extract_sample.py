#!/usr/bin/env python3
"""Extract one normalised ERA5 interpolation window for the release tests.

Reads the fp32 memmap the paper's evaluation uses, applies exactly the
normalisation of ``ERA5MemmapDataset.__getitem__`` — per-channel mean and
standard deviation, pressure levels and surface handled separately, then
``nan_to_num`` — keeps the 24 prognostic channels and writes fp16.

fp16 is what the archive ships, so the reference errors recorded alongside it
are computed from these exact bytes: a reviewer reproduces them without
needing the original dataset.

Usage (on the machine holding the memmap):
    python scripts/extract_sample.py \\
        --memmap-dir /tmp/wb2_0p5_cache --year 2020 --t0 4368 \\
        --stats data/json_stats_0p5.nc --surface-stats data/surface_stats_0p5.json \\
        --out data/sample_era5_2020070100.npz
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

PL_NAMES = ["T", "U", "V", "Q", "Z"]
PL_LEVELS = [1000, 925, 850, 700]
SURFACE_KEPT = ["t2m", "u10", "v10", "mslp"]
N_PL = len(PL_NAMES) * len(PL_LEVELS)
N_SURFACE_STORED = 7  # sst, tcc and tcwv are stored but unused by the models
CHANNELS = [f"{v}{l}" for v in PL_NAMES for l in PL_LEVELS] + SURFACE_KEPT


def _pl_stats(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Mean and standard deviation of the 20 pressure-level channels.

    The file stores one ``climate_statistics(params, stats)`` array whose
    ``stats`` axis is (mean, std) and whose ``params`` axis is named by
    channel, so the rows have to be selected by name rather than by position.
    """
    names = [f"{v}{l}" for v in PL_NAMES for l in PL_LEVELS]
    import xarray as xr  # noqa: PLC0415

    with xr.open_dataset(str(path)) as stats:
        subset = stats["climate_statistics"].sel(params=names)
        mu = np.asarray(subset.isel(stats=0).values, dtype=np.float32)
        sigma = np.asarray(subset.isel(stats=1).values, dtype=np.float32)
    return mu.reshape(-1, 1, 1), np.maximum(sigma, 1e-6).reshape(-1, 1, 1)


def _surface_stats(path: Path) -> tuple[np.ndarray, np.ndarray]:
    stats = json.loads(path.read_text())
    mu = np.array([stats[v]["mean"] for v in SURFACE_KEPT], dtype=np.float32)
    sigma = np.array([max(stats[v]["std"], 1e-6) for v in SURFACE_KEPT], dtype=np.float32)
    return mu.reshape(-1, 1, 1), sigma.reshape(-1, 1, 1)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--memmap-dir", required=True, type=Path)
    parser.add_argument("--year", required=True, type=int)
    parser.add_argument("--t0", required=True, type=int, help="hour index of the left anchor")
    parser.add_argument("--delta-t", type=int, default=6)
    parser.add_argument("--stats", required=True, type=Path)
    parser.add_argument("--surface-stats", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    meta = json.loads((args.memmap_dir / f"wb2_{args.year}.json").read_text())
    total, height, width = meta["T"], meta["H"], meta["W"]
    n_stored = meta["n_channels"]
    array = np.memmap(
        str(args.memmap_dir / f"wb2_{args.year}.bin"),
        dtype=np.float32, mode="r", shape=(total, n_stored, height, width),
    )
    if args.t0 + args.delta_t >= total:
        raise SystemExit(f"t0={args.t0} + {args.delta_t} exceeds T={total}")

    pl_mu, pl_sigma = _pl_stats(args.stats)
    sf_mu, sf_sigma = _surface_stats(args.surface_stats)

    def state(index: int) -> np.ndarray:
        raw = np.array(array[index], dtype=np.float32)
        pl = (raw[:N_PL] - pl_mu) / pl_sigma
        surface = (raw[N_PL:N_PL + len(SURFACE_KEPT)] - sf_mu) / sf_sigma
        out = np.concatenate([pl, surface], axis=0)
        return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)

    taus = list(range(1, args.delta_t))
    payload = {
        "x0": state(args.t0).astype(np.float16),
        "xT": state(args.t0 + args.delta_t).astype(np.float16),
    }
    for tau in taus:
        payload[f"target_tau{tau}"] = state(args.t0 + tau).astype(np.float16)

    valid = datetime(args.year, 1, 1) + timedelta(hours=args.t0)
    payload["channels"] = np.array(CHANNELS)
    payload["taus"] = np.array(taus)
    payload["delta_t_hours"] = np.array(args.delta_t)
    payload["anchor_valid_time"] = np.array(valid.isoformat())
    payload["source"] = np.array(
        f"ERA5 0.5 deg via WeatherBench-2, wb2_{args.year}.bin index {args.t0}"
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(str(args.out), **payload)
    size = args.out.stat().st_size / 1e6
    print(f"wrote {args.out} ({size:.1f} MB)")
    print(f"  anchor valid time : {valid.isoformat()}Z")
    print(f"  interior hours    : {taus}")
    print(f"  grid              : {height} x {width}, {len(CHANNELS)} channels")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
