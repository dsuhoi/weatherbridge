"""Build a tiny deterministic synthetic ERA5 fixture for EvaluationRunner tests.

Outputs (relative to repo root):

    tests/fixtures/synth_memmap/wb2_2020.bin  (T=240, C=27, H=32, W=64, fp32)
    tests/fixtures/synth_memmap/wb2_2020.json (metadata)
    tests/fixtures/synth_stats/json_stats_0p5.nc       (pressure-level stats)
    tests/fixtures/synth_stats/surface_stats_0p5.json  (surface stats)
    tests/fixtures/synth_stats/static_features_0p5.pt  (5,H,W) static features

Total disk: ~ 20 MB. Reproducible with seed=42.

Usage:
    python tests/fixtures/build_synth_memmap.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import xarray as xr

REPO = Path(__file__).resolve().parents[2]
FIX = REPO / "tests" / "fixtures"
MEM = FIX / "synth_memmap"
STA = FIX / "synth_stats"
MEM.mkdir(parents=True, exist_ok=True)
STA.mkdir(parents=True, exist_ok=True)

PL_NAMES = ["T", "U", "V", "Q", "Z"]
PL_LEVELS = [1000, 925, 850, 700]
SURF_NAMES = ["t2m", "u10", "v10", "mslp", "sst", "tcc", "tcwv"]

# Realistic-ish mu/sigma so denormalisation doesn't blow up.
PL_STATS = {
    # (mu, sigma) per (var, level)
    ("T", 1000): (288.0, 12.0), ("T", 925): (282.0, 13.0),
    ("T", 850):  (275.0, 14.0), ("T", 700):  (260.0, 15.0),
    ("U", 1000): (0.0, 8.0),    ("U", 925): (1.0, 10.0),
    ("U", 850):  (3.0, 12.0),   ("U", 700):  (5.0, 14.0),
    ("V", 1000): (0.0, 6.0),    ("V", 925): (0.0, 7.0),
    ("V", 850):  (0.0, 8.0),    ("V", 700):  (0.0, 10.0),
    ("Q", 1000): (0.008, 0.005), ("Q", 925): (0.006, 0.004),
    ("Q", 850):  (0.004, 0.003), ("Q", 700):  (0.002, 0.0015),
    ("Z", 1000): (1000.0, 1000.0), ("Z", 925): (8000.0, 800.0),
    ("Z", 850):  (15000.0, 700.0), ("Z", 700): (30000.0, 600.0),
}
SURF_STATS = {
    "t2m":  (288.0, 15.0),
    "u10":  (0.0, 5.0),
    "v10":  (0.0, 4.0),
    "mslp": (101300.0, 1000.0),
    "sst":  (290.0, 8.0),
    "tcc":  (0.5, 0.3),
    "tcwv": (25.0, 15.0),
}

T = 120        # 5 days at hourly cadence (samples_per_date=2 → 10 windows / day // tau=1..5)
H, W = 16, 32
N_PL = len(PL_NAMES) * len(PL_LEVELS)   # 20
N_SURF = len(SURF_NAMES)                # 7
N_CH = N_PL + N_SURF                    # 27


def build_data() -> np.ndarray:
    rng = np.random.default_rng(42)
    arr = np.zeros((T, N_CH, H, W), dtype=np.float32)
    # Per-channel: smooth temporal + spatial structure + small noise so
    # advection / hermite produce non-trivial differences vs bilinear.
    t_idx = np.arange(T, dtype=np.float32)
    y_idx = np.linspace(-1.0, 1.0, H, dtype=np.float32)
    x_idx = np.linspace(-1.0, 1.0, W, dtype=np.float32)
    Y, X = np.meshgrid(y_idx, x_idx, indexing="ij")

    ci = 0
    for v in PL_NAMES:
        for lvl in PL_LEVELS:
            mu, sigma = PL_STATS[(v, lvl)]
            phase = float(rng.uniform(0, 2 * np.pi))
            kx = float(rng.uniform(0.5, 2.5))
            ky = float(rng.uniform(0.5, 2.5))
            kt = float(rng.uniform(0.01, 0.08))
            spatial = np.sin(kx * np.pi * X + phase) * np.cos(ky * np.pi * Y)
            for t in range(T):
                temporal = 1.0 + 0.5 * np.sin(kt * t_idx[t] + phase)
                noise = rng.standard_normal((H, W)).astype(np.float32) * 0.05
                arr[t, ci] = mu + sigma * (spatial * temporal + noise)
            ci += 1

    for sv in SURF_NAMES:
        mu, sigma = SURF_STATS[sv]
        phase = float(rng.uniform(0, 2 * np.pi))
        kx = float(rng.uniform(0.5, 2.0))
        ky = float(rng.uniform(0.5, 2.0))
        kt = float(rng.uniform(0.01, 0.08))
        spatial = np.cos(kx * np.pi * X + phase) * np.sin(ky * np.pi * Y)
        for t in range(T):
            temporal = 1.0 + 0.4 * np.cos(kt * t_idx[t] + phase)
            noise = rng.standard_normal((H, W)).astype(np.float32) * 0.05
            arr[t, ci] = mu + sigma * (spatial * temporal + noise)
        ci += 1

    assert ci == N_CH
    return arr


def write_memmap(arr: np.ndarray) -> None:
    bin_path = MEM / "wb2_2020.bin"
    mm = np.memmap(str(bin_path), dtype=np.float32, mode="w+", shape=arr.shape)
    mm[...] = arr
    mm.flush()
    del mm
    meta = {
        "T": T, "H": H, "W": W, "n_channels": N_CH,
        "year": 2020,
        "channels_pl": [f"{v}{l}" for v in PL_NAMES for l in PL_LEVELS],
        "channels_surf": list(SURF_NAMES),
    }
    with open(MEM / "wb2_2020.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"wrote {bin_path} ({bin_path.stat().st_size/1024/1024:.1f} MiB)")


def write_stats_nc() -> None:
    """Build climate_statistics dataset with params dimension covering PL_NAMES * LEVELS."""
    params = [f"{v}{l}" for v in PL_NAMES for l in PL_LEVELS]
    means = np.array([PL_STATS[(v, l)][0] for v in PL_NAMES for l in PL_LEVELS],
                     dtype=np.float32)
    stds = np.array([PL_STATS[(v, l)][1] for v in PL_NAMES for l in PL_LEVELS],
                    dtype=np.float32)
    data = np.stack([means, stds], axis=0)  # (2, N_PL)
    ds = xr.Dataset(
        {
            "climate_statistics": xr.DataArray(
                data,
                dims=("stats", "params"),
                coords={"stats": ["mean", "std"], "params": params},
            )
        }
    )
    out = STA / "json_stats_0p5.nc"
    ds.to_netcdf(out)
    print(f"wrote {out}")


def write_surface_stats_json() -> None:
    payload = {sv: {"mean": float(mu), "std": float(sigma)}
               for sv, (mu, sigma) in SURF_STATS.items()}
    out = STA / "surface_stats_0p5.json"
    with open(out, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"wrote {out}")


def write_static_features() -> None:
    rng = np.random.default_rng(123)
    static = rng.standard_normal((5, H, W)).astype(np.float32)
    out = STA / "static_features_0p5.pt"
    torch.save(torch.from_numpy(static), out)
    print(f"wrote {out}")


def main() -> None:
    arr = build_data()
    write_memmap(arr)
    write_stats_nc()
    write_surface_stats_json()
    write_static_features()
    print("done")


if __name__ == "__main__":
    main()
