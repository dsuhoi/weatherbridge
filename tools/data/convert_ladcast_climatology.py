#!/usr/bin/env python3
"""Convert LADCast `ERA5_normal_1979_2019.json` to our project's stats files.

Outputs:
  - data/json_stats_climatology.nc   (xarray Dataset with `climate_statistics`)
  - data/surface_stats_climatology.json

Notes:
- Our pressure levels [1000, 950, 900, 850] are not all present in LADCast
  (LADCast has 1000, 925, 850, ...). Missing 950 / 900 are linearly
  interpolated from the two nearest available levels.
- `tisr` (toa_incident_solar_radiation) is absent in LADCast. We fall back to
  the sample stats from `data/surface_stats.json` if found.
"""
import argparse, json
from pathlib import Path
import numpy as np
import xarray as xr

# Mapping: our short var -> LADCast surface key
SURFACE_MAP = {
    "t2m":  "2m_temperature",
    "u10":  "10m_u_component_of_wind",
    "v10":  "10m_v_component_of_wind",
    "mslp": "mean_sea_level_pressure",
    "sst":  "sea_surface_temperature",
    "tcc":  None,  # absent in LADCast — uses GENERIC_FALLBACK
    "tcwv": None,  # absent in LADCast
    "tisr": None,  # absent in LADCast (analytic)
}

# Fallback stats for variables not in LADCast and not in sample stats.
# Used as last-resort estimates so training does not crash.
GENERIC_FALLBACK = {
    "tcc":  {"mean": 0.5,  "std": 0.3},   # cloud cover fraction [0,1]
}

# Mapping: our PL short -> LADCast PL key
PL_MAP = {
    "T": "temperature",
    "U": "u_component_of_wind",
    "V": "v_component_of_wind",
    "Q": "specific_humidity",
    "Z": "geopotential",
}


def lerp_level(stat: dict, level: int) -> float:
    """Linear-interpolate a `{level: value}` dict at the requested integer level."""
    levels = sorted(int(k) for k in stat.keys())
    if level in levels:
        return float(stat[str(level)])
    lo = max((l for l in levels if l < level), default=None)
    hi = min((l for l in levels if l > level), default=None)
    if lo is None or hi is None:
        raise ValueError(f"Cannot interpolate level {level}; available: {levels}")
    w = (level - lo) / (hi - lo)
    return float((1 - w) * float(stat[str(lo)]) + w * float(stat[str(hi)]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ladcast-json", required=True)
    ap.add_argument("--our-pl-vars",     nargs="+", default=["T", "U", "V", "Q", "Z"])
    ap.add_argument("--our-pl-levels",   nargs="+", type=int, default=[1000, 950, 900, 850])
    ap.add_argument("--our-surface-vars", nargs="+", default=["t2m", "u10", "v10", "mslp", "tisr"])
    ap.add_argument("--fallback-surface-stats",
                    default="data/surface_stats.json",
                    help="Source of fallback stats for vars missing in LADCast (e.g. tisr).")
    ap.add_argument("--out-pl-nc",      default="data/json_stats_climatology.nc")
    ap.add_argument("--out-surface-json", default="data/surface_stats_climatology.json")
    args = ap.parse_args()

    src = json.loads(Path(args.ladcast_json).read_text())

    # ── PL stats → xarray Dataset matching the existing schema ────────────
    params = []
    means, stds = [], []
    for v in args.our_pl_vars:
        ladcast_key = PL_MAP.get(v)
        if ladcast_key is None or ladcast_key not in src:
            raise KeyError(f"LADCast missing PL variable: {ladcast_key}")
        for lvl in args.our_pl_levels:
            mean = lerp_level(src[ladcast_key]["mean"], lvl)
            std  = lerp_level(src[ladcast_key]["std"],  lvl)
            params.append(f"{v}{lvl}")
            means.append(mean)
            stds.append(std)
    arr = np.array([means, stds], dtype=np.float64)
    ds = xr.Dataset(
        {"climate_statistics": (("stats", "params"), arr)},
        coords={"stats": ["mean", "std"], "params": params},
    )
    out_nc = Path(args.out_pl_nc)
    out_nc.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(out_nc)
    print(f"[ok] wrote {out_nc} :: {len(params)} params")

    # ── Surface stats → JSON in our existing format ────────────────────────
    fb_stats = {}
    fb_path = Path(args.fallback_surface_stats)
    if fb_path.exists():
        fb_stats = json.loads(fb_path.read_text())

    surface = {}
    fallback_used, climate_used = [], []
    for sv in args.our_surface_vars:
        ladcast_key = SURFACE_MAP.get(sv)
        if ladcast_key and ladcast_key in src and "mean" in src[ladcast_key]:
            entry = {
                "mean": float(src[ladcast_key]["mean"]),
                "std":  float(src[ladcast_key]["std"]),
            }
            climate_used.append(sv)
        elif sv in fb_stats:
            entry = fb_stats[sv]
            fallback_used.append(sv)
        elif sv in GENERIC_FALLBACK:
            entry = GENERIC_FALLBACK[sv]
            fallback_used.append(f"{sv}(generic)")
        else:
            raise KeyError(f"Stats for '{sv}' missing in LADCast, sample, and generic fallback")
        surface[sv] = entry
        # also write the long-name alias if the short and long differ
        if ladcast_key and ladcast_key not in surface:
            surface[ladcast_key] = entry

    Path(args.out_surface_json).write_text(json.dumps(surface, indent=2, ensure_ascii=False))
    print(f"[ok] wrote {args.out_surface_json}")
    print(f"     climatology: {climate_used}")
    print(f"     sample fallback: {fallback_used}")


if __name__ == "__main__":
    main()
