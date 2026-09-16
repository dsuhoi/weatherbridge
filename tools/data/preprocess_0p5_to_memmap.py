#!/usr/bin/env python3
"""Pre-process 0.5° year zarrs → local SSD memmap (fp16).

Each year output: /cache/wb2_0p5/wb2_YYYY.bin (T × 27 channels × 360 × 720 × 2B)
+ metadata YYYY.json with shape, dtype, channel_names.

Schema:
  - Channels 0-19: PL [T1000, T925, T850, T700, U1000, ..., Z700]
  - Channels 20-26: Surface [t2m, u10, v10, mslp, sst, tcc, tcwv]

After: ~110 GiB per year on local SSD (7 years × ~770 GB).
"""
import argparse, json, time, sys
import numpy as np
import xarray as xr
from pathlib import Path

PL_VARS = ["t", "u", "v", "q", "z"]
PL_LEVELS = [1000, 925, 850, 700]
SURF_VARS = ["t2m", "u10", "v10", "mslp", "sst", "tcc", "tcwv"]
SRC_ROOT = Path("/workspace-SR006.nfs2/weather_data/time_interpolation_0p5")
DST_ROOT = Path("/tmp/wb2_0p5_cache")


def preprocess_year(year: int):
    DST_ROOT.mkdir(parents=True, exist_ok=True)
    pl_zarr = SRC_ROOT / f"zarr_{year}.zarr"
    surf_zarr = SRC_ROOT / f"surface_{year}.zarr"

    print(f"\n=== YEAR {year} ===", flush=True)
    print(f"  loading PL: {pl_zarr}", flush=True)
    t0 = time.time()
    ds_pl = xr.open_zarr(str(pl_zarr), consolidated=True)
    ds_surf = xr.open_zarr(str(surf_zarr), consolidated=True)
    T = ds_pl.sizes["time"]
    H, W = 360, 720
    n_pl = len(PL_VARS) * len(PL_LEVELS)
    n_surf = len(SURF_VARS)
    n_ch = n_pl + n_surf
    print(f"  T={T}, channels=20+7={n_ch}, spatial={H}×{W}, fp16", flush=True)

    out_bin = DST_ROOT / f"wb2_{year}.bin"
    meta_path = DST_ROOT / f"wb2_{year}.json"
    if out_bin.exists() and meta_path.exists():
        # Check size
        expected = T * n_ch * H * W * 4  # fp32 = 4 bytes
        if out_bin.stat().st_size == expected:
            print(f"  EXISTS and size matches ({expected/1024**3:.1f} GiB) — skip", flush=True)
            return
        else:
            print(f"  EXISTS but size mismatch — overwriting", flush=True)
            out_bin.unlink()

    # Pre-allocate memmap
    print(f"  allocating memmap {out_bin}", flush=True)
    mm = np.memmap(str(out_bin), dtype=np.float32, mode="w+",
                   shape=(T, n_ch, H, W))

    # PL channels (0-19): for each var × level
    ch_idx = 0
    channel_names = []
    for var in PL_VARS:
        for li, lvl in enumerate(PL_LEVELS):
            print(f"    [{ch_idx:2d}] {var.upper()}{lvl}: loading...", flush=True)
            t_c = time.time()
            arr = ds_pl[var].isel(level=li).values  # (T, H, W) — float32
            mm[:, ch_idx] = arr.astype(np.float32)
            channel_names.append(f"{var.upper()}{lvl}")
            print(f"      done in {time.time()-t_c:.1f}s, mean={float(arr.mean()):.3g}", flush=True)
            ch_idx += 1

    # Surface channels (20-26)
    for var in SURF_VARS:
        print(f"    [{ch_idx:2d}] {var}: loading...", flush=True)
        t_c = time.time()
        arr = ds_surf[var].values  # (T, H, W)
        # NaN handling: fill with mean
        if np.isnan(arr).any():
            arr = np.nan_to_num(arr, nan=float(np.nanmean(arr)))
        mm[:, ch_idx] = arr.astype(np.float32)
        channel_names.append(var)
        print(f"      done in {time.time()-t_c:.1f}s, mean={float(arr.mean()):.3g}", flush=True)
        ch_idx += 1

    mm.flush()
    del mm

    # Write metadata
    meta = {
        "year": year, "T": int(T), "n_channels": n_ch,
        "H": H, "W": W, "dtype": "float32",
        "channel_names": channel_names,
        "shape": [int(T), int(n_ch), H, W],
        "size_bytes": int(out_bin.stat().st_size),
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  DONE {year}: {out_bin.stat().st_size/1024**3:.2f} GiB in {(time.time()-t0)/60:.1f} min",
          flush=True)
    ds_pl.close(); ds_surf.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=int, nargs="+",
                    default=[2014, 2015, 2016, 2017, 2018, 2019, 2020])
    args = ap.parse_args()
    t_global = time.time()
    print(f"target dir: {DST_ROOT}")
    for y in args.years:
        preprocess_year(y)
    print(f"\n=== ALL YEARS DONE in {(time.time()-t_global)/60:.1f} min ===", flush=True)


if __name__ == "__main__":
    main()
