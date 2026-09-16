"""Verify downloaded WB2 0.5° per-year Zarrs: dims, ranges, sample integrity."""
import sys
import numpy as np
import xarray as xr
from pathlib import Path

ROOT = Path("/workspace-SR006.nfs2/weather_data/time_interpolation_0p5")
EXPECTED_LEVELS = [1000, 925, 850, 700]
EXPECTED_PL_VARS = ["t", "u", "v", "q", "z"]
EXPECTED_SURF_VARS = ["t2m", "u10", "v10", "mslp", "sst", "tcc"]

# Sane physical ranges
SANE = {
    "t":    (150, 340),
    "u":    (-120, 120),
    "v":    (-120, 120),
    "q":    (0, 0.04),
    "z":    (-1e4, 6e4),   # geopotential can be negative (deep cyclones, Antarctic)
    "t2m":  (180, 330),
    "u10":  (-100, 100),
    "v10":  (-100, 100),
    "mslp": (85000, 110000),
    "sst":  (265, 315),    # Arctic sea-ice formation point ~271K
    "tcc":  (0, 1.0001),
}


def verify_year(year):
    print(f"\n=== YEAR {year} ===")
    pl_path = ROOT / f"zarr_{year}.zarr"
    surf_path = ROOT / f"surface_{year}.zarr"
    if not pl_path.exists():
        print(f"  [MISSING] {pl_path.name}")
        return False
    if not surf_path.exists():
        print(f"  [MISSING] {surf_path.name}")
        return False

    issues = []
    # PL check
    ds_pl = xr.open_zarr(str(pl_path), consolidated=True)
    expected_T = 8784 if year % 4 == 0 else 8760
    print(f"  PL: dims={dict(ds_pl.sizes)}, vars={list(ds_pl.data_vars)}")
    if ds_pl.sizes.get("time", 0) != expected_T:
        issues.append(f"PL time={ds_pl.sizes.get('time')}, expected {expected_T}")
    if ds_pl.sizes.get("latitude", 0) != 360 or ds_pl.sizes.get("longitude", 0) != 720:
        issues.append(f"PL spatial={ds_pl.sizes.get('latitude')}x{ds_pl.sizes.get('longitude')}, expected 360x720")
    levels_present = sorted(int(L) for L in ds_pl.level.values) if "level" in ds_pl.coords else []
    if levels_present != sorted(EXPECTED_LEVELS):
        issues.append(f"PL levels={levels_present}, expected {EXPECTED_LEVELS}")
    missing_vars = [v for v in EXPECTED_PL_VARS if v not in ds_pl.data_vars]
    if missing_vars:
        issues.append(f"PL missing vars: {missing_vars}")

    # PL ranges (3 sample timesteps)
    for v in [vv for vv in EXPECTED_PL_VARS if vv in ds_pl.data_vars]:
        sample_t = [0, expected_T // 2, expected_T - 1]
        arr = ds_pl[v].isel(time=sample_t).values
        vmin = float(np.nanmin(arr)); vmax = float(np.nanmax(arr))
        nan_pct = 100 * np.isnan(arr).mean()
        lo, hi = SANE[v]
        status = "OK" if (lo <= vmin and vmax <= hi and nan_pct < 1) else "FAIL"
        print(f"    {v}: range [{vmin:.4g}, {vmax:.4g}] nan={nan_pct:.2f}% expect [{lo}, {hi}] {status}")
        if status == "FAIL":
            issues.append(f"PL {v}: out of range [{vmin:.4g}, {vmax:.4g}] vs [{lo}, {hi}]")

    # Surface check
    ds_surf = xr.open_zarr(str(surf_path), consolidated=True)
    print(f"  Surface: dims={dict(ds_surf.sizes)}, vars={list(ds_surf.data_vars)}")
    if ds_surf.sizes.get("time", 0) != expected_T:
        issues.append(f"Surface time={ds_surf.sizes.get('time')}, expected {expected_T}")
    missing_surf = [v for v in EXPECTED_SURF_VARS if v not in ds_surf.data_vars]
    if missing_surf:
        issues.append(f"Surface missing: {missing_surf}")
    for v in [vv for vv in EXPECTED_SURF_VARS if vv in ds_surf.data_vars]:
        sample_t = [0, expected_T // 2, expected_T - 1]
        arr = ds_surf[v].isel(time=sample_t).values
        vmin = float(np.nanmin(arr)); vmax = float(np.nanmax(arr))
        nan_pct = 100 * np.isnan(arr).mean()
        lo, hi = SANE[v]
        # SST allowed to have NaN over land (~70% of globe is ocean, so ~30% NaN expected)
        nan_threshold = 50 if v == "sst" else 1
        status = "OK" if (lo <= vmin and vmax <= hi and nan_pct < nan_threshold) else "FAIL"
        print(f"    {v}: range [{vmin:.4g}, {vmax:.4g}] nan={nan_pct:.2f}% expect [{lo}, {hi}] {status}")
        if status == "FAIL":
            issues.append(f"Surface {v}: [{vmin:.4g}, {vmax:.4g}] nan={nan_pct:.2f}%")

    ds_pl.close(); ds_surf.close()
    if issues:
        print(f"  ❌ ISSUES:")
        for i in issues:
            print(f"    - {i}")
        return False
    print(f"  ✅ PASSED")
    return True


def main():
    if not ROOT.exists():
        print(f"NO DATA DIR: {ROOT}")
        sys.exit(1)
    years_found = sorted([int(p.name.split("_")[1].split(".")[0])
                           for p in ROOT.glob("zarr_*.zarr")])
    print(f"Years available: {years_found}")
    all_ok = True
    for year in years_found:
        ok = verify_year(year)
        if not ok:
            all_ok = False
    print()
    print("=" * 60)
    if all_ok:
        print("✅ ALL YEARS PASSED")
    else:
        print("❌ SOME YEARS HAD ISSUES — see above")


if __name__ == "__main__":
    main()
