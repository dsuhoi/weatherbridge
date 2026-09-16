"""Quick sanity check: CDS native 0.5° → coarsen 2× → 1°, compare to existing NFS 1°.

Both datasets are ERA5 reanalysis for Jan 2020. The 1° NFS data was downloaded
from WB2 `1959-2022-1h-360x181_equiangular_with_poles_conservative.zarr` (which
uses conservative regridding from 0.25° native). CDS direct returns native
0.5° via grid=[0.5, 0.5] (bilinear interpolation from spectral truncation).

If our coarsen method (xarray.coarsen.mean) produces 1° output close to WB2's
1° conservative regrid, this validates that simple coarsening is a reasonable
approximation of paper-standard conservative regridding.

Sample: 1 timestep (2020-01-01 00:00) on 4 variables.
"""

import numpy as np
import xarray as xr
from pathlib import Path

# Inputs
CDS_PL = Path("/workspace-SR006.nfs2/weather_data/cds_validation/jan2020_pl_0p5.nc")
CDS_SURF = Path("/workspace-SR006.nfs2/weather_data/cds_validation/jan2020_surface_0p5.nc")
NFS_1DEG_PL = Path("/workspace-SR006.nfs2/weather_data/time_interpolation/zarr_2020.zarr")
NFS_1DEG_SURF = Path("/workspace-SR006.nfs2/weather_data/time_interpolation/surface_2020.zarr")


def main():
    print("=== Loading CDS native 0.5° ===")
    cds_pl = xr.open_dataset(str(CDS_PL), engine="netcdf4")
    cds_surf = xr.open_dataset(str(CDS_SURF), engine="netcdf4")
    print(f"CDS PL: dims={dict(cds_pl.sizes)}, lats {cds_pl.latitude.values[0]:.2f}..{cds_pl.latitude.values[-1]:.2f}")
    print(f"CDS Surf vars: {list(cds_surf.data_vars)}")

    print()
    print("=== Loading existing NFS 1° (WB2-conservative-regrid) ===")
    nfs_pl = xr.open_zarr(str(NFS_1DEG_PL), consolidated=True)
    nfs_surf = xr.open_zarr(str(NFS_1DEG_SURF), consolidated=True)
    print(f"NFS PL: dims={dict(nfs_pl.sizes)}, vars={list(nfs_pl.data_vars)}")
    print(f"NFS Surf: dims={dict(nfs_surf.sizes)}, vars={list(nfs_surf.data_vars)}")
    print(f"NFS lats: {nfs_pl.latitude.values[0]:.2f}..{nfs_pl.latitude.values[-1]:.2f}")

    # Coarsen CDS 0.5° → 1° (factor 2)
    print()
    print("=== Coarsen CDS 0.5° → 1° (factor 2, mean, skipna=True) ===")
    cds_pl_1deg = cds_pl.coarsen(latitude=2, longitude=2, boundary="trim").mean(skipna=True)
    cds_surf_1deg = cds_surf.coarsen(latitude=2, longitude=2, boundary="trim").mean(skipna=True)
    print(f"  CDS coarsened PL dims: {dict(cds_pl_1deg.sizes)}")
    print(f"  CDS coarsened lats: {cds_pl_1deg.latitude.values[0]:.2f}..{cds_pl_1deg.latitude.values[-1]:.2f}")

    print()
    print("=== Per-channel range/mean comparison (single timestep 2020-01-01 00:00) ===")
    t0 = "2020-01-01T00:00:00"
    print(f"{'ch':>8} {'level':>5} | {'CDS_1deg mean/std':>22} | {'NFS_1deg mean/std':>22} | {'rel_err':>8}")
    print("-" * 95)

    pl_map = [("t", "t", [1000, 850]), ("u", "u", [1000, 850]),
              ("v", "v", [1000, 850]), ("q", "q", [1000, 850]),
              ("z", "z", [1000, 850])]
    for cds_v, nfs_v, levels in pl_map:
        if cds_v not in cds_pl_1deg.data_vars or nfs_v not in nfs_pl.data_vars:
            print(f"  skip {cds_v} (missing)")
            continue
        for lvl in levels:
            try:
                a = cds_pl_1deg[cds_v].sel(valid_time=t0, pressure_level=lvl).values
                b = nfs_pl[nfs_v].sel(time=t0, level=lvl).values
                # align shapes (min crop)
                h, w = min(a.shape[0], b.shape[0]), min(a.shape[1], b.shape[1])
                a, b = a[:h, :w], b[:h, :w]
                am, as_ = float(np.nanmean(a)), float(np.nanstd(a))
                bm, bs_ = float(np.nanmean(b)), float(np.nanstd(b))
                diff = float(np.nanmean(np.abs(a - b)))
                rel = 100 * diff / max(abs(am), 1e-6)
                print(f"{cds_v:>8} {lvl:>5} | {am:>10.2f} / {as_:>9.2f} | {bm:>10.2f} / {bs_:>9.2f} | {rel:>7.3f}%")
            except Exception as e:
                print(f"  err {cds_v}@{lvl}: {e!r}")

    print()
    print("--- Surface ---")
    surf_map = [("t2m", "t2m"), ("u10", "u10"), ("v10", "v10"),
                ("msl", "mslp"), ("sst", "sst"), ("tcc", "tcc")]
    for cds_v, nfs_v in surf_map:
        if cds_v not in cds_surf_1deg.data_vars:
            print(f"  CDS missing {cds_v}")
            continue
        if nfs_v not in nfs_surf.data_vars:
            print(f"  NFS missing {nfs_v}")
            continue
        try:
            a = cds_surf_1deg[cds_v].sel(valid_time=t0).values
            b = nfs_surf[nfs_v].sel(time=t0).values
            h, w = min(a.shape[0], b.shape[0]), min(a.shape[1], b.shape[1])
            a, b = a[:h, :w], b[:h, :w]
            am, as_ = float(np.nanmean(a)), float(np.nanstd(a))
            bm, bs_ = float(np.nanmean(b)), float(np.nanstd(b))
            diff = float(np.nanmean(np.abs(a - b)))
            rel = 100 * diff / max(abs(am), 1e-6)
            print(f"{cds_v:>8} {'-':>5} | {am:>10.2f} / {as_:>9.2f} | {bm:>10.2f} / {bs_:>9.2f} | {rel:>7.3f}%")
        except Exception as e:
            print(f"  err {cds_v}: {e!r}")


if __name__ == "__main__":
    main()
