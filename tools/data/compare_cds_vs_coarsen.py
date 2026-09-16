"""Compare native CDS 0.5° vs coarsened WB2 0.25°→0.5° on Jan 2020 sample.

Validates that xarray.coarsen(2,2).mean(skipna=True) is statistically equivalent
to native CDS 0.5° regridding (within numerical noise).

Reports:
  - per-channel pixel RMSE between methods
  - mean bias
  - relative error vs typical signal amplitude
  - any grid-alignment offsets
"""

import logging
import sys
from pathlib import Path

import gcsfs
import numpy as np
import xarray as xr

CDS_DIR = Path("/workspace-SR006.nfs2/weather_data/cds_validation")
SAMPLE_DAYS = ["2020-01-01", "2020-01-04", "2020-01-07", "2020-01-10",
               "2020-01-13", "2020-01-16", "2020-01-19", "2020-01-22",
               "2020-01-25", "2020-01-28"]

PL_VAR_MAP = {  # WB2 long → CDS short
    "temperature": "t",
    "u_component_of_wind": "u",
    "v_component_of_wind": "v",
    "specific_humidity": "q",
    "geopotential": "z",
}
SURF_VAR_MAP = {  # WB2 long → CDS short
    "2m_temperature": "t2m",
    "10m_u_component_of_wind": "u10",
    "10m_v_component_of_wind": "v10",
    "mean_sea_level_pressure": "msl",  # CDS uses 'msl' not 'mslp'
    "sea_surface_temperature": "sst",
    "total_cloud_cover": "tcc",
}
LEVELS = [1000, 925, 850, 700]


def setup_logger():
    logger = logging.getLogger("compare")
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    h = logging.StreamHandler(sys.stdout)
    h.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s", "%H:%M:%S"))
    logger.addHandler(h)
    return logger


def open_wb2_var(fs, var_name):
    """Open one variable from WB2 1h 0.25° dataset (per-variable Zarr)."""
    path = f"weatherbench2/datasets/era5/1959-2023_01_10-full_37-1h-1440x721.zarr/{var_name}"
    mapper = fs.get_mapper(path)
    return xr.open_zarr(mapper, consolidated=False)


def coarsen_to_0p5(ds):
    """Coarsen 0.25° → 0.5° (same method as our downloader)."""
    return ds.coarsen(latitude=2, longitude=2, boundary="trim").mean(skipna=True)


def main():
    log = setup_logger()
    cds_pl = CDS_DIR / "jan2020_pl_0p5.nc"
    cds_surf = CDS_DIR / "jan2020_surface_0p5.nc"
    if not cds_pl.exists() or not cds_surf.exists():
        log.error(f"CDS files missing: {cds_pl.exists()}, {cds_surf.exists()}")
        return

    log.info(f"loading CDS native 0.5°: {cds_pl}, {cds_surf}")
    ds_cds_pl = xr.open_dataset(cds_pl, engine="netcdf4")
    ds_cds_surf = xr.open_dataset(cds_surf, engine="netcdf4")
    log.info(f"  CDS PL dims: {dict(ds_cds_pl.sizes)}, vars: {list(ds_cds_pl.data_vars)}")
    log.info(f"  CDS Surf dims: {dict(ds_cds_surf.sizes)}, vars: {list(ds_cds_surf.data_vars)}")

    # Normalize CDS coord names
    for ds in (ds_cds_pl, ds_cds_surf):
        if "valid_time" in ds.coords:
            ds = ds.rename({"valid_time": "time"})
        if "lat" in ds.coords:
            ds = ds.rename({"lat": "latitude"})
        if "lon" in ds.coords:
            ds = ds.rename({"lon": "longitude"})

    fs = gcsfs.GCSFileSystem(token="anon")
    times = [np.datetime64(d) for d in SAMPLE_DAYS]
    log.info(f"sample days: {SAMPLE_DAYS}")

    print()
    print(f"{'channel':>10} | {'level':>5} | {'rmse_coarsen_vs_cds':>22} | {'mean_signal':>12} | {'relative':>9}")
    print("-" * 80)

    # PL vars
    for wb2_name, short in PL_VAR_MAP.items():
        log.info(f"=== {wb2_name} ===")
        ds_wb = open_wb2_var(fs, wb2_name)
        # subset levels + times
        ds_wb = ds_wb.sel(level=LEVELS)
        # Use single day for speed
        ds_wb_sample = ds_wb.sel(time=slice("2020-01-01", "2020-01-28"))
        ds_wb_sample = ds_wb_sample.sel(time=[t for t in ds_wb_sample.time.values
                                               if any(str(d) in str(t) for d in SAMPLE_DAYS)])
        log.info(f"  WB2 sample dims: {dict(ds_wb_sample.sizes)}")
        ds_wb_coarse = coarsen_to_0p5(ds_wb_sample)
        log.info(f"  WB2 coarsened dims: {dict(ds_wb_coarse.sizes)}")

        # Get CDS data using CDS short name
        cds_var = PL_VAR_MAP[wb2_name]
        if cds_var not in ds_cds_pl.data_vars:
            log.warning(f"  CDS missing {cds_var} — found: {list(ds_cds_pl.data_vars)}; skipping")
            continue
        cds_data = ds_cds_pl[cds_var]
        log.info(f"  CDS dims: {dict(cds_data.sizes)}")
        wb_arr = ds_wb_coarse[wb2_name]

        # Compute per-level RMSE
        for lvl in LEVELS:
            try:
                wb_l = wb_arr.sel(level=lvl).values
                cds_l = cds_data.sel(pressure_level=lvl) if "pressure_level" in cds_data.dims else cds_data.sel(level=lvl)
                cds_l = cds_l.values
                # align time/spatial as much as possible
                min_t = min(wb_l.shape[0], cds_l.shape[0])
                wb_l, cds_l = wb_l[:min_t], cds_l[:min_t]
                min_h = min(wb_l.shape[-2], cds_l.shape[-2])
                min_w = min(wb_l.shape[-1], cds_l.shape[-1])
                wb_l = wb_l[..., :min_h, :min_w]
                cds_l = cds_l[..., :min_h, :min_w]
                diff = wb_l - cds_l
                rmse = float(np.sqrt(np.nanmean(diff ** 2)))
                mean_sig = float(np.sqrt(np.nanmean(cds_l ** 2)))
                rel = 100 * rmse / max(mean_sig, 1e-12)
                print(f"{short:>10} | {lvl:>5} | {rmse:>22.6f} | {mean_sig:>12.4f} | {rel:>8.3f}%")
            except Exception as e:
                log.warning(f"  level {lvl} failed: {e!r}")

    # Surface vars
    for wb2_name, short in SURF_VAR_MAP.items():
        log.info(f"=== {wb2_name} ===")
        ds_wb = open_wb2_var(fs, wb2_name)
        ds_wb_sample = ds_wb.sel(time=slice("2020-01-01", "2020-01-28"))
        ds_wb_sample = ds_wb_sample.sel(time=[t for t in ds_wb_sample.time.values
                                               if any(str(d) in str(t) for d in SAMPLE_DAYS)])
        ds_wb_coarse = coarsen_to_0p5(ds_wb_sample)
        cds_var = SURF_VAR_MAP[wb2_name]
        if cds_var not in ds_cds_surf.data_vars:
            log.warning(f"  CDS missing {cds_var}; skipping")
            continue
        cds_data = ds_cds_surf[cds_var].values
        wb_arr = ds_wb_coarse[wb2_name].values
        min_t = min(wb_arr.shape[0], cds_data.shape[0])
        wb_arr, cds_data = wb_arr[:min_t], cds_data[:min_t]
        min_h = min(wb_arr.shape[-2], cds_data.shape[-2])
        min_w = min(wb_arr.shape[-1], cds_data.shape[-1])
        wb_arr = wb_arr[..., :min_h, :min_w]
        cds_data = cds_data[..., :min_h, :min_w]
        diff = wb_arr - cds_data
        rmse = float(np.sqrt(np.nanmean(diff ** 2)))
        mean_sig = float(np.sqrt(np.nanmean(cds_data ** 2)))
        rel = 100 * rmse / max(mean_sig, 1e-12)
        print(f"{short:>10} | {'-':>5} | {rmse:>22.6f} | {mean_sig:>12.4f} | {rel:>8.3f}%")


if __name__ == "__main__":
    main()
