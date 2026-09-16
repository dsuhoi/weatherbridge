"""Direct 0.5° comparison: WB2 0.25° → coarsen 2× → 0.5° vs CDS native 0.5°.

Same 10 sample days (Jan 2020). WB2 stored as per-variable Zarr Arrays (not
Groups), so we read each via zarr.open() + reconstruct dims from
_ARRAY_DIMENSIONS attr.
"""

import sys
from pathlib import Path

import gcsfs
import numpy as np
import xarray as xr
import zarr

WB2_ROOT = "weatherbench2/datasets/era5/1959-2023_01_10-full_37-1h-1440x721.zarr"
CDS_PL = Path("/workspace-SR006.nfs2/weather_data/cds_validation/jan2020_pl_0p5.nc")
CDS_SURF = Path("/workspace-SR006.nfs2/weather_data/cds_validation/jan2020_surface_0p5.nc")

SAMPLE_DAYS = [
    np.datetime64("2020-01-01T00:00:00"),
    np.datetime64("2020-01-04T00:00:00"),
    np.datetime64("2020-01-07T00:00:00"),
    np.datetime64("2020-01-10T00:00:00"),
    np.datetime64("2020-01-13T00:00:00"),
    np.datetime64("2020-01-16T00:00:00"),
    np.datetime64("2020-01-19T00:00:00"),
    np.datetime64("2020-01-22T00:00:00"),
    np.datetime64("2020-01-25T00:00:00"),
    np.datetime64("2020-01-28T00:00:00"),
]

# WB2 long name -> CDS short name -> our short name
PL_MAP = [
    ("temperature", "t"),
    ("u_component_of_wind", "u"),
    ("v_component_of_wind", "v"),
    ("specific_humidity", "q"),
    ("geopotential", "z"),
]
SURF_MAP = [
    ("2m_temperature", "t2m"),
    ("10m_u_component_of_wind", "u10"),
    ("10m_v_component_of_wind", "v10"),
    ("mean_sea_level_pressure", "msl"),
    ("sea_surface_temperature", "sst"),
    ("total_cloud_cover", "tcc"),
]
LEVELS = [1000, 925, 850, 700]


def open_wb2_array(fs, var_name):
    """Open one WB2 variable as zarr.Array + return dims list."""
    mapper = fs.get_mapper(f"{WB2_ROOT}/{var_name}")
    arr = zarr.open(mapper, mode="r")
    dims = arr.attrs.get("_ARRAY_DIMENSIONS")
    return arr, dims


def load_wb2_dataarray(fs, var_name, time_indices, level_indices=None):
    """Read a slice from WB2 per-variable Zarr Array, return as xarray DataArray."""
    arr, dims = open_wb2_array(fs, var_name)
    print(f"  {var_name}: shape={arr.shape}, dims={dims}")
    # Build slicer: indices for time, levels (if applicable), all spatial
    slicers = []
    for d in dims:
        if d == "time":
            slicers.append(time_indices)
        elif d == "level":
            slicers.append(level_indices if level_indices is not None else slice(None))
        else:
            slicers.append(slice(None))
    # Note: zarr fancy indexing supports lists only on one axis at a time.
    # We do time as list, level slice, lat/lon full.
    data = arr.oindex[tuple(slicers)]
    return data


def main():
    fs = gcsfs.GCSFileSystem(token="anon")

    # WB2 layout (hardcoded — only "time" coord at root, no level/lat/lon):
    # time:  hours since 1959-01-01 00:00:00, shape=561264 (up to 2023-01-10)
    # level: 37 standard ECMWF pressure levels (descending)
    # lat:   721 points, 0.25° from 90.0 to -90.0
    # lon:   1440 points, 0.25° from 0 to 359.75
    print("=== WB2 0.25° layout (hardcoded by spec) ===")
    BASE = np.datetime64("1959-01-01T00:00:00")
    ECMWF_LEVELS = np.array([1, 2, 3, 5, 7, 10, 20, 30, 50, 70, 100, 125, 150, 175, 200,
                              225, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750,
                              775, 800, 825, 850, 875, 900, 925, 950, 975, 1000])
    print(f"  base epoch: {BASE}")
    print(f"  levels: {ECMWF_LEVELS}")
    print(f"  lat: 721 pts, 90.0..-90.0 step 0.25")

    # Verify time coord (root has it)
    t_arr, _ = open_wb2_array(fs, "time")
    times_raw = t_arr[:]
    print(f"  time raw shape: {times_raw.shape}, first 3: {times_raw[:3]}")
    times = BASE + times_raw.astype("timedelta64[h]")
    print(f"  time first: {times[0]}, last: {times[-1]}")

    # Find time indices for our sample days
    time_idx = []
    for d in SAMPLE_DAYS:
        match = np.where(times == d)[0]
        if len(match) == 0:
            print(f"  WARN: no match for {d}")
            continue
        time_idx.append(int(match[0]))
    print(f"  matched time indices: {time_idx[:5]}... (total {len(time_idx)})")

    # Level indices
    level_idx = [int(np.where(ECMWF_LEVELS == L)[0][0]) for L in LEVELS]
    print(f"  level indices for {LEVELS}: {level_idx}")

    # ---- CDS reference ----
    print()
    print("=== CDS native 0.5° ===")
    cds_pl = xr.open_dataset(str(CDS_PL), engine="netcdf4")
    cds_surf = xr.open_dataset(str(CDS_SURF), engine="netcdf4")
    # CDS times -> matching indices
    cds_times = cds_pl.valid_time.values  # numpy datetime64
    cds_idx = [int(np.where(cds_times == d)[0][0]) for d in SAMPLE_DAYS]
    print(f"  CDS time indices: {cds_idx[:5]}... (total {len(cds_idx)})")

    # ---- PL comparison ----
    print()
    print("=== 0.5° comparison: WB2 coarsen vs CDS native ===")
    print(f"{'channel':>8} {'level':>5} | {'WB2coarse mean':>14} | {'CDS_native mean':>15} | {'RMSE':>10} | {'rel_err(%)':>10}")
    print("-" * 80)

    for wb_name, short in PL_MAP:
        # Read WB2 0.25° subset: (n_time, n_levels, 721, 1440)
        try:
            data_wb_025 = load_wb2_dataarray(fs, wb_name, time_idx, level_idx)
        except Exception as e:
            print(f"  load fail {wb_name}: {e!r}")
            continue
        # Coarsen 2x via numpy: (T, L, 721, 1440) -> trim -> (T, L, 720, 1440) -> mean -> (T, L, 360, 720)
        T, L, H, W = data_wb_025.shape
        data_t = data_wb_025[:, :, :720, :]  # drop last lat row to make even
        coarse = np.nanmean(
            data_t.reshape(T, L, 360, 2, 720, 2),
            axis=(3, 5),
        )  # → (T, L, 360, 720)

        cds_arr = cds_pl[short].values  # (240, 4, 361, 720)
        cds_sub = cds_arr[cds_idx]  # (10, 4, 361, 720)
        cds_sub = cds_sub[:, :, :360, :]  # match shape with coarsen

        diff = coarse - cds_sub
        for li, L_val in enumerate(LEVELS):
            wb_l = coarse[:, li]
            cds_l = cds_sub[:, li]
            wb_mean = float(np.nanmean(wb_l))
            cds_mean = float(np.nanmean(cds_l))
            rmse = float(np.sqrt(np.nanmean((wb_l - cds_l) ** 2)))
            std_ref = float(np.nanstd(cds_l))
            rel = 100 * rmse / max(std_ref, 1e-9)
            print(f"{short:>8} {L_val:>5} | {wb_mean:>14.4f} | {cds_mean:>15.4f} | {rmse:>10.5f} | {rel:>10.3f}")

    # ---- Surface comparison ----
    print()
    print("--- Surface ---")
    for wb_name, short in SURF_MAP:
        try:
            data_wb_025 = load_wb2_dataarray(fs, wb_name, time_idx, level_indices=None)
        except Exception as e:
            print(f"  load fail {wb_name}: {e!r}")
            continue
        T, H, W = data_wb_025.shape
        data_t = data_wb_025[:, :720, :]  # trim 721->720
        coarse = np.nanmean(
            data_t.reshape(T, 360, 2, 720, 2),
            axis=(2, 4),
        )  # → (T, 360, 720)
        if short not in cds_surf.data_vars:
            print(f"  CDS missing {short}")
            continue
        cds_arr = cds_surf[short].values  # (240, 361, 720)
        cds_sub = cds_arr[cds_idx][:, :360, :]
        wb_mean = float(np.nanmean(coarse))
        cds_mean = float(np.nanmean(cds_sub))
        rmse = float(np.sqrt(np.nanmean((coarse - cds_sub) ** 2)))
        std_ref = float(np.nanstd(cds_sub))
        rel = 100 * rmse / max(std_ref, 1e-9)
        print(f"{short:>8} {'-':>5} | {wb_mean:>14.4f} | {cds_mean:>15.4f} | {rmse:>10.5f} | {rel:>10.3f}")


if __name__ == "__main__":
    main()
