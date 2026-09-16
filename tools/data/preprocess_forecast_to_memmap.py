"""Convert subsample forecast zarr → per-init fp32 memmap.

Layout per init: {out}/init_YYYY-MM-DDTHH.bin with metadata sidecar
    shape:  (n_leads, n_channels, 360, 720)
    dtype:  float32
    channels: 24 canonical order T*4 U*4 V*4 Q*4 Z*4 + t2m u10 v10 mslp
    leads:  6h step from T+0 (or T+6 for aurora) to T+240h
    lead_hours: stored in sidecar json
"""
import argparse, hashlib, json, warnings
warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np
import xarray as xr

from weather_time_interp.grid import (
    WB2_BLOCK_GRID_NAME,
    wb2_block_average_latitudes,
    wb2_block_average_longitudes,
)

CHANNELS_ORDER = [
    "T1000","T925","T850","T700",
    "U1000","U925","U850","U700",
    "V1000","V925","V850","V700",
    "Q1000","Q925","Q850","Q700",
    "Z1000","Z925","Z850","Z700",
    "t2m","u10","v10","mslp",
]

VAR_MAP = {
    "T": "temperature",
    "U": "u_component_of_wind",
    "V": "v_component_of_wind",
    "Q": "specific_humidity",
    "Z": "geopotential",
}
SURFACE_MAP = {
    "t2m":  "2m_temperature",
    "u10":  "10m_u_component_of_wind",
    "v10":  "10m_v_component_of_wind",
    "mslp": "mean_sea_level_pressure",
}


def extract_channel_stack(ds_init: xr.Dataset) -> tuple[np.ndarray, list[str]]:
    """Return (arr[n_lead, n_ch, H, W], available_channels)."""
    src_levels = ds_init.level.values.tolist()
    out_slices = []
    available = []
    for ch in CHANNELS_ORDER:
        if ch in SURFACE_MAP:
            key = SURFACE_MAP[ch]
            if key not in ds_init.data_vars:
                out_slices.append(None); continue
            v = ds_init[key].values  # (lead, lat, lon)
            out_slices.append(v.astype(np.float32))
            available.append(ch)
        else:
            var_short = ch[0]      # T/U/V/Q/Z
            lvl = int(ch[1:])
            key = VAR_MAP[var_short]
            if key not in ds_init.data_vars or lvl not in src_levels:
                out_slices.append(None); continue
            v = ds_init[key].sel(level=lvl).values.astype(np.float32)
            out_slices.append(v)
            available.append(ch)

    # Assemble stack; use NaN for missing channels
    n_lead = ds_init.sizes["prediction_timedelta"]
    n_ch   = len(CHANNELS_ORDER)
    H, W   = ds_init.sizes["latitude"], ds_init.sizes["longitude"]
    out = np.full((n_lead, n_ch, H, W), np.nan, dtype=np.float32)
    for i, sl in enumerate(out_slices):
        if sl is not None:
            out[:, i] = sl
    return out, available


def iter_source_datasets(path: Path):
    """Yield (init_tag, ds_single_time) for each init.

    Supports two layouts:
      A) `path` is a single monolithic zarr (has `.zmetadata` OR `.zgroup`)
         with a `time` dim → yield each time slice.
      B) `path` is a directory of per-init zarrs named `init_YYYY-MM-DDTHH.zarr`
         (v2 downloader) → yield each in sorted order.
    """
    if (path / ".zmetadata").exists() or (path / ".zgroup").exists():
        ds = xr.open_zarr(path, consolidated=(path / ".zmetadata").exists())
        for i in range(ds.sizes["time"]):
            t = ds.time.values[i]
            tag = str(np.datetime_as_string(t, unit="h")).replace(":", "")
            yield tag, ds.isel(time=i)
        return

    inits = sorted(p for p in path.iterdir() if p.name.startswith("init_") and p.is_dir())
    for p in inits:
        ds = xr.open_zarr(p, consolidated=(p / ".zmetadata").exists())
        # v2 stores time-dim=1
        t = ds.time.values[0]
        tag = str(np.datetime_as_string(t, unit="h")).replace(":", "")
        yield tag, ds.isel(time=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zarr-path", required=True,
                    help="Either a monolithic zarr or a dir of per-init zarrs")
    ap.add_argument("--out-dir",   required=True)
    args = ap.parse_args()

    src = Path(args.zarr_path)
    out_root = Path(args.out_dir); out_root.mkdir(parents=True, exist_ok=True)

    n_written = 0
    for tag, ds_i in iter_source_datasets(src):
        stem = out_root / f"init_{tag}"
        bin_p, json_p = stem.with_suffix(".bin"), stem.with_suffix(".json")
        if bin_p.exists() and json_p.exists():
            print(f"[skip] {tag}"); continue

        latitude = np.asarray(ds_i.latitude.values, dtype=np.float64)
        longitude = np.asarray(ds_i.longitude.values, dtype=np.float64)
        np.testing.assert_array_equal(latitude, wb2_block_average_latitudes())
        np.testing.assert_array_equal(longitude, wb2_block_average_longitudes())
        if ds_i.attrs.get("target_grid") != WB2_BLOCK_GRID_NAME:
            raise ValueError("forecast zarr lacks canonical-grid provenance")
        if ds_i.attrs.get("latitude_order") != "north_to_south":
            raise ValueError("forecast zarr latitude order is not canonical")
        ds_i = ds_i.load()
        arr, available = extract_channel_stack(ds_i)
        lead_hours = (ds_i.prediction_timedelta.values / np.timedelta64(1, "h")).astype(int).tolist()

        arr.tofile(bin_p)
        meta = {
            "schema_version": 2,
            "init_time": str(ds_i.time.values) if "time" in ds_i.coords else tag,
            "lead_hours": lead_hours,
            "shape": list(arr.shape),
            "channel_order": CHANNELS_ORDER,
            "channels_available": available,
            "dtype": "float32",
            "grid": {
                "name": WB2_BLOCK_GRID_NAME,
                "latitude_order": "north_to_south",
                "latitude_sha256": hashlib.sha256(
                    np.ascontiguousarray(latitude).view(np.uint8)
                ).hexdigest(),
                "longitude_sha256": hashlib.sha256(
                    np.ascontiguousarray(longitude).view(np.uint8)
                ).hexdigest(),
            },
            "source": {
                "zarr_path": str(src.resolve()),
                "source_uri": ds_i.attrs.get("source_uri"),
                "spatial_transform": ds_i.attrs.get("spatial_transform"),
            },
        }
        with open(json_p, "w") as f:
            json.dump(meta, f, indent=2)
        n_written += 1
        print(f"[write] {bin_p.name}  {arr.shape}  "
              f"{arr.nbytes/1024/1024:.1f} MB  avail={len(available)}/{len(CHANNELS_ORDER)}")
    print(f"[done] wrote {n_written} init file(s)")

if __name__ == "__main__":
    main()
