"""Streamlit demo for the weather_time_interpolation project.

Compares 4-5 neural interpolation models on pre-computed ERA5 6h windows in
2020. UI lets you pick an interesting event, a region, and a meteorological
variable; the main panel shows a grid of (model x tau=1..5) maps.

Usage:
    streamlit run demo_app.py -- --precomputed-dir ./precomputed

Data is read from NPZ files written by `demo_precompute.py`.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import streamlit as st

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    HAS_CARTOPY = True
except Exception:
    HAS_CARTOPY = False


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

HERE = Path(__file__).resolve().parent

UNITS = {
    "t2m": "K", "u10": "m/s", "v10": "m/s", "mslp": "hPa",
    "T850": "K", "Q1000": "g/kg", "Z700": "m", "Z500": "m",
}
CMAP_PRED = {
    "t2m": "RdBu_r", "u10": "RdBu_r", "v10": "RdBu_r", "mslp": "viridis",
    "T850": "RdBu_r", "Q1000": "BrBG", "Z700": "viridis", "Z500": "viridis",
}
VAR_LABEL = {
    "t2m":   "2-m temperature (K)",
    "u10":   "10-m zonal wind (m/s)",
    "v10":   "10-m meridional wind (m/s)",
    "mslp":  "Mean sea-level pressure (hPa)",
    "T850":  "Temperature at 850 hPa (K)",
    "Q1000": "Specific humidity at 1000 hPa (g/kg)",
    "Z700":  "Geopotential at 700 hPa (m)",
    "Z500":  "Geopotential at 500 hPa (m)",
}

# Pin a stable column order so the layout is reproducible.
PREFERRED_ORDER = ["ERA5", "Bilinear", "WeatherDCAE", "S-DYff", "ATM-VFI", "FuXi", "ModAFNO"]

TAUS = [1, 2, 3, 4, 5]


# ---------------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--precomputed-dir", default=str(HERE / "precomputed"))
    parser.add_argument("--events-json", default=str(HERE / "demo_events.json"))
    parser.add_argument("--regions-json", default=str(HERE / "regions.json"))
    # streamlit appends its own argv; only consume what we know
    known, _ = parser.parse_known_args(sys.argv[1:])
    return known


ARGS = parse_args()


@st.cache_data(show_spinner=False)
def load_events_cfg(path: str) -> List[dict]:
    return json.loads(Path(path).read_text())["events"]


@st.cache_data(show_spinner=False)
def load_regions_cfg(path: str) -> Dict[str, dict]:
    return json.loads(Path(path).read_text())["regions"]


@st.cache_data(show_spinner=False)
def load_manifest(precomputed_dir: str) -> Optional[dict]:
    p = Path(precomputed_dir) / "manifest.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())


@st.cache_data(show_spinner=False)
def load_tile(precomputed_dir: str, event_id: str, region: str, variable: str) -> Optional[dict]:
    safe_region = region.replace(" ", "_").replace("/", "_")
    p = Path(precomputed_dir) / event_id / f"{safe_region}__{variable}.npz"
    if not p.exists():
        return None
    with np.load(p, allow_pickle=True) as z:
        out = {k: z[k] for k in z.files}
    out["methods"] = [str(x) for x in list(out["methods"])]
    return out


def order_methods(methods: List[str]) -> List[int]:
    """Return indices into `methods` in PREFERRED_ORDER, unknowns appended."""
    indices, seen = [], set()
    for m in PREFERRED_ORDER:
        if m in methods:
            i = methods.index(m)
            indices.append(i)
            seen.add(i)
    for i, m in enumerate(methods):
        if i not in seen:
            indices.append(i)
    return indices


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def figure_for_event(tile: dict, variable: str, region_extent: Tuple[float, float, float, float],
                     mode: str, title: str):
    methods: List[str] = tile["methods"]
    panels: np.ndarray = tile["panels"]  # (M, T, H, W)
    taus: List[int] = [int(x) for x in tile["taus"]]
    lat = tile["lat"]
    lon = tile["lon"]
    rmse = tile.get("rmse")  # (M, T) or None

    # Convert lon to -180..180 for cartopy (the data axis is monotone within the
    # crop; just rebase any value > 180).
    lon_plot = np.where(lon > 180.0, lon - 360.0, lon)
    # Pcolormesh edges
    if len(lon) > 1:
        dlon = lon[1] - lon[0]
    else:
        dlon = 0.5
    if len(lat) > 1:
        dlat = lat[1] - lat[0]
    else:
        dlat = -0.5
    lon_edges = np.append(lon_plot - dlon / 2, lon_plot[-1] + dlon / 2)
    lat_edges = np.append(lat - dlat / 2, lat[-1] + dlat / 2)

    # Ordering / column reduction
    col_idx = order_methods(methods)
    methods_ord = [methods[i] for i in col_idx]
    panels_ord = panels[col_idx]
    rmse_ord = rmse[col_idx] if rmse is not None else None

    if mode == "Bias maps (model - ERA5)":
        if "ERA5" not in methods_ord:
            st.error("ERA5 reference missing in precomputed tile; cannot plot bias.")
            return None
        gt_idx = methods_ord.index("ERA5")
        gt = panels_ord[gt_idx]
        keep = [i for i in range(len(methods_ord)) if i != gt_idx]
        methods_ord = [methods_ord[i] for i in keep]
        panels_ord = panels_ord[keep] - gt[None]
        if rmse_ord is not None:
            rmse_ord = rmse_ord[keep]
        cmap = "RdBu_r"
        vabs = float(np.nanpercentile(np.abs(panels_ord), 98))
        vabs = max(vabs, 1e-6)
        vmin, vmax = -vabs, vabs
    else:
        cmap = CMAP_PRED.get(variable, "viridis")
        # 1..99 percentile pooled across all panels keeps colorbar stable
        flat = panels_ord.ravel()
        vmin = float(np.nanpercentile(flat, 1))
        vmax = float(np.nanpercentile(flat, 99))
        if variable in ("u10", "v10"):  # symmetric for winds
            vabs = max(abs(vmin), abs(vmax))
            vmin, vmax = -vabs, vabs

    n_cols = len(methods_ord)
    n_rows = len(taus)

    proj = ccrs.PlateCarree() if HAS_CARTOPY else None
    fig = plt.figure(figsize=(n_cols * 2.5, n_rows * 1.95))
    gs = GridSpec(n_rows, n_cols, wspace=0.05, hspace=0.16, figure=fig,
                  left=0.05, right=0.91, top=0.91, bottom=0.05)

    im_for_cbar = None
    for ri, tau in enumerate(taus):
        for ci, method in enumerate(methods_ord):
            if HAS_CARTOPY:
                ax = fig.add_subplot(gs[ri, ci], projection=proj)
            else:
                ax = fig.add_subplot(gs[ri, ci])
            data = panels_ord[ci, ri]
            if HAS_CARTOPY:
                im = ax.pcolormesh(lon_edges, lat_edges, data,
                                   cmap=cmap, vmin=vmin, vmax=vmax,
                                   shading="auto", transform=ccrs.PlateCarree())
                ax.coastlines(linewidth=0.5, color="black")
                ax.add_feature(cfeature.BORDERS, linewidth=0.3, edgecolor="gray")
                # Use unwrapped longitudes for extent (region_extent comes from
                # regions.json in -180..180 unless explicitly wrapped)
                lon_w, lon_e = region_extent[0], region_extent[1]
                if lon_w > 180.0: lon_w -= 360.0
                if lon_e > 180.0: lon_e -= 360.0
                if lon_e <= lon_w: lon_e = lon_w + (region_extent[1] - region_extent[0])
                ax.set_extent([lon_w, lon_e, region_extent[2], region_extent[3]],
                              crs=ccrs.PlateCarree())
            else:
                extent = [lon_plot[0], lon_plot[-1], lat[-1], lat[0]]
                im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax,
                               extent=extent, origin="upper", aspect="auto")
            im_for_cbar = im

            if ri == 0:
                ax.set_title(method, fontsize=10, fontweight="bold")
            if ci == 0:
                ax.text(-0.12, 0.5, f"τ = {tau} h", transform=ax.transAxes,
                        rotation=90, va="center", ha="right", fontsize=10,
                        fontweight="bold")
            if mode == "Predictions" and rmse_ord is not None and method != "ERA5":
                # rmse_ord indexes match methods_ord *after* bias removal
                # In Predictions mode we did not drop ERA5; so the row whose
                # method == "ERA5" gets no annotation.
                try:
                    rmse_val = float(rmse_ord[ci, ri])
                    if rmse_val == rmse_val:  # not NaN
                        ax.text(0.02, 0.04, f"RMSE = {rmse_val:.3g}",
                                transform=ax.transAxes, fontsize=7,
                                bbox=dict(facecolor="white", alpha=0.75, linewidth=0,
                                          boxstyle="round,pad=0.15"))
                except Exception:
                    pass

    fig.suptitle(title, fontsize=12, fontweight="bold", y=0.985)
    if im_for_cbar is not None:
        cax = fig.add_axes([0.92, 0.06, 0.012, 0.85])
        cb = fig.colorbar(im_for_cbar, cax=cax)
        cb.set_label(UNITS.get(variable, ""), fontsize=9)
        cb.ax.tick_params(labelsize=8)
    return fig


# ---------------------------------------------------------------------------
# Streamlit app
# ---------------------------------------------------------------------------

st.set_page_config(page_title="WTI demo", layout="wide", page_icon=":cyclone:")

events = load_events_cfg(ARGS.events_json)
regions = load_regions_cfg(ARGS.regions_json)
manifest = load_manifest(ARGS.precomputed_dir)

st.sidebar.title("Weather Time Interpolation")
st.sidebar.caption(
    "ERA5 6 h window  →  interpolated τ = 1..5 h by neural baselines.  Pick "
    "a 2020 case, region, and field."
)

if manifest is None:
    st.sidebar.warning(
        f"No `manifest.json` found in `{ARGS.precomputed_dir}`. UI will still "
        "render but the panel grid will be empty until `demo_precompute.py` has "
        "been run on cloudru."
    )

# -- Event selection
event_labels = [
    f"{e['title_ru']}" if "title_ru" in e else f"{e['id']}  ({e['init_time']})"
    for e in events
]
ev_idx = st.sidebar.selectbox(
    "Event (2020):",
    options=list(range(len(events))),
    format_func=lambda i: event_labels[i],
    index=0,
)
event = events[ev_idx]

# -- Region
region_keys = list(regions.keys())
default_region_key = event.get("default_region", region_keys[0])
region_idx = region_keys.index(default_region_key) if default_region_key in region_keys else 0
region_name = st.sidebar.selectbox(
    "Region:", options=region_keys,
    format_func=lambda r: f"{r}  ({regions[r].get('label_ru', '')})",
    index=region_idx,
)
region = regions[region_name]
region_extent = (
    region["lon_min"], region["lon_max"],
    region["lat_min"], region["lat_max"],
)

# -- Variable
available_vars = manifest["variables"] if manifest else list(UNITS.keys())[:7]
default_var = event.get("default_variable", available_vars[0])
var_idx = available_vars.index(default_var) if default_var in available_vars else 0
variable = st.sidebar.selectbox(
    "Variable:",
    options=available_vars,
    format_func=lambda v: VAR_LABEL.get(v, v),
    index=var_idx,
)

# -- Mode
mode = st.sidebar.radio(
    "Plot mode:",
    options=["Predictions", "Bias maps (model - ERA5)"],
    index=0,
)

with st.sidebar.expander("Window & data info", expanded=False):
    st.write(f"**t0:** `{event['init_time']}` UTC")
    st.write(f"**τ:** {TAUS}")
    if manifest:
        st.write(f"**Precomputed at:** {manifest.get('generated_at', '?')}")
        st.write(f"**Models on disk:** {', '.join(manifest.get('models_run', []))}")
        st.write(f"**Variables:** {', '.join(manifest.get('variables', []))}")
        st.write(f"**Events:** {len(manifest.get('events', []))}")

# -- Header
title_ru = event.get("title_ru") or event.get("title_en") or event["id"]
st.title(title_ru)
st.markdown(
    f"**Region:** {region_name} ({region.get('label_ru', '')})   "
    f"&nbsp;   **Variable:** {VAR_LABEL.get(variable, variable)}   "
    f"&nbsp;   **Mode:** {mode}"
)

# -- Tile load
tile = load_tile(ARGS.precomputed_dir, event["id"], region_name, variable)
if tile is None:
    st.warning(
        "No precomputed tile for this (event, region, variable). Re-run "
        "`demo_precompute.py` with these in scope, or pick a different "
        "combination."
    )
    if manifest:
        st.code(
            "python demo_precompute.py "
            f"--events-json demo_events.json --regions-json regions.json "
            f"--out-dir ./precomputed "
            f"--events {event['id']} --regions \"{region_name}\" "
            f"--variables {variable}",
            language="bash",
        )
    st.stop()

fig = figure_for_event(tile, variable, region_extent, mode, title_ru)
if fig is not None:
    st.pyplot(fig, use_container_width=True)

# -- RMSE table
with st.expander("RMSE table (vs ERA5, per τ)"):
    methods = tile["methods"]
    rmse = tile.get("rmse")
    if rmse is None:
        st.info("RMSE not stored in this tile.")
    else:
        rows = []
        for i, m in enumerate(methods):
            if m == "ERA5":
                continue
            row = {"method": m}
            for j, tau in enumerate([int(x) for x in tile["taus"]]):
                row[f"τ={tau}h"] = round(float(rmse[i, j]), 3)
            rows.append(row)
        st.dataframe(rows, hide_index=True, use_container_width=True)

st.caption(
    "Source: ERA5 0.5°/1h via WB2 memmap on cloudru.  Models: WeatherDCAE "
    "(no-skip 24ch 6yr), FuXi-SwinV2 (24ch 6yr v2), S-DYff DYffusion (24ch "
    "6yr), ATM-VFI (24ch v2 static 3yr), ModAFNO (24ch 6yr). See "
    "`PROJECT_MAP.md` for checkpoint paths."
)
