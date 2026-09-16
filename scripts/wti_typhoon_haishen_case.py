"""Case study: Typhoon Haishen, 7 September 2020, East Asia.

Generates 3 figures (mslp / t2m / u10), each is a 4×5 grid:
  rows = {ERA5, Bilinear, S-DYff 24ch, WeatherDCAE no-skip 24ch}
  cols = τ ∈ {1, 2, 3, 4, 5} hours past the t0=2020-09-07 00 UTC anchor.

Bbox: lat [20, 55], lon [100, 145].

Outputs (in ROOT/figs_vfi/case_studies/typhoon_haishen_4models/):
  fig_haishen_mslp.png  + .pdf
  fig_haishen_t2m.png   + .pdf
  fig_haishen_u10.png   + .pdf
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("MKL_NUM_THREADS", "8")
# S-DYff config so the 0.5° checkpoint can be reconstructed.
os.environ.setdefault("SDYFF_NLAT", "360")
os.environ.setdefault("SDYFF_NLON", "720")
os.environ.setdefault("SDYFF_LAT_CROP", "0")

import numpy as np
import torch
import xarray as xr

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

ROOT = Path(os.environ.get(
    "WTI_ROOT",
    "/workspace/code/wti" if Path("/workspace/code/wti").exists()
    else "/home/jovyan/dsuhoi/weather_time_interpolation",
))
sys.path.insert(0, str(ROOT))

from weather_time_interp.memmap_dataset import ERA5MemmapDataset  # noqa
from trainer_weather_hermite import WeatherHermiteLightningModule  # noqa


# -----------------------------------------------------------------------------
# Paths / constants
# -----------------------------------------------------------------------------

MEMMAP_DIR = "/tmp/wb2_0p5_cache"
STATS_PATH = str(ROOT / "data" / "json_stats_0p5.nc")
SURFACE_STATS_PATH = str(ROOT / "data" / "surface_stats_0p5.json")
STATIC_PATH = str(ROOT / "data" / "static_features_0p5.pt")

# Channel indices in the 24-ch ordering
# PL block (0..19): [T1000,T925,T850,T700, U1000..U700, V1000..V700, Q1000..Q700, Z1000..Z700]
# Surface block (20..23): t2m, u10, v10, mslp
CH_T2M = 20
CH_U10 = 21
CH_V10 = 22
CH_MSLP = 23

MAX_TAU = 6
TAUS = list(range(1, 6))  # 1..5

LAT = np.linspace(89.75, -89.75, 360)
LON = np.linspace(0.0, 359.5, 720)

# Case
CASE_DATETIME = datetime(2020, 9, 7, 0)
LAT_RANGE = (20.0, 55.0)
LON_RANGE = (100.0, 145.0)
OUT_SUBDIR = "typhoon_haishen_4models"

# Models: order of rows after ERA5 + Bilinear.
MODELS_REL = {
    # Strict ep8 checkpoints (matching paper main table provenance).
    "S-DYff":         "exp_sdyff_dyffusion_0p5_6yr/epoch=7-step=70064.ckpt",
    "WeatherDCAE":    "weatherdcae_noskip_24ch_6yr_ep8.ckpt",
}

MODEL_LABELS_EN = {
    "ERA5":         "ERA5 (truth)",
    "Bilinear":     "Bilinear",
    "S-DYff":       "S-DYff",
    "WeatherDCAE":  "WeatherDCAE",
}

FIELD_LABELS_EN = {
    "mslp": "Mean sea-level pressure (hPa)",
    "t2m":  "2-metre temperature (K)",
    "u10":  "10-metre zonal wind (m/s)",
}

# Highlight boxes drawn on every panel of the τ=3 column for the mslp
# field. Coordinates are in degrees (lon_w, lon_e, lat_s, lat_n) and
# target the Typhoon Haishen eye (centred over the southern Korean
# peninsula / Tsushima Strait around 7 Sep 2020 06–12 UTC) plus a
# secondary divergence region downstream of the cyclone.
HIGHLIGHT_BOXES_TAU3 = {
    "mslp": [
        # Eyewall / cyclone centre: ~33-37 N, 127-132 E
        dict(lon_w=127.0, lon_e=132.0, lat_s=33.0, lat_n=37.0,
             color="#ff3030", lw=2.0, label="eye"),
        # Secondary outer rain-band region to the south-east of the eye
        # over Kyushu / East China Sea: ~26-32 N, 122-130 E
        dict(lon_w=122.0, lon_e=130.0, lat_s=26.0, lat_n=32.0,
             color="#ff8a00", lw=1.6, label="outer band"),
    ],
}


# -----------------------------------------------------------------------------
# Stats / data loaders
# -----------------------------------------------------------------------------

def load_normalization() -> Tuple[torch.Tensor, torch.Tensor]:
    """Return (mu, sigma) for the first 24 channels, shape (24, 1, 1)."""
    pl_names = ["T", "U", "V", "Q", "Z"]
    pl_levels = [1000, 925, 850, 700]
    channel_names = [f"{v}{l}" for v in pl_names for l in pl_levels]
    with xr.open_dataset(STATS_PATH) as ds:
        sub = ds["climate_statistics"].sel(params=channel_names)
        mu_pl = torch.from_numpy(sub.isel(stats=0).values).float()
        sig_pl = torch.from_numpy(sub.isel(stats=1).values).float()
    with open(SURFACE_STATS_PATH) as f:
        ss = json.load(f)
    surf_names = ["t2m", "u10", "v10", "mslp"]
    mu_sf = torch.tensor([ss[v]["mean"] for v in surf_names]).float()
    sig_sf = torch.tensor([max(ss[v]["std"], 1e-6) for v in surf_names]).float()
    mu = torch.cat([mu_pl, mu_sf]).view(-1, 1, 1)
    sig = torch.cat([sig_pl, sig_sf]).view(-1, 1, 1)
    return mu, sig


class Memmap2020Reader:
    def __init__(self):
        meta_p = Path(MEMMAP_DIR) / "wb2_2020.json"
        bin_p = Path(MEMMAP_DIR) / "wb2_2020.bin"
        with open(meta_p) as f:
            meta = json.load(f)
        self.T = int(meta["T"])
        self.n_ch = int(meta["n_channels"])
        self.H = int(meta["H"])
        self.W = int(meta["W"])
        self.arr = np.memmap(str(bin_p), dtype=np.float32, mode="r",
                             shape=(self.T, self.n_ch, self.H, self.W))
        self.mu, self.sigma = load_normalization()
        self.year_start = datetime(2020, 1, 1)

    def datetime_to_t0(self, dt: datetime) -> int:
        delta = dt - self.year_start
        return int(delta.total_seconds() // 3600)

    def read_24(self, t: int) -> torch.Tensor:
        raw = torch.from_numpy(np.ascontiguousarray(self.arr[t, :24])).float()
        raw = (raw - self.mu) / self.sigma
        raw = torch.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)
        return raw


# -----------------------------------------------------------------------------
# Model load / forward
# -----------------------------------------------------------------------------

def load_model(ckpt_path: Path, device: torch.device) -> Optional[WeatherHermiteLightningModule]:
    if not ckpt_path.exists():
        print(f"  [SKIP] missing checkpoint: {ckpt_path}", flush=True)
        return None
    try:
        ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        hp = ckpt.get("hyper_parameters", {})
        ch_groups = hp.get("channel_groups", None)
        model = WeatherHermiteLightningModule.load_from_checkpoint(
            str(ckpt_path),
            map_location=device,
            channel_groups=ch_groups,
            strict=False,
        )
        model.eval().to(device)
        return model
    except Exception as e:
        print(f"  [ERROR] cannot load {ckpt_path}: {e}", flush=True)
        return None


def model_forward(
    model: WeatherHermiteLightningModule,
    x0: torch.Tensor,
    xT: torch.Tensor,
    tau_hour: int,
    static: Optional[torch.Tensor],
    device: torch.device,
) -> torch.Tensor:
    tau_norm = torch.tensor([[tau_hour / 6.0]], dtype=torch.float32, device=device)
    cond = torch.tensor([6.0], dtype=torch.float32, device=device)
    x0_b = x0.unsqueeze(0).to(device)
    xT_b = xT.unsqueeze(0).to(device)
    static_b = static.unsqueeze(0).to(device) if static is not None else None
    with torch.no_grad():
        out = model(x0_b, xT_b, tau_norm, cond, static=static_b)
        if isinstance(out, tuple):
            pred = out[0]
        else:
            pred = out
    return pred.detach().cpu()


def bilinear_pred(x0: torch.Tensor, xT: torch.Tensor, tau_hour: int) -> torch.Tensor:
    a = tau_hour / MAX_TAU
    return (1.0 - a) * x0 + a * xT


# -----------------------------------------------------------------------------
# Field extraction
# -----------------------------------------------------------------------------

def denorm_24(x_norm: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    return x_norm * sigma + mu


def extract_field(x_phys: torch.Tensor, field: str) -> np.ndarray:
    if field == "mslp":
        return x_phys[CH_MSLP].numpy()
    if field == "t2m":
        return x_phys[CH_T2M].numpy()
    if field == "u10":
        return x_phys[CH_U10].numpy()
    raise ValueError(field)


# -----------------------------------------------------------------------------
# Cropping
# -----------------------------------------------------------------------------

def lat_to_idx(lat_deg: float) -> int:
    """LAT[i] = 89.75 - 0.5*i, so i = (89.75 - lat) / 0.5."""
    return int(round((89.75 - lat_deg) / 0.5))


def crop_region(field2d: np.ndarray,
                lat_range: Tuple[float, float],
                lon_range: Tuple[float, float]):
    H, W = field2d.shape
    lat_top = lat_range[1]
    lat_bot = lat_range[0]
    i_top = max(0, lat_to_idx(lat_top))
    i_bot = min(H, lat_to_idx(lat_bot) + 1)
    lat_slice = LAT[i_top:i_bot]

    lon_w, lon_e = lon_range
    # Source longitude grid is 0..360 (0.5 step). The bbox (100..145) lies entirely in 0..180 → no wrap.
    j_w = int(round((lon_w % 360.0) / 0.5))
    j_e = int(round((lon_e % 360.0) / 0.5))
    if j_e <= j_w:
        arr = np.concatenate([field2d[i_top:i_bot, j_w:], field2d[i_top:i_bot, : j_e + 1]], axis=1)
        lon_arr = np.concatenate([np.arange(j_w, W) * 0.5, np.arange(0, j_e + 1) * 0.5 + 360.0])
    else:
        arr = field2d[i_top:i_bot, j_w : j_e + 1]
        lon_arr = np.arange(j_w, j_e + 1) * 0.5
    return arr, lat_slice, lon_arr


# -----------------------------------------------------------------------------
# Prediction caching for one window
# -----------------------------------------------------------------------------

def predict_all(reader: Memmap2020Reader,
                t0: int,
                models: Dict[str, Optional[WeatherHermiteLightningModule]],
                static: Optional[torch.Tensor],
                device: torch.device) -> Dict[int, Dict[str, torch.Tensor]]:
    """Returns {tau_hour: {method: physical-units (24,H,W)}} for tau=1..5.

    Includes 'ERA5' and 'Bilinear' methods plus each available model row.
    """
    mu, sigma = reader.mu, reader.sigma
    x0_n = reader.read_24(t0)
    xT_n = reader.read_24(t0 + MAX_TAU)
    out: Dict[int, Dict[str, torch.Tensor]] = {}
    for tau_h in TAUS:
        d: Dict[str, torch.Tensor] = {}
        y_n = reader.read_24(t0 + tau_h)
        d["ERA5"] = denorm_24(y_n, mu, sigma)
        d["Bilinear"] = denorm_24(bilinear_pred(x0_n, xT_n, tau_h), mu, sigma)
        for name, model in models.items():
            if model is None:
                continue
            pred_n = model_forward(model, x0_n, xT_n, tau_h, static, device)
            d[name] = denorm_24(pred_n.squeeze(0), mu, sigma)
        out[tau_h] = d
    return out


# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------

def plot_field(preds: Dict[int, Dict[str, torch.Tensor]],
               field: str,
               out_dir: Path) -> Dict[str, Dict[int, float]]:
    """Render one field as 4×5 grid. Returns RMSE table {method: {tau_h: rmse}}.

    RMSE is on the cropped bbox, in physical units of the field.
    """
    # Decide row order (skip absent models)
    panels = ["ERA5", "Bilinear"]
    for m in ["S-DYff", "WeatherDCAE"]:
        if m in preds[TAUS[0]]:
            panels.append(m)
    n_rows = len(panels)
    n_cols = len(TAUS)

    # First pass: collect cropped arrays + global vmin/vmax over all 20 panels.
    cropped: Dict[Tuple[str, int], np.ndarray] = {}
    lat_axis: np.ndarray = None
    lon_axis: np.ndarray = None
    for pname in panels:
        for tau_h in TAUS:
            phys = preds[tau_h][pname]
            arr2d = extract_field(phys, field)
            ca, lats, lons = crop_region(arr2d, LAT_RANGE, LON_RANGE)
            if field == "mslp":
                ca = ca / 100.0  # Pa -> hPa
            cropped[(pname, tau_h)] = ca
            lat_axis, lon_axis = lats, lons

    stack = np.stack([cropped[(p, h)] for p in panels for h in TAUS]).ravel()
    vmin = float(np.percentile(stack, 1))
    vmax = float(np.percentile(stack, 99))

    if field == "mslp":
        cmap = "viridis"
    elif field == "t2m":
        cmap = "RdBu_r"
    else:  # u10
        vabs = max(abs(vmin), abs(vmax))
        vmin, vmax = -vabs, vabs
        cmap = "RdBu_r"

    # RMSE per (method, tau) in physical units (Pa for mslp — convert back from hPa for reporting? Keep hPa for mslp).
    gt_key = "ERA5"
    rmse: Dict[str, Dict[int, float]] = {p: {} for p in panels if p != gt_key}
    for p in rmse:
        for h in TAUS:
            diff = cropped[(p, h)] - cropped[(gt_key, h)]
            rmse[p][h] = float(np.sqrt(np.mean(diff ** 2)))

    # ------------------------------------------------------------------
    # Render
    # ------------------------------------------------------------------
    proj = ccrs.PlateCarree() if HAS_CARTOPY else None
    # 2x wider panels so the figure fits a 2-column textwidth layout.
    fig = plt.figure(figsize=(n_cols * 5.4, n_rows * 2.7))
    gs = GridSpec(n_rows, n_cols, wspace=0.06, hspace=0.18, figure=fig)

    # Edges for pcolormesh
    dlat = lat_axis[1] - lat_axis[0] if len(lat_axis) > 1 else -0.5
    dlon = lon_axis[1] - lon_axis[0] if len(lon_axis) > 1 else 0.5
    lat_edges = np.append(lat_axis - dlat / 2, lat_axis[-1] + dlat / 2)
    lon_edges = np.append(lon_axis - dlon / 2, lon_axis[-1] + dlon / 2)
    # Convert lon to -180..180 contiguous for plotting if any value > 180
    lon_edges_plot = np.where(lon_edges > 180.0, lon_edges - 360.0, lon_edges)
    lon_axis_plot = np.where(lon_axis > 180.0, lon_axis - 360.0, lon_axis)

    im_for_cbar = None
    for ri, pname in enumerate(panels):
        for ci, tau_h in enumerate(TAUS):
            if HAS_CARTOPY:
                ax = fig.add_subplot(gs[ri, ci], projection=proj)
            else:
                ax = fig.add_subplot(gs[ri, ci])
            data = cropped[(pname, tau_h)]
            if HAS_CARTOPY:
                im = ax.pcolormesh(lon_edges_plot, lat_edges, data,
                                   cmap=cmap, vmin=vmin, vmax=vmax,
                                   shading="auto", transform=ccrs.PlateCarree())
                if field == "mslp":
                    try:
                        ax.contour(lon_axis_plot, lat_axis, data,
                                   levels=np.arange(int(vmin), int(vmax) + 1, 4),
                                   colors="black", linewidths=0.4,
                                   transform=ccrs.PlateCarree())
                    except Exception:
                        pass
                ax.coastlines(linewidth=0.5, color="black")
                ax.add_feature(cfeature.BORDERS, linewidth=0.3, edgecolor="gray")
                ax.set_extent([LON_RANGE[0], LON_RANGE[1], LAT_RANGE[0], LAT_RANGE[1]],
                              crs=ccrs.PlateCarree())
            else:
                extent = [lon_axis_plot[0], lon_axis_plot[-1], lat_axis[-1], lat_axis[0]]
                im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax,
                               extent=extent, origin="upper", aspect="auto")
                if field == "mslp":
                    try:
                        ax.contour(lon_axis_plot, lat_axis, data,
                                   levels=np.arange(int(vmin), int(vmax) + 1, 4),
                                   colors="black", linewidths=0.4)
                    except Exception:
                        pass
            im_for_cbar = im

            if ri == 0:
                ax.set_title(f"τ = {tau_h} h", fontsize=10)
            if ci == 0:
                ax.text(-0.12, 0.5, MODEL_LABELS_EN.get(pname, pname),
                        transform=ax.transAxes, rotation=90,
                        va="center", ha="right", fontsize=11, fontweight="bold")
            # RMSE caption for non-GT rows
            if pname in rmse:
                ax.text(0.02, 0.04, f"RMSE={rmse[pname][tau_h]:.3g}",
                        transform=ax.transAxes, fontsize=7,
                        bbox=dict(facecolor="white", alpha=0.7, linewidth=0,
                                  boxstyle="round,pad=0.15"))

            # Highlight zones on the τ=3 column for the mslp field, drawn
            # on every panel so the reader can compare ERA5 truth vs each
            # method in the same region.
            if tau_h == 3 and field in HIGHLIGHT_BOXES_TAU3:
                for box in HIGHLIGHT_BOXES_TAU3[field]:
                    rect = plt.Rectangle(
                        (box["lon_w"], box["lat_s"]),
                        box["lon_e"] - box["lon_w"],
                        box["lat_n"] - box["lat_s"],
                        fill=False, edgecolor=box["color"], linewidth=box["lw"],
                        linestyle="-", zorder=10,
                        transform=ccrs.PlateCarree() if HAS_CARTOPY else None,
                    )
                    ax.add_patch(rect)

    fig.suptitle(f"Typhoon Haishen, 7 September 2020 — {field.upper()} field",
                 fontsize=13, fontweight="bold", y=0.995)
    fig.subplots_adjust(right=0.92, top=0.93, left=0.04, bottom=0.06)
    if im_for_cbar is not None:
        cax = fig.add_axes([0.93, 0.10, 0.012, 0.78])
        cb = fig.colorbar(im_for_cbar, cax=cax)
        cb.ax.tick_params(labelsize=8)
        if field == "mslp":
            cb.set_label("hPa", fontsize=9)
        elif field == "t2m":
            cb.set_label("K", fontsize=9)
        else:
            cb.set_label("m/s", fontsize=9)

    out_dir.mkdir(parents=True, exist_ok=True)
    base = out_dir / f"fig_haishen_{field}"
    fig.savefig(base.with_suffix(".png"), dpi=140, bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {base}.png/.pdf", flush=True)
    return rmse


# -----------------------------------------------------------------------------
# Driver
# -----------------------------------------------------------------------------

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}", flush=True)
    try:
        torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "8")))
    except Exception:
        pass

    print("Loading reader (2020 memmap)...", flush=True)
    reader = Memmap2020Reader()
    t0 = reader.datetime_to_t0(CASE_DATETIME)
    print(f"  t0={t0} (2020-09-07 00 UTC), window = t0..t0+{MAX_TAU}", flush=True)

    static = None
    if Path(STATIC_PATH).exists():
        try:
            static = torch.load(STATIC_PATH, weights_only=False).float()
            print(f"  static features loaded: {static.shape}", flush=True)
        except Exception as e:
            print(f"  static load failed: {e}", flush=True)

    print("Loading models...", flush=True)
    models: Dict[str, Optional[WeatherHermiteLightningModule]] = {}
    for name, rel in MODELS_REL.items():
        print(f"  - {name} ({rel})", flush=True)
        models[name] = load_model(ROOT / "logs" / rel, device)
    avail = [n for n, m in models.items() if m is not None]
    print(f"  available models: {avail}", flush=True)

    print("Generating predictions for τ=1..5 ...", flush=True)
    preds = predict_all(reader, t0, models, static, device)
    for tau_h in TAUS:
        keys = list(preds[tau_h].keys())
        print(f"  τ={tau_h}: methods={keys}", flush=True)

    out_dir = ROOT / "figs_vfi" / "case_studies" / OUT_SUBDIR
    print(f"Output dir: {out_dir}", flush=True)

    all_rmse: Dict[str, Dict[str, Dict[int, float]]] = {}
    for field in ["mslp", "t2m", "u10"]:
        print(f"\n=== plotting field {field} ===", flush=True)
        all_rmse[field] = plot_field(preds, field, out_dir)

    # -----------------------------------------------------------------
    # Report
    # -----------------------------------------------------------------
    print("\n=================== RMSE REPORT ===================", flush=True)
    for field, mtab in all_rmse.items():
        unit = {"mslp": "hPa", "t2m": "K", "u10": "m/s"}[field]
        print(f"\n[{field}] (units = {unit})", flush=True)
        header = "  method".ljust(22) + "".join(f"  τ={h:<2}     " for h in TAUS)
        print(header, flush=True)
        for m in mtab:
            row = f"  {m:<20}" + "".join(f"  {mtab[m][h]:8.3f}" for h in TAUS)
            print(row, flush=True)
        # Dramatic τ — pick the one with worst Bilinear-DCAE gap
        if "Bilinear" in mtab and "WeatherDCAE" in mtab:
            gaps = {h: mtab["Bilinear"][h] - mtab["WeatherDCAE"][h] for h in TAUS}
            best_h = max(gaps, key=gaps.get)
            print(f"  >> largest Bilinear − WeatherDCAE gap at τ={best_h} h: "
                  f"{gaps[best_h]:+.3f} {unit}", flush=True)

    # Write JSON summary alongside the figures
    summary_path = out_dir / "rmse_summary.json"
    serial = {f: {m: {str(h): v for h, v in tab.items()} for m, tab in mtab.items()}
              for f, mtab in all_rmse.items()}
    with open(summary_path, "w") as f:
        json.dump(serial, f, indent=2)
    print(f"\nWrote summary: {summary_path}", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
