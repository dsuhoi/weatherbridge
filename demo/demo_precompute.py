"""Pre-compute predictions for the WTI Streamlit demo.

Runs on cloudru where:
  - memmap (245 GB) `/tmp/wb2_0p5_cache/wb2_2020.bin` is present
  - all model checkpoints live under `~/dsuhoi/weather_time_interpolation/logs/...`

Output structure (default `--out-dir`: ROOT / 'demo' / 'precomputed'):

    precomputed/
        manifest.json
        <event_id>/
            <region>_<variable>.npz       # cropped 2D arrays for all (tau, method)
            <region>_<variable>__meta.json  # per-tile metadata

NPZ keys:
    methods   : list[str]    e.g. ["ERA5","Bilinear","WeatherDCAE","FuXi", ...]
    taus      : list[int]    [1,2,3,4,5]
    lat       : (H,)         cropped lat axis (degrees, descending)
    lon       : (W,)         cropped lon axis (degrees east, may be 0..360)
    panels    : (n_methods, n_taus, H, W) float32  -- physical units
    rmse      : (n_methods-1, n_taus) float32     -- vs ERA5 row, methods[1:]
                                                       (ERA5 row dropped)

Loop order is *model outer / events inner* to minimise checkpoint loads.

Skip / resume: an existing NPZ for a (event, region, variable) is reused;
the script merges in any new model columns it computes.

Time budget: ~25 s per (event, variable, model) on A100. With 5 models, 20
events, 7 variables, 7 regions -> regions are computed in parallel from a
single forward pass (model output is global; crop happens afterwards).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("MKL_NUM_THREADS", "8")
os.environ.setdefault("SDYFF_NLAT", "360")
os.environ.setdefault("SDYFF_NLON", "720")
os.environ.setdefault("SDYFF_LAT_CROP", "0")

import numpy as np
import torch
import xarray as xr


# ---------------------------------------------------------------------------
# Paths and constants
# ---------------------------------------------------------------------------

# Project root (overridable by --root for non-cloudru hosts).
DEFAULT_ROOT = Path("/home/jovyan/dsuhoi/weather_time_interpolation")

MEMMAP_DIR_DEFAULT = "/tmp/wb2_0p5_cache"

# 24-channel name -> (block_index_in_24, label, unit, cmap_pred, cmap_bias).
# The demo exposes:
#   surface : t2m, mslp, u10, v10
#   PL      : T850, Q1000, Z500-equivalent  (we substitute Z700 because
#             the 0.5°/27-ch dataset does not include 500 hPa)
# Adjust by exporting `DEMO_VARS` env var (comma-separated).
CHANNEL_INDEX_24 = {
    # PL block (0..19): T1000,T925,T850,T700, U1000..U700, V1000..V700,
    #                   Q1000..Q700, Z1000..Z700
    "T1000": 0, "T925": 1, "T850": 2, "T700": 3,
    "U1000": 4, "U925": 5, "U850": 6, "U700": 7,
    "V1000": 8, "V925": 9, "V850": 10, "V700": 11,
    "Q1000": 12, "Q925": 13, "Q850": 14, "Q700": 15,
    "Z1000": 16, "Z925": 17, "Z850": 18, "Z700": 19,
    # surface block (20..23)
    "t2m": 20, "u10": 21, "v10": 22, "mslp": 23,
}

UNITS = {
    "t2m": "K", "u10": "m/s", "v10": "m/s", "mslp": "hPa",
    "T850": "K", "Q1000": "g/kg", "Z700": "m", "Z500": "m",
}
# Bias maps use diverging cmap, predictions sequential (mslp uses contour-friendly viridis).
CMAP_PRED = {
    "t2m": "RdBu_r", "u10": "RdBu_r", "v10": "RdBu_r", "mslp": "viridis",
    "T850": "RdBu_r", "Q1000": "BrBG", "Z700": "viridis", "Z500": "viridis",
}

DEFAULT_VARS = ["t2m", "mslp", "u10", "v10", "Q1000", "T850", "Z700"]

# The 0.5° grid is descending lat 89.75..-89.75, lon 0..359.5.
LAT_AXIS = np.linspace(89.75, -89.75, 360)
LON_AXIS = np.linspace(0.0, 359.5, 720)

MAX_TAU = 6
TAUS = list(range(1, 6))  # tau hours within the 6h window


# Per-variable physical scaling applied AFTER denormalisation.
# (memmap stores Pa, model returns Pa -> convert to hPa for plotting / RMSE)
PHYS_SCALE = {"mslp": (1.0 / 100.0)}
PHYS_OFFSET = {}


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

# Note: file system is shared between cloudru and fibo, with fibo runs synced
# under /workspace-SR006.nfs2/.../experiments/. The launcher of an inference
# task picks whichever last.ckpt is reachable. We use logs/.../last.ckpt by
# convention (see PROJECT_MAP.md and scripts/wti_typhoon_haishen_case.py).
MODELS_REL = {
    # display_name : checkpoint path relative to ROOT/logs/
    "WeatherDCAE": "exp_dcae_noskip_24ch_6yr_10ep_cloudru/last.ckpt",
    "FuXi":        "exp_fuxi_24ch_6yr_v2/last.ckpt",
    "S-DYff":      "exp_sdyff_24ch_6yr/last.ckpt",
    "ATM-VFI":     "exp_atm_vfi_24ch_v2_static_3yr_135only_fibo/last.ckpt",
    "ModAFNO":     "exp_modafno_24ch_6yr/last.ckpt",
}

# Fallback inside an experiment dir if last.ckpt is missing: latest epoch
# checkpoint by epoch number.
def resolve_ckpt(exp_root: Path) -> Optional[Path]:
    if (exp_root / "last.ckpt").exists():
        return exp_root / "last.ckpt"
    if not exp_root.is_dir():
        return None
    candidates = sorted(
        exp_root.glob("epoch=*-step=*.ckpt"),
        key=lambda p: int(p.name.split("=")[1].split("-")[0]),
    )
    if candidates:
        return candidates[-1]
    return None


# ---------------------------------------------------------------------------
# Data reader (memmap)
# ---------------------------------------------------------------------------

class Memmap2020Reader:
    """Lightweight reader for the wb2 2020 memmap (T, 27, 360, 720)."""

    def __init__(self, memmap_dir: str, stats_path: str, surface_stats_path: str):
        meta_p = Path(memmap_dir) / "wb2_2020.json"
        bin_p = Path(memmap_dir) / "wb2_2020.bin"
        if not bin_p.exists():
            raise FileNotFoundError(
                f"{bin_p} not found. Run demo_precompute on cloudru where the "
                "memmap is provisioned, or override --memmap-dir."
            )
        meta = json.loads(meta_p.read_text())
        self.T = int(meta["T"])
        self.n_ch = int(meta["n_channels"])
        self.H = int(meta["H"])
        self.W = int(meta["W"])
        self.arr = np.memmap(
            str(bin_p), dtype=np.float32, mode="r",
            shape=(self.T, self.n_ch, self.H, self.W),
        )
        self.mu, self.sigma = self._load_norm(stats_path, surface_stats_path)
        self.year_start = datetime(2020, 1, 1)

    @staticmethod
    def _load_norm(stats_path: str, surface_stats_path: str) -> Tuple[torch.Tensor, torch.Tensor]:
        pl_names = ["T", "U", "V", "Q", "Z"]
        pl_levels = [1000, 925, 850, 700]
        channel_names = [f"{v}{l}" for v in pl_names for l in pl_levels]
        with xr.open_dataset(stats_path) as ds:
            sub = ds["climate_statistics"].sel(params=channel_names)
            mu_pl = torch.from_numpy(sub.isel(stats=0).values).float()
            sig_pl = torch.from_numpy(sub.isel(stats=1).values).float()
        with open(surface_stats_path) as f:
            ss = json.load(f)
        surf_names = ["t2m", "u10", "v10", "mslp"]
        mu_sf = torch.tensor([ss[v]["mean"] for v in surf_names]).float()
        sig_sf = torch.tensor([max(ss[v]["std"], 1e-6) for v in surf_names]).float()
        mu = torch.cat([mu_pl, mu_sf]).view(-1, 1, 1)
        sig = torch.cat([sig_pl, sig_sf]).view(-1, 1, 1)
        return mu, sig

    def datetime_to_t0(self, dt: datetime) -> int:
        delta = dt - self.year_start
        return int(delta.total_seconds() // 3600)

    def read_24(self, t: int) -> torch.Tensor:
        raw = torch.from_numpy(np.ascontiguousarray(self.arr[t, :24])).float()
        raw = (raw - self.mu) / self.sigma
        return torch.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)

    def denorm_24(self, x_norm: torch.Tensor) -> torch.Tensor:
        return x_norm * self.sigma + self.mu


# ---------------------------------------------------------------------------
# Model load / forward (mirrors wti_typhoon_haishen_case.py)
# ---------------------------------------------------------------------------

def load_model(ckpt_path: Path, device: torch.device):
    from trainer_weather_hermite import WeatherHermiteLightningModule
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


def model_forward(model, x0, xT, tau_hour, static, device):
    tau_norm = torch.tensor([[tau_hour / 6.0]], dtype=torch.float32, device=device)
    cond = torch.tensor([6.0], dtype=torch.float32, device=device)
    x0_b = x0.unsqueeze(0).to(device)
    xT_b = xT.unsqueeze(0).to(device)
    static_b = static.unsqueeze(0).to(device) if static is not None else None
    with torch.no_grad():
        out = model(x0_b, xT_b, tau_norm, cond, static=static_b)
        pred = out[0] if isinstance(out, tuple) else out
    return pred.detach().cpu().squeeze(0)


def bilinear_pred(x0, xT, tau_hour):
    a = tau_hour / MAX_TAU
    return (1.0 - a) * x0 + a * xT


# ---------------------------------------------------------------------------
# Cropping helpers
# ---------------------------------------------------------------------------

def lat_to_idx(lat_deg: float) -> int:
    return int(round((89.75 - lat_deg) / 0.5))


def crop_region(field2d: np.ndarray, lat_range: Tuple[float, float], lon_range: Tuple[float, float]):
    """Crop a global 360×720 field to the requested bbox.

    `lon_range` is interpreted in -180..180 east; the global grid lives on 0..360,
    so we wrap when needed. Returns (cropped, lat_axis, lon_axis_relative).
    """
    H, W = field2d.shape
    lat_top = lat_range[1]
    lat_bot = lat_range[0]
    i_top = max(0, lat_to_idx(lat_top))
    i_bot = min(H, lat_to_idx(lat_bot) + 1)
    lat_slice = LAT_AXIS[i_top:i_bot]

    lon_w_360 = lon_range[0] % 360.0
    lon_e_360 = lon_range[1] % 360.0
    j_w = int(round(lon_w_360 / 0.5))
    j_e = int(round(lon_e_360 / 0.5))
    if j_e <= j_w:  # wrap (e.g. lon_range crosses 0)
        arr = np.concatenate(
            [field2d[i_top:i_bot, j_w:], field2d[i_top:i_bot, : j_e + 1]], axis=1
        )
        lon_arr = np.concatenate(
            [np.arange(j_w, W) * 0.5, np.arange(0, j_e + 1) * 0.5 + 360.0]
        )
    else:
        arr = field2d[i_top:i_bot, j_w: j_e + 1]
        lon_arr = np.arange(j_w, j_e + 1) * 0.5
    return arr, lat_slice, lon_arr


def apply_phys_scale(field: np.ndarray, varname: str) -> np.ndarray:
    s = PHYS_SCALE.get(varname, 1.0)
    o = PHYS_OFFSET.get(varname, 0.0)
    return field * s + o


# ---------------------------------------------------------------------------
# Precompute driver
# ---------------------------------------------------------------------------

def parse_dt(s: str) -> datetime:
    return datetime.fromisoformat(s)


def npz_path(out_dir: Path, event_id: str, region: str, variable: str) -> Path:
    safe_region = region.replace(" ", "_").replace("/", "_")
    return out_dir / event_id / f"{safe_region}__{variable}.npz"


def load_existing(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    try:
        with np.load(path, allow_pickle=True) as z:
            return {k: z[k] for k in z.files}
    except Exception:
        return None


def merge_methods(prev: Optional[dict], new_methods: List[str], new_panels: np.ndarray,
                  taus: List[int], lat: np.ndarray, lon: np.ndarray) -> dict:
    """Merge a freshly computed (methods, panels) batch into a previous NPZ payload.

    `new_panels` is (n_new_methods, n_taus, H, W). Order in `new_methods` corresponds.
    Returns dict ready for np.savez.
    """
    if prev is None:
        return {
            "methods": np.array(new_methods, dtype=object),
            "taus": np.array(taus, dtype=np.int32),
            "lat": lat.astype(np.float32),
            "lon": lon.astype(np.float32),
            "panels": new_panels.astype(np.float32),
        }
    methods = [str(x) for x in list(prev["methods"])]
    panels_list = [prev["panels"][i] for i in range(len(methods))]
    for i, m in enumerate(new_methods):
        if m in methods:
            idx = methods.index(m)
            panels_list[idx] = new_panels[i]
        else:
            methods.append(m)
            panels_list.append(new_panels[i])
    return {
        "methods": np.array(methods, dtype=object),
        "taus": np.array(taus, dtype=np.int32),
        "lat": lat.astype(np.float32),
        "lon": lon.astype(np.float32),
        "panels": np.stack(panels_list, axis=0).astype(np.float32),
    }


def compute_rmse(npz_payload: dict) -> np.ndarray:
    methods = [str(x) for x in list(npz_payload["methods"])]
    panels = npz_payload["panels"]  # (M, T, H, W)
    if "ERA5" not in methods:
        return np.full((len(methods), panels.shape[1]), np.nan, dtype=np.float32)
    gt = panels[methods.index("ERA5")]
    out = np.zeros((len(methods), panels.shape[1]), dtype=np.float32)
    for i in range(len(methods)):
        diff = panels[i] - gt
        out[i] = np.sqrt(np.mean(diff * diff, axis=(-1, -2)))
    return out


# ---------------------------------------------------------------------------
# Method registry (separates expensive [models] from cheap [reference + bilinear]).
# ---------------------------------------------------------------------------

def stage_reference(reader: Memmap2020Reader, t0: int) -> Dict[int, torch.Tensor]:
    """Return ERA5 truth (24, H, W) per tau in physical units."""
    out = {}
    for tau_h in TAUS:
        y_n = reader.read_24(t0 + tau_h)
        out[tau_h] = reader.denorm_24(y_n)
    return out


def stage_bilinear(reader: Memmap2020Reader, t0: int) -> Dict[int, torch.Tensor]:
    x0_n = reader.read_24(t0)
    xT_n = reader.read_24(t0 + MAX_TAU)
    out = {}
    for tau_h in TAUS:
        out[tau_h] = reader.denorm_24(bilinear_pred(x0_n, xT_n, tau_h))
    return out


def stage_model(model, reader: Memmap2020Reader, t0: int, static, device) -> Dict[int, torch.Tensor]:
    x0_n = reader.read_24(t0)
    xT_n = reader.read_24(t0 + MAX_TAU)
    out = {}
    for tau_h in TAUS:
        try:
            pred_n = model_forward(model, x0_n, xT_n, tau_h, static, device)
        except Exception as e:
            print(f"    [WARN] forward failed at tau={tau_h}: {type(e).__name__}: {e}",
                  flush=True)
            return None
        out[tau_h] = reader.denorm_24(pred_n)
    return out


# ---------------------------------------------------------------------------
# Per-event/variable persistence
# ---------------------------------------------------------------------------

def write_tile(out_dir: Path, event_id: str, region: str, variable: str,
               method_name: str, method_fields: Dict[int, np.ndarray],
               lat_axis: np.ndarray, lon_axis: np.ndarray) -> Path:
    p = npz_path(out_dir, event_id, region, variable)
    p.parent.mkdir(parents=True, exist_ok=True)
    panels = np.stack([method_fields[tau] for tau in TAUS], axis=0)[None]  # (1, T, H, W)
    prev = load_existing(p)
    merged = merge_methods(prev, [method_name], panels, TAUS, lat_axis, lon_axis)
    np.savez_compressed(p, **merged)
    return p


def write_rmse(out_dir: Path, event_id: str, region: str, variable: str):
    p = npz_path(out_dir, event_id, region, variable)
    data = load_existing(p)
    if data is None:
        return
    rmse = compute_rmse(data)
    data["rmse"] = rmse.astype(np.float32)
    np.savez_compressed(p, **data)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(DEFAULT_ROOT), help="Project root on cluster.")
    ap.add_argument("--memmap-dir", default=MEMMAP_DIR_DEFAULT)
    ap.add_argument("--events-json", required=True)
    ap.add_argument("--regions-json", required=True)
    ap.add_argument("--out-dir", required=True, help="Where NPZ tiles are written.")
    ap.add_argument("--variables", default=",".join(DEFAULT_VARS),
                    help="Comma-separated variable names.")
    ap.add_argument("--models", default=",".join(MODELS_REL.keys()),
                    help="Models to run. Comma-separated subset of "
                         + ",".join(MODELS_REL.keys()))
    ap.add_argument("--events", default=None,
                    help="Optional comma-separated subset of event IDs to precompute.")
    ap.add_argument("--regions", default=None,
                    help="Optional comma-separated subset of regions.")
    ap.add_argument("--skip-models", action="store_true",
                    help="Only write ERA5 + Bilinear (no model loads); useful for "
                         "infrastructure smoke tests.")
    ap.add_argument("--skip-reference", action="store_true",
                    help="Skip Pass 1 (ERA5+Bilinear). Useful when adding new "
                         "models to an existing precomputed directory.")
    ap.add_argument("--time-budget-s", type=int, default=0,
                    help="Stop launching new model runs once wall-clock exceeds this "
                         "many seconds. 0 = no limit.")
    args = ap.parse_args()

    root = Path(args.root)
    sys.path.insert(0, str(root))

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    events_cfg = json.loads(Path(args.events_json).read_text())
    regions_cfg = json.loads(Path(args.regions_json).read_text())
    events = events_cfg["events"]
    regions = regions_cfg["regions"]

    if args.events:
        wanted = set(args.events.split(","))
        events = [e for e in events if e["id"] in wanted]
    if args.regions:
        wanted = set(args.regions.split(","))
        regions = {k: v for k, v in regions.items() if k in wanted}

    variables = [v.strip() for v in args.variables.split(",") if v.strip()]
    for v in variables:
        if v not in CHANNEL_INDEX_24:
            raise ValueError(f"Unknown variable '{v}'. Known: {list(CHANNEL_INDEX_24)}")
    models_subset = [m.strip() for m in args.models.split(",") if m.strip()]

    print(f"Precompute target: {len(events)} events, {len(regions)} regions, "
          f"{len(variables)} variables, models={models_subset}", flush=True)
    print(f"  out_dir = {out_dir}", flush=True)

    # Stats + static paths
    stats_path = str(root / "data" / "json_stats_0p5.nc")
    surface_stats_path = str(root / "data" / "surface_stats_0p5.json")
    static_path = root / "data" / "static_features_0p5.pt"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  device = {device}", flush=True)

    reader = Memmap2020Reader(args.memmap_dir, stats_path, surface_stats_path)

    static = None
    if static_path.exists():
        try:
            static = torch.load(str(static_path), weights_only=False).float()
            print(f"  static features: {static.shape}", flush=True)
        except Exception as e:
            print(f"  WARN: static load failed: {e}", flush=True)

    t_global = time.time()

    # ---- pass 1: ERA5 + Bilinear (cheap, no GPU) ----
    if args.skip_reference:
        print("\n=== Pass 1: SKIPPED (--skip-reference) ===", flush=True)
        events_pass1 = []
    else:
        print("\n=== Pass 1: ERA5 + Bilinear ===", flush=True)
        events_pass1 = events
    for event in events_pass1:
        t0 = reader.datetime_to_t0(parse_dt(event["init_time"]))
        if not (0 <= t0 + MAX_TAU < reader.T):
            print(f"  SKIP {event['id']} -- t0={t0} out of memmap range", flush=True)
            continue
        ref = stage_reference(reader, t0)
        bil = stage_bilinear(reader, t0)
        for region_name, region in regions.items():
            for var in variables:
                ch = CHANNEL_INDEX_24[var]
                ref_fields, bil_fields = {}, {}
                lat_a, lon_a = None, None
                for tau in TAUS:
                    arr_ref = apply_phys_scale(ref[tau][ch].numpy(), var)
                    arr_bil = apply_phys_scale(bil[tau][ch].numpy(), var)
                    crop_ref, lat_a, lon_a = crop_region(
                        arr_ref, (region["lat_min"], region["lat_max"]),
                        (region["lon_min"], region["lon_max"]),
                    )
                    crop_bil, _, _ = crop_region(
                        arr_bil, (region["lat_min"], region["lat_max"]),
                        (region["lon_min"], region["lon_max"]),
                    )
                    ref_fields[tau] = crop_ref
                    bil_fields[tau] = crop_bil
                write_tile(out_dir, event["id"], region_name, var, "ERA5", ref_fields, lat_a, lon_a)
                write_tile(out_dir, event["id"], region_name, var, "Bilinear", bil_fields, lat_a, lon_a)
        print(f"  done {event['id']} (t0={t0})", flush=True)

    # ---- pass 2: per-model passes ----
    if not args.skip_models:
        for model_name in models_subset:
            if model_name not in MODELS_REL:
                print(f"WARN: unknown model '{model_name}' -- skipping", flush=True)
                continue
            if args.time_budget_s and (time.time() - t_global) > args.time_budget_s:
                print(f"  TIME-BUDGET reached, skipping model {model_name}", flush=True)
                continue
            ckpt_path = resolve_ckpt(root / "logs" / Path(MODELS_REL[model_name]).parent)
            if ckpt_path is None:
                print(f"  SKIP {model_name} -- no checkpoint resolved", flush=True)
                continue
            print(f"\n=== Pass 2: model {model_name} ({ckpt_path.name}) ===", flush=True)
            mdl = load_model(ckpt_path, device)
            if mdl is None:
                continue
            for event in events:
                if args.time_budget_s and (time.time() - t_global) > args.time_budget_s:
                    print(f"  TIME-BUDGET reached during {model_name}", flush=True)
                    break
                t0 = reader.datetime_to_t0(parse_dt(event["init_time"]))
                if not (0 <= t0 + MAX_TAU < reader.T):
                    continue
                t_start = time.time()
                fields = stage_model(mdl, reader, t0, static, device)
                if fields is None:
                    # forward incompatible (e.g. channel-group mismatch). Skip
                    # the entire model rather than abort the run.
                    print(f"  {model_name}: forward failed -- skipping model", flush=True)
                    break
                # crop + write per (region, variable)
                for region_name, region in regions.items():
                    for var in variables:
                        ch = CHANNEL_INDEX_24[var]
                        per_tau = {}
                        lat_a, lon_a = None, None
                        for tau in TAUS:
                            full = apply_phys_scale(fields[tau][ch].numpy(), var)
                            crop, lat_a, lon_a = crop_region(
                                full, (region["lat_min"], region["lat_max"]),
                                (region["lon_min"], region["lon_max"]),
                            )
                            per_tau[tau] = crop
                        write_tile(out_dir, event["id"], region_name, var, model_name,
                                   per_tau, lat_a, lon_a)
                print(f"  {model_name}: {event['id']} ({time.time() - t_start:.1f}s)", flush=True)
            del mdl
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # ---- pass 3: compute RMSE arrays for each tile ----
    print("\n=== Pass 3: RMSE post-pass ===", flush=True)
    n_tiles = 0
    for event in events:
        for region_name in regions:
            for var in variables:
                p = npz_path(out_dir, event["id"], region_name, var)
                if p.exists():
                    write_rmse(out_dir, event["id"], region_name, var)
                    n_tiles += 1
    print(f"  rmse written for {n_tiles} tiles", flush=True)

    # ---- manifest ----
    manifest = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "events": [e["id"] for e in events],
        "regions": list(regions.keys()),
        "variables": variables,
        "models_run": models_subset if not args.skip_models else [],
        "taus": TAUS,
        "wall_clock_s": round(time.time() - t_global, 1),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\nWrote manifest: {out_dir / 'manifest.json'}", flush=True)
    print(f"Total wall-clock: {manifest['wall_clock_s']}s", flush=True)


if __name__ == "__main__":
    main()
