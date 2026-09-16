#!/usr/bin/env python3
"""Per-field tau-evolution RMSE maps for the journal article.

Companion to ``make_fig_bias_maps_5tau_per_field.py`` but accumulates
per-pixel *squared* differences in physical units and reports
``sqrt(mean((pred - ERA5)**2))``. The signed-bias presentation hides
how much each model errs because variance is dominant; the RMSE map
makes the headline result (ML models beat bilinear by ~2x) directly
visible.

For each of 4 weather fields {t2m, u10, mslp, Q1000} produces a
separate 3x5 RMSE-map grid:
  - rows: Linear Interp., WeatherDCAE-14M, WeatherBridge
  - cols: tau in {1, 2, 3, 4, 5} h
  - cell: time-averaged spatial RMSE (physical units), darker = worse
  - sequential colour scale (magma_r) shared per figure (one cbar per figure)

Outputs (4 PDFs):
  paper/images/fig_rmse_maps_t2m_5tau.pdf
  paper/images/fig_rmse_maps_u10_5tau.pdf
  paper/images/fig_rmse_maps_mslp_5tau.pdf
  paper/images/fig_rmse_maps_Q1000_5tau.pdf
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(
    os.environ.get("REPO_ROOT", Path(__file__).resolve().parent.parent)
).resolve()
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

try:
    import torch

    from weather_time_interp.eval_datasets import ERA5WeatherHermiteDataset
    from weather_time_interp.memmap_dataset import ERA5MemmapDataset
    from weather_time_interp.normalization import file_provenance
except ModuleNotFoundError:
    torch = None  # type: ignore[assignment]
    ERA5MemmapDataset = None  # type: ignore[assignment]
    ERA5WeatherHermiteDataset = None  # type: ignore[assignment]

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from paper_plot_style import COLUMN_WIDTH_IN, add_panel_labels

# -----------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------

FIELDS = [
    # (key,  channel_idx, display_label,                   unit,            vmax_default)
    ("t2m",  20, "2 m temperature",            "K",                3.0),
    ("u10",  21, "10 m zonal wind",            r"m s$^{-1}$",      1.5),
    ("mslp", 23, "Mean sea level pressure",    "Pa",             150.0),
    ("Q1000", 12, "Specific humidity 1000 hPa", r"g kg$^{-1}$",    0.6),
]
FIELD_KEYS = [f[0] for f in FIELDS]

TAUS = [1, 2, 3, 4, 5]

MODELS = [
    ("Linear Interp.",            None,           None),
    ("WeatherDCAE-14M",     "CKPT_WEATHERDCAE",
        "/home/jovyan/shares/SR006.nfs2/dsuhoi/capmatched_logs/"
        "exp_weatherdcae_14m_6h_6yr_refinev1_s202707_bs4/last.ckpt"),
    ("WeatherBridge", "BARE_FLOW_SPECTRAL",
        "weights/weatherbridge_14m_6h_bare.pt"),
    ("SwinV2",              "CKPT_FUXI",
        "logs/exp_fuxi_24ch_6yr_v2/epoch=7-step=17520.ckpt"),
    ("S-DYff",              "CKPT_SDYFF",
        "logs/exp_sdyff_24ch_6yr/epoch=7-step=35032.ckpt"),
    ("PixelAttn-VFI",       "CKPT_ATMVFI",
        "/home/jovyan/shares/SR006.nfs2/dsuhoi/capmatched_logs/"
        "exp_atmvfi_24ch_6h_2014_19_sparse135_lr1e4_matched/last.ckpt"),
]
MODEL_NAMES = [m[0] for m in MODELS]
PLOT_MODEL_NAMES = [
    "Linear Interp.",
    "WeatherDCAE-14M",
    "WeatherBridge",
]

CACHE_STEMS = {
    "Linear Interp.": "Bilinear",
    "PixelAttn-VFI": "ATM-VFI",
}


def _cache_stem(name: str) -> str:
    return CACHE_STEMS.get(name, name.replace(" ", "_").replace("(", "").replace(")", ""))

MAX_SAMPLES = int(os.environ.get("RMSE_MAX_SAMPLES", os.environ.get("BIAS_MAX_SAMPLES", "96")))
LOG_EVERY = int(os.environ.get("RMSE_LOG_EVERY", os.environ.get("BIAS_LOG_EVERY", "2")))


# -----------------------------------------------------------------------
# Squared-error accumulation
# -----------------------------------------------------------------------

def _forward_model(model, x0, xT, tau, cond, static, *, atm_vfi: bool = False):
    if atm_vfi:
        return model.net(x0, xT, tau)
    out = model(x0, xT, tau, cond, static=static)
    return out[0] if isinstance(out, tuple) else out


def select_sample_indices(ds, max_samples: int) -> list[int]:
    if max_samples <= 0 or max_samples >= len(ds):
        return list(range(len(ds)))
    return np.rint(np.linspace(0, len(ds) - 1, max_samples)).astype(int).tolist()


def selection_sha256(ds, ds_base, sample_indices: list[int]) -> str:
    grouped = getattr(ds, "_grouped_indices", None)
    keys = []
    for index in sample_indices:
        base_index = grouped[index][0] if grouped is not None else index
        year, t0, *_ = ds_base.index[base_index]
        keys.append(f"{year},{t0}")
    return hashlib.sha256("\n".join(keys).encode("utf-8")).hexdigest()


def compute_rmse_per_tau(model, model_type, ds, ds_base, device, sample_indices):
    """Return dict mapping (field_key, tau_h) -> (H, W) RMSE in physical units."""
    first_year = next(iter(ds_base.memmaps))
    _, _, H, W = ds_base.memmaps[first_year].shape

    n_pl = ds_base.in_channels
    surf_sigma = ds_base.surface_sigma.view(-1).numpy()
    pl_sigma = ds_base.sigma.view(-1).numpy()

    def get_sigma(ch_idx: int) -> float:
        if ch_idx < n_pl:
            return float(pl_sigma[ch_idx])
        return float(surf_sigma[ch_idx - n_pl])

    acc_sq = {(fk, tau): np.zeros((H, W), dtype=np.float64)
              for fk in FIELD_KEYS for tau in TAUS}
    n = 0
    is_atmvfi = (model_type == "atm_vfi_pixel_attn")
    if model is not None:
        model = model.to(device).eval()

    t_start = time.time()
    total = len(sample_indices)
    for position, sample_index in enumerate(sample_indices):
        sample = ds[sample_index]
        x0 = sample["x0"][:24, :, :].unsqueeze(0).to(device)
        xT = sample["xT"][:24, :, :].unsqueeze(0).to(device)
        target_all = sample["target"]
        if target_all.ndim != 4 or target_all.shape[0] < 5:
            raise RuntimeError(f"unexpected target shape {tuple(target_all.shape)}")

        cond = sample.get("cond", torch.zeros(1, 1, device=device))
        if cond.ndim == 1:
            cond = cond.unsqueeze(0)
        cond = cond.to(device)
        static = sample.get("static")
        if static is not None:
            static = static.to(device)
            if static.ndim == 3:
                static = static.unsqueeze(0)

        delta_t = 6.0
        for tau_h in TAUS:
            tgt_norm = target_all[tau_h - 1, :24].unsqueeze(0).to(device)
            # Models expect tau as a fraction of delta_t (0..1).
            tau_in = torch.tensor([[float(tau_h) / delta_t]], device=device)
            with torch.no_grad():
                if model is None:
                    w = float(tau_h) / delta_t
                    pred = (1.0 - w) * x0 + w * xT
                else:
                    pred_full = _forward_model(
                        model, x0, xT, tau_in, cond, static, atm_vfi=is_atmvfi,
                    )
                    pred = pred_full[:, :24, :, :]
            diff_norm = (pred - tgt_norm).squeeze(0).detach().float().cpu().numpy()
            for fk, ch_idx, *_ in FIELDS:
                sigma = get_sigma(ch_idx)
                phys = diff_norm[ch_idx] * sigma
                if fk == "Q1000":
                    phys = phys * 1000.0  # kg/kg -> g/kg
                acc_sq[(fk, tau_h)] += phys * phys
        n += 1
        if (position + 1) % LOG_EVERY == 0:
            dt = time.time() - t_start
            rate = (position + 1) / max(dt, 1e-6)
            print(f"    [{position+1}/{total}] elapsed={dt:.1f}s rate={rate:.2f}/s "
                  f"({rate*5:.2f} fwd/s)", flush=True)

    if n == 0:
        return {k: np.full((H, W), np.nan, dtype=np.float32) for k in acc_sq}
    return {k: np.sqrt(acc_sq[k] / n).astype(np.float32) for k in acc_sq}


# -----------------------------------------------------------------------
# Plot
# -----------------------------------------------------------------------

def plot_one_field(field_key, ch_label, unit, vmax_default,
                   rmses_by_model_tau, out_pdf, H, W, n_windows):
    n_rows = len(PLOT_MODEL_NAMES)
    n_cols = len(TAUS)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        # Drawn at the manuscript column width, so the figure is placed at
        # scale one and its labels keep the point sizes chosen below.
        figsize=(COLUMN_WIDTH_IN, 6.8 * COLUMN_WIDTH_IN / 10.0),
        sharex=True,
        sharey=True,
    )

    # vmax: shared across all (model, tau) panels of this field.
    vals = []
    for mn in PLOT_MODEL_NAMES:
        for tau in TAUS:
            arr = rmses_by_model_tau.get((mn, tau))
            if arr is not None and np.isfinite(arr).any():
                vals.append(arr)
    if vals:
        empirical = float(np.percentile(np.concatenate([v.ravel() for v in vals]), 99))
        vmax = max(empirical, 0.3 * vmax_default)
    else:
        vmax = vmax_default

    # Centres produced by the declared 2x2 block-average preprocessing.
    lats = np.linspace(89.875, -89.625, H)
    lons = np.linspace(0.125, 359.625, W)
    last_im = None

    for ri, mname in enumerate(PLOT_MODEL_NAMES):
        for ci, tau in enumerate(TAUS):
            ax = axes[ri, ci]
            arr = rmses_by_model_tau.get((mname, tau))
            if arr is None or not np.isfinite(arr).any():
                ax.set_facecolor("#eeeeee")
                ax.text(0.5, 0.5, "n/a", ha="center", va="center",
                        transform=ax.transAxes, color="#888", fontsize=5.2)
                ax.set_xticks([]); ax.set_yticks([])
                continue
            im = ax.pcolormesh(lons, lats, arr, cmap="magma_r",
                               vmin=0.0, vmax=vmax, shading="auto",
                               rasterized=True)
            last_im = im
            ax.set_xlim(0, 360); ax.set_ylim(-90, 90)
            ax.set_xticks([60, 180, 300])
            ax.set_xticklabels([r"$60^\circ$E", r"$180^\circ$", r"$300^\circ$E"])
            ax.set_yticks([-60, 0, 60])
            ax.set_yticklabels([r"$60^\circ$S", r"$0^\circ$", r"$60^\circ$N"])
            ax.tick_params(
                labelsize=5.6,
                labelleft=ci == 0,
                labelbottom=ri == n_rows - 1,
                length=2.5,
                pad=2,
            )
            ax.grid(True, color="white", alpha=0.35, lw=0.4, ls="--")
            if ri == 0:
                ax.set_title(rf"$\tau={tau}$ h", fontsize=6.6, fontweight="bold")
            if ci == 0:
                ax.text(
                    -0.42,
                    0.5,
                    mname,
                    transform=ax.transAxes,
                    fontsize=6.4,
                    fontweight="bold",
                    color="#D62728" if mname == "WeatherBridge" else "black",
                    va="center",
                    ha="right",
                )
            if mname == "WeatherBridge":
                for spine in ax.spines.values():
                    spine.set_color("#D62728")
                    spine.set_linewidth(1.4)

    add_panel_labels(axes.flat, inside=True, fontsize=6.4)
    fig.subplots_adjust(
        left=0.16,
        right=0.99,
        top=0.89,
        bottom=0.18,
        wspace=0.08,
        hspace=0.12,
    )
    if last_im is not None:
        cbar_ax = fig.add_axes([0.22, 0.075, 0.62, 0.025])
        cbar = fig.colorbar(last_im, cax=cbar_ax, orientation="horizontal")
        cbar.ax.tick_params(labelsize=5.6, length=1.6)
        cbar.set_label(rf"RMSE ({unit}); darker is worse", fontsize=6.4)

    fig.suptitle(
        rf"{ch_label}: per-grid-cell RMSE over {n_windows} uniformly sampled 2020 windows",
        fontsize=7.2, fontweight="bold", y=0.965,
    )
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_pdf}", flush=True)


def load_plot_only_caches(cache_dir: Path):
    """Load provenance-matched map caches without opening ERA5 or checkpoints."""
    rmses = {}
    shapes = set()
    sample_hashes = set()
    sample_counts = set()
    legacy_names = {"WeatherBridge": "Flow-Spectral"}

    for model_name in PLOT_MODEL_NAMES:
        cache_file = cache_dir / f"{_cache_stem(model_name)}.npz"
        if not cache_file.is_file():
            raise FileNotFoundError(f"missing plot cache {cache_file}")
        with np.load(cache_file, allow_pickle=False) as data:
            if "metadata_json" not in data.files:
                raise ValueError(f"{cache_file}: missing provenance metadata")
            metadata = json.loads(str(data["metadata_json"].item()))
            accepted_name = legacy_names.get(model_name, model_name)
            if (
                metadata.get("schema_version") != 1
                or metadata.get("year") != 2020
                or metadata.get("sample_strategy") != "uniform_over_full_year"
                or metadata.get("model_name") not in {model_name, accepted_name}
                or not metadata.get("sample_selection_sha256")
                or int(metadata.get("n_samples", 0)) <= 0
                or not metadata.get("model_source")
            ):
                raise ValueError(f"{cache_file}: invalid plot-cache provenance")
            sample_hashes.add(metadata["sample_selection_sha256"])
            sample_counts.add(int(metadata["n_samples"]))
            for field_key in FIELD_KEYS:
                for tau in TAUS:
                    key = f"{field_key}__tau{tau}"
                    if key not in data.files:
                        raise KeyError(f"{cache_file}: missing {key}")
                    values = data[key].copy()
                    if values.ndim != 2 or not np.isfinite(values).any():
                        raise ValueError(f"{cache_file}: invalid {key}")
                    shapes.add(values.shape)
                    rmses[(model_name, tau, field_key)] = values

    if len(sample_hashes) != 1 or len(sample_counts) != 1 or len(shapes) != 1:
        raise ValueError("plot caches do not share one sample index and grid")
    height, width = next(iter(shapes))
    return rmses, height, width, next(iter(sample_counts))


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

def main():
    out_dir = REPO / "paper" / "images"
    cache_dir = REPO / "metrics" / "rmse_maps_5tau_per_field"
    cache_dir.mkdir(parents=True, exist_ok=True)

    print(f"REPO: {REPO}", flush=True)
    print(f"MAX_SAMPLES per model: {MAX_SAMPLES}; tau list: {TAUS}", flush=True)

    if os.environ.get("RMSE_PLOT_ONLY", "") == "1":
        rmses, H, W, n_windows = load_plot_only_caches(cache_dir)
        for field_key, _, ch_label, unit, vmax_default in FIELDS:
            out_pdf = out_dir / f"fig_rmse_maps_{field_key}_5tau.pdf"
            sub = {
                (model_name, tau): rmses[(model_name, tau, field_key)]
                for model_name in PLOT_MODEL_NAMES
                for tau in TAUS
            }
            plot_one_field(
                field_key,
                ch_label,
                unit,
                vmax_default,
                sub,
                out_pdf,
                H,
                W,
                n_windows,
            )
        return

    print("loading dataset (year=2020, max_tau_hours=6, samples_per_date=4)...", flush=True)
    ds_base = ERA5MemmapDataset(
        memmap_dir="/tmp/wb2_0p5_cache",
        years=[2020], max_tau_hours=6, samples_per_date=4, train=False,
        static_path=str(REPO / "data" / "static_features_0p5.pt"),
        stats_path=str(REPO / "data" / "json_stats_0p5.nc"),
        surface_stats_path=str(REPO / "data" / "surface_stats_0p5.json"),
    )
    ds = ERA5WeatherHermiteDataset(ds_base, delta_t_hours=6.0)
    channel_groups = ds_base.channel_groups
    sample_indices = select_sample_indices(ds, MAX_SAMPLES)
    sample_sha = selection_sha256(ds, ds_base, sample_indices)
    print(
        f"sample selection: {len(sample_indices)}/{len(ds)} windows, "
        f"uniform over year, sha256={sample_sha}",
        flush=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}; len(ds)={len(ds)}", flush=True)

    rmses = {}
    H = W = None
    cache_only = os.environ.get("RMSE_CACHE_ONLY", "") == "1"
    allow_legacy_cache = (
        os.environ.get("RMSE_ALLOW_LEGACY_CACHE", "") == "1"
    )

    for mname, env_var, default in MODELS:
        cache_file = cache_dir / f"{_cache_stem(mname)}.npz"
        if mname == "Linear Interp.":
            ckpt_abs = None
            expected_provenance = {"kind": "analytic_linear_interpolation"}
        else:
            ckpt = os.environ.get(env_var, "") or default
            ckpt_abs = ckpt if os.path.isabs(ckpt) else str(REPO / ckpt)
            if not Path(ckpt_abs).exists():
                if cache_only and cache_file.exists():
                    cached = np.load(cache_file)
                    metadata = (
                        json.loads(str(cached["metadata_json"].item()))
                        if "metadata_json" in cached.files
                        else {}
                    )
                    expected_provenance = metadata.get("model_source")
                else:
                    print(
                        f"[{mname}] MISSING ckpt {ckpt_abs} - skipping",
                        flush=True,
                    )
                    continue
            else:
                expected_provenance = file_provenance(ckpt_abs)

        if cache_file.exists() and os.environ.get("RMSE_FORCE", "") != "1":
            data = np.load(cache_file)
            metadata = {}
            if "metadata_json" in data.files:
                metadata = json.loads(str(data["metadata_json"].item()))
            provenance_matches = (
                metadata.get("sample_selection_sha256") == sample_sha
                and metadata.get("n_samples") == len(sample_indices)
                and metadata.get("model_source") == expected_provenance
            )
            legacy_cache_allowed = allow_legacy_cache and not metadata
            if provenance_matches or legacy_cache_allowed:
                status = "CACHED" if provenance_matches else "LEGACY CACHE"
                print(f"[{mname}] {status} -> {cache_file}", flush=True)
                for fk in FIELD_KEYS:
                    for tau in TAUS:
                        key = f"{fk}__tau{tau}"
                        if key in data.files:
                            arr = data[key]
                            rmses[(mname, tau, fk)] = arr
                            H, W = arr.shape
                continue
            print(f"[{mname}] stale cache without matching provenance; recomputing")

        if cache_only:
            raise RuntimeError(
                f"{mname}: no provenance-matched cache at {cache_file}"
            )

        print(f"\n=== Processing {mname} ===", flush=True)
        if mname == "Linear Interp.":
            model, mt = None, "bilinear"
        else:
            try:
                t0 = time.time()
                if mname == "WeatherBridge":
                    from examples._bare_loader import load_bare
                    model = load_bare(ckpt_abs, str(device))
                    mt = "bare"
                else:
                    from tools.eval.batch_eval_12h_memmap import _load_model_safe
                    model, mt = _load_model_safe(
                        ckpt_abs,
                        device,
                        channel_groups,
                        static_path=str(REPO / "data" / "static_features_0p5.pt"),
                    )
                print(f"[{mname}] loaded ({mt}) {time.time() - t0:.1f}s", flush=True)
            except Exception as e:
                import traceback
                print(f"[{mname}] LOAD FAILED: {type(e).__name__}: {e}", flush=True)
                traceback.print_exc()
                continue
        try:
            result = compute_rmse_per_tau(
                model,
                mt,
                ds,
                ds_base,
                device,
                sample_indices,
            )
        except Exception as e:
            import traceback
            print(f"[{mname}] FORWARD FAILED: {type(e).__name__}: {e}", flush=True)
            traceback.print_exc()
            continue
        finally:
            if model is not None:
                del model
                torch.cuda.empty_cache()

        save_dict = {}
        for (fk, tau), arr in result.items():
            save_dict[f"{fk}__tau{tau}"] = arr
            rmses[(mname, tau, fk)] = arr
            H, W = arr.shape
        save_dict["metadata_json"] = np.asarray(json.dumps({
            "schema_version": 1,
            "year": 2020,
            "n_samples": len(sample_indices),
            "sample_strategy": "uniform_over_full_year",
            "sample_selection_sha256": sample_sha,
            "max_samples": MAX_SAMPLES,
            "model_name": mname,
            "model_source": expected_provenance,
        }))
        np.savez_compressed(cache_file, **save_dict)
        print(f"[{mname}] cached -> {cache_file}", flush=True)
        # Per-(field, tau) min/max summary at tau=3 for headline reporting
        for fk in FIELD_KEYS:
            arr = result.get((fk, 3))
            if arr is not None:
                finite = arr[np.isfinite(arr)]
                if finite.size:
                    print(f"    [{mname}/{fk}/tau=3] min={finite.min():.4g}  "
                          f"max={finite.max():.4g}  mean={finite.mean():.4g}",
                          flush=True)

    missing = [
        (model_name, tau, field_key)
        for model_name in MODEL_NAMES
        for tau in TAUS
        for field_key in FIELD_KEYS
        if (model_name, tau, field_key) not in rmses
        or not np.isfinite(rmses[(model_name, tau, field_key)]).any()
    ]
    if missing:
        preview = ", ".join(map(str, missing[:8]))
        raise RuntimeError(
            f"missing RMSE-map entries ({len(missing)}): {preview}"
        )

    if H is None:
        H, W = 360, 720

    for field_key, ch_idx, ch_label, unit, vmax_default in FIELDS:
        out_pdf = out_dir / f"fig_rmse_maps_{field_key}_5tau.pdf"
        sub = {(mn, tau): rmses.get((mn, tau, field_key))
               for mn in MODEL_NAMES for tau in TAUS}
        plot_one_field(field_key, ch_label, unit, vmax_default,
                       sub, out_pdf, H, W, len(sample_indices))


if __name__ == "__main__":
    main()
