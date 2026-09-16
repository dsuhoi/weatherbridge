#!/usr/bin/env python3
"""Specific-channel line graphs (no averaging across pressure levels).

Generates two families of figures:
  - body figures: 850hPa pressure-level panels + 4 surface panels  (9 panels)
  - appendix figures: 1000hPa, 925hPa, 700hPa rows                  (15 panels each)

Produces:
  paper/images/fig_channels_body_6h.pdf,   fig_channels_body_12h.pdf, fig_channels_body_ood2021.pdf
  paper/images/fig_channels_app_6h.pdf,    fig_channels_app_12h.pdf,  fig_channels_app_ood2021.pdf
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
from typing import List, Tuple
import numpy as np
from paper_plot_style import add_panel_labels

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from paper_plot_style import MODEL_COLORS, MODEL_MARKERS

ROOT = Path(__file__).resolve().parent.parent

# 850hPa atmospheric channels + surface = body (9 panels: 5+4)
BODY_CHANNELS = [
    ("T850",  "Temperature, 850 hPa"),
    ("U850",  "Zonal wind, 850 hPa"),
    ("V850",  "Meridional wind, 850 hPa"),
    ("Q850",  "Specific humidity, 850 hPa"),
    ("Z850",  "Geopotential, 850 hPa"),
    ("t2m",   "Temperature at 2 m"),
    ("u10",   "Zonal wind at 10 m"),
    ("v10",   "Meridional wind at 10 m"),
    ("mslp",  "MSLP"),
]
# Other pressure levels for Appendix (full 5-variable matrix used for 6h+OOD).
APP_CHANNELS = []
for lvl in [1000, 925, 700]:
    for var, varname in [("T", "Temperature"), ("U", "Zonal wind"),
                         ("V", "Meridional wind"), ("Q", "Specific humidity"),
                         ("Z", "Geopotential")]:
        APP_CHANNELS.append((f"{var}{lvl}", f"{varname}, {lvl} hPa"))

# Compact 12h appendix layout (3 levels x 3 variables = 9 panels).
# V (meridional wind) and Z (geopotential) are omitted from the compact
# appendix layout; all channel families remain in the body-level aggregate.
APP_CHANNELS_12H = []
for lvl in [1000, 925, 700]:
    for var, varname in [("T", "Temperature"), ("U", "Zonal wind"),
                         ("Q", "Specific humidity")]:
        APP_CHANNELS_12H.append((f"{var}{lvl}", f"{varname}, {lvl} hPa"))

# Index map
CHANNELS_ORDER = [
    "T1000", "T925", "T850", "T700",
    "U1000", "U925", "U850", "U700",
    "V1000", "V925", "V850", "V700",
    "Q1000", "Q925", "Q850", "Q700",
    "Z1000", "Z925", "Z850", "Z700",
    "t2m", "u10", "v10", "mslp",
]
CIDX = {c: i for i, c in enumerate(CHANNELS_ORDER)}


def models_6h() -> List[Tuple[str, str, str, str]]:
    return [
        (name, filename, MODEL_COLORS[name], MODEL_MARKERS[name])
        for name, filename in [
            ("Linear Interp.", "bilinear_24ch.json"),
            ("SwinV2", "fuxi_24ch_6yr_ep8_unified.json"),
            ("ModAFNO", "modafno_24ch_6yr_ep8_unified.json"),
            ("S-DYff", "sdyff_24ch_6yr_ep8_unified.json"),
            ("PixelAttn-VFI", "atm_vfi_v2_135only_unified.json"),
            ("WeatherDCAE-14M", "deep_noskip_14M_3yr_ep8.json"),
            ("WeatherBridge", "wb_flow_135_unified.json"),
        ]
    ]


def models_12h() -> List[Tuple[str, str, str, str]]:
    return [
        (name, filename, MODEL_COLORS[name], MODEL_MARKERS[name])
        for name, filename in [
            ("Linear Interp.", "__BILINEAR__"),
            ("SwinV2", "FuXi_3yr_12h_fibo.json"),
            ("S-DYff", "SDyff_3yr_12h_fibo.json"),
            ("PixelAttn-VFI", "ATM-VFI_bs16_compute_matched_3yr_12h.json"),
            ("WeatherDCAE-14M", "DC-AE_NoSkip_3yr_12h_fibo.json"),
            ("WeatherBridge", "WeatherBridge_PP3_3yr_12h.json"),
        ]
    ]


def models_ood() -> List[Tuple[str, str, str, str]]:
    # OOD 2021 reads from eval_0p5_2021_ood/ — keep paper-original filenames.
    return [
        (name, filename, MODEL_COLORS[name], MODEL_MARKERS[name])
        for name, filename in [
            ("Linear Interp.", "__BILINEAR_OOD__"),
            ("SwinV2", "fuxi_24ch_6yr_ep8.json"),
            ("ModAFNO", "modafno_24ch_6yr_ep8.json"),
            ("S-DYff", "sdyff_24ch_6yr_ep8.json"),
            ("PixelAttn-VFI", "atm_vfi_v2_24ch_6yr_ep8.json"),
        ]
    ]


def load_6h(path: Path) -> np.ndarray:
    if str(path).endswith("__BILINEAR_OOD__"):
        ref = path.parent / "weatherdcae_noskip_24ch_6yr_ep8.json"
        d = json.load(open(ref))
        out = np.zeros((5, 24))
        for ti, tau in enumerate(range(1, 6)):
            e = d["per_hour"][str(tau)].get("bilinear", {})
            for ci, ch in enumerate(CHANNELS_ORDER):
                out[ti, ci] = e.get(f"rmse_norm_{ch}", e.get(f"rmse_{ch}", np.nan))
        return out
    d = json.load(open(path))
    if "rmse_model_norm" in d:
        return np.asarray(d["rmse_model_norm"])  # (5,24)
    if "per_hour" in d:
        out = np.zeros((5, 24))
        for ti, tau in enumerate(range(1, 6)):
            e = d["per_hour"][str(tau)].get("model", {})
            for ci, ch in enumerate(CHANNELS_ORDER):
                out[ti, ci] = e.get(f"rmse_norm_{ch}", e.get(f"rmse_{ch}", np.nan))
        return out
    raise KeyError(f"no RMSE in {path}")


def load_6h_ci(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Load asymptotic (lo, hi) CI cubes shaped (5, 24).

    Companion to ``load_6h``. Returns NaN arrays when the JSON lacks the
    ``rmse_<CH>_ci_{low,high}`` keys (e.g. it isn't a CI sidecar)."""
    d = json.load(open(path))
    if "per_hour" not in d:
        nan = np.full((5, 24), np.nan)
        return nan, nan
    lo = np.full((5, 24), np.nan)
    hi = np.full((5, 24), np.nan)
    for ti, tau in enumerate(range(1, 6)):
        e = d["per_hour"].get(str(tau), {}).get("model", {})
        for ci, ch in enumerate(CHANNELS_ORDER):
            lo_key = f"rmse_{ch}_ci_low"
            hi_key = f"rmse_{ch}_ci_high"
            if lo_key in e and hi_key in e:
                lo[ti, ci] = e[lo_key]
                hi[ti, ci] = e[hi_key]
    return lo, hi


def load_6h_ci_bilinear_ood(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Bilinear CI from the CI directory: reads
    per_hour[τ].bilinear.rmse_<CH>_ci_{low,high}.  Path argument is the
    sentinel __BILINEAR_OOD__ inside ci_dir; we redirect to the canonical
    weatherdcae CI sidecar (all models share num_samples, so the CI factor
    is identical)."""
    p = Path(str(path).replace("__BILINEAR_OOD__",
                               "weatherdcae_noskip_24ch_6yr_ep8.json"))
    if not p.exists():
        nan = np.full((5, 24), np.nan)
        return nan, nan
    d = json.load(open(p))
    if "per_hour" not in d:
        nan = np.full((5, 24), np.nan)
        return nan, nan
    lo = np.full((5, 24), np.nan)
    hi = np.full((5, 24), np.nan)
    for ti, tau in enumerate(range(1, 6)):
        e = d["per_hour"].get(str(tau), {}).get("bilinear", {})
        for ci, ch in enumerate(CHANNELS_ORDER):
            lo_key = f"rmse_{ch}_ci_low"
            hi_key = f"rmse_{ch}_ci_high"
            if lo_key in e and hi_key in e:
                lo[ti, ci] = e[lo_key]
                hi[ti, ci] = e[hi_key]
    return lo, hi


def load_12h(path: Path) -> np.ndarray:
    if str(path).endswith("__BILINEAR__"):
        ref = path.parent / "WeatherDCAE_NoSkip_6yr_12h.json"
        if not ref.exists():
            ref = path.parent / "FuXi_3yr_12h_fibo.json"
        d = json.load(open(ref))
        pt = d["per_tau"]
        out = np.zeros((11, 24))
        for ti, tau in enumerate(range(1, 12)):
            e = pt[str(tau)]["bilinear"]
            for ci, ch in enumerate(CHANNELS_ORDER):
                out[ti, ci] = e.get(f"rmse_norm_{ch}", np.nan)
        return out
    d = json.load(open(path))
    pt = d["per_tau"]
    out = np.zeros((11, 24))
    for ti, tau in enumerate(range(1, 12)):
        e = pt[str(tau)]["model"]
        for ci, ch in enumerate(CHANNELS_ORDER):
            out[ti, ci] = e.get(f"rmse_norm_{ch}", np.nan)
    return out


def plot_grid(loader, models, taus, channels, ncols, out_pdf, title,
              metrics_dir, unseen_taus=None,
              ci_dir=None, ci_loader=None, ci_loader_bilinear=None) -> None:
    nrows = (len(channels) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.5, nrows * 2.4),
                             sharex=True)
    if nrows == 1:
        axes = np.atleast_2d(axes)
    axes_flat = axes.flat
    legend_handles = []
    for ai, (chkey, chlabel) in enumerate(channels):
        ax = axes_flat[ai]
        if unseen_taus:
            for ut in unseen_taus:
                ax.axvspan(ut - 0.4, ut + 0.4, color="#ffd6d6", alpha=0.5, zorder=0)
        cidx = CIDX[chkey]
        for name, fname, color, marker in models:
            p = metrics_dir / fname
            if fname not in ("__BILINEAR__", "__BILINEAR_OOD__") and not p.exists():
                continue
            arr = loader(p)
            y = arr[:, cidx]
            # Optional 95% asymptotic CI band per model
            if ci_dir is not None and ci_loader is not None:
                pci = ci_dir / fname
                lo_full = hi_full = None
                if fname == "__BILINEAR_OOD__" and ci_loader_bilinear is not None:
                    lo_full, hi_full = ci_loader_bilinear(pci)
                elif fname not in ("__BILINEAR__", "__BILINEAR_OOD__") and pci.exists():
                    lo_full, hi_full = ci_loader(pci)
                if lo_full is not None and not np.all(np.isnan(lo_full[:, cidx])):
                    ax.fill_between(taus, lo_full[:, cidx], hi_full[:, cidx],
                                    color=color, alpha=0.18, linewidth=0,
                                    zorder=1)
            ls = "--" if name == "Linear Interp." else "-"
            lw = 1.9 if name.startswith(("WeatherBridge", "WeatherDCAE")) else 1.3
            (line,) = ax.plot(taus, y, marker=marker, color=color,
                              linestyle=ls, linewidth=lw, markersize=4,
                              label=name, zorder=3)
            if ai == 0:
                legend_handles.append(line)
        ax.set_title(chlabel, fontsize=13)
        ax.grid(alpha=0.3)
        ax.tick_params(axis="both", labelsize=11)
        if ai // ncols == nrows - 1:
            ax.set_xlabel("τ (h)", fontsize=12)
        if ai % ncols == 0:
            ax.set_ylabel("RMSE norm", fontsize=12)
    # blank unused axes
    for ax in axes_flat[len(channels):]:
        ax.set_visible(False)
    add_panel_labels(list(axes.flat)[:len(channels)], inside=True)
    if unseen_taus:
        legend_handles.append(Patch(facecolor="#ffd6d6", alpha=0.5, label="held-out τ"))
    fig.suptitle(title, fontsize=15, fontweight="bold", y=1.0)
    # Reserve ~12% at the bottom of the layout so the large-font legend has
    # room to breathe and does not collide with the bottom-row axes; the
    # bbox_to_anchor sits a hair below the reserved strip.
    fig.tight_layout(rect=[0, 0.11, 1, 0.97])
    fig.legend(handles=legend_handles,
               loc="lower center", ncol=min(len(legend_handles), 5),
               bbox_to_anchor=(0.5, 0.0), fontsize=18, frameon=False)
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, bbox_inches="tight")
    print(f"  wrote {out_pdf}")
    plt.close(fig)


def main() -> None:
    out = ROOT / "paper" / "images"
    md_6h  = ROOT / "metrics" / "eval_6h_2020_paper_leaderboard"
    md_12h = ROOT / "metrics" / "eval_12h_2020_ep10"
    md_ood = ROOT / "metrics" / "eval_0p5_2021_ood"

    # BODY: 850hPa + surface
    plot_grid(load_6h, models_6h(), [1,2,3,4,5], BODY_CHANNELS, ncols=3,
              out_pdf=out / "fig_channels_body_6h.pdf",
              title="6 h, 2020 — RMSE per channel (850 hPa + surface)",
              metrics_dir=md_6h, unseen_taus=[2,4])
    plot_grid(load_12h, models_12h(), list(range(1,12)), BODY_CHANNELS, ncols=3,
              out_pdf=out / "fig_channels_body_12h.pdf",
              title="12 h oddskip, 2020 — RMSE per channel (850 hPa + surface)",
              metrics_dir=md_12h, unseen_taus=[4,6,8])

    # APPENDIX: 1000, 925, 700 hPa levels
    plot_grid(load_6h, models_6h(), [1,2,3,4,5], APP_CHANNELS, ncols=5,
              out_pdf=out / "fig_channels_app_6h.pdf",
              title="6 h, 2020 — RMSE per pressure-level channel",
              metrics_dir=md_6h, unseen_taus=[2,4])
    plot_grid(load_12h, models_12h(), list(range(1,12)), APP_CHANNELS_12H, ncols=3,
              out_pdf=out / "fig_channels_app_12h.pdf",
              title="12 h oddskip, 2020 — RMSE per pressure-level channel (T, U, Q)",
              metrics_dir=md_12h, unseen_taus=[4,6,8])
    plot_grid(load_6h, models_ood(), [1,2,3,4,5], BODY_CHANNELS, ncols=3,
              out_pdf=out / "fig_channels_body_ood2021.pdf",
              title="6 h, 2021 OOD — RMSE per channel (850 hPa + surface)",
              metrics_dir=md_ood, unseen_taus=[2,4])
    plot_grid(load_6h, models_ood(), [1,2,3,4,5], APP_CHANNELS, ncols=5,
              out_pdf=out / "fig_channels_app_ood2021.pdf",
              title="6 h, 2021 OOD — RMSE per pressure-level channel",
              metrics_dir=md_ood, unseen_taus=[2,4])


if __name__ == "__main__":
    main()
