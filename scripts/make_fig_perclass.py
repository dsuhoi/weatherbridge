#!/usr/bin/env python3
"""Per-channel-class line graphs (RMSE vs τ for 6 channel classes × 7 models)
for the 6 h and 12 h horizons, plus the OOD 2021 generalisation contrast.

Reads channel-averaged RMSE in normalised units from the canonical
``metrics/journal_unified/{6h,12h}_{2020,2021}`` evaluator outputs.

and emits three PDFs (paper/figs/fig_perclass_{6h,12h,ood2021}.pdf), each a
2x3 panel grid of {T, U, V, Q, Z, surface} versus τ for the seven (or six)
architectures. Bilinear is shown as a dashed grey baseline.

Usage:
  python scripts/make_fig_perclass.py            # all three
  python scripts/make_fig_perclass.py --mode 6h  # one only
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from paper_plot_style import MODEL_COLORS, MODEL_MARKERS
from weather_time_interp.normalization import load_channel_stats

ROOT = Path(__file__).resolve().parent.parent

# Channel groups (24-channel layout used by all eval JSONs).
CHANNELS_ORDER = [
    "T1000", "T925", "T850", "T700",
    "U1000", "U925", "U850", "U700",
    "V1000", "V925", "V850", "V700",
    "Q1000", "Q925", "Q850", "Q700",
    "Z1000", "Z925", "Z850", "Z700",
    "t2m", "u10", "v10", "mslp",
]
CHANNEL_CLASSES = {
    "T":   ["T1000", "T925", "T850", "T700"],
    "U":   ["U1000", "U925", "U850", "U700"],
    "V":   ["V1000", "V925", "V850", "V700"],
    "Q":   ["Q1000", "Q925", "Q850", "Q700"],
    "Z":   ["Z1000", "Z925", "Z850", "Z700"],
    "Surface": ["t2m", "u10", "v10", "mslp"],
}
CLASS_TITLE = {
    "T":   "Temperature (T)",
    "U":   "Zonal wind (U)",
    "V":   "Meridional wind (V)",
    "Q":   "Specific humidity (Q)",
    "Z":   "Geopotential (Z)",
    "Surface": "Surface (t2m, u10, v10, mslp)",
}

# Model display palette: (display name, JSON file, color, marker)
def models_6h() -> List[Tuple[str, str, str, str]]:
    return [
        (name, filename, MODEL_COLORS[name], MODEL_MARKERS[name])
        for name, filename in [
            ("Linear Interp.", "__LINEAR__"),
            ("SwinV2", "fuxi_24ch_6yr_ep8.json"),
            ("ModAFNO", "modafno_24ch_6yr_ep8.json"),
            ("S-DYff", "sdyff_24ch_6yr_ep8.json"),
            ("PixelAttn-VFI", "atm_vfi_6yr_ep8_matched.json"),
            ("WeatherDCAE-14M", "weatherdcae_14m_6yr_ep8_matched.json"),
            ("WeatherBridge", "weatherbridge_pp3_14m_6yr_ep8.json"),
        ]
    ]

def models_ood() -> List[Tuple[str, str, str, str]]:
    return models_6h()

def models_12h() -> List[Tuple[str, str, str, str]]:
    return [
        (name, filename, MODEL_COLORS[name], MODEL_MARKERS[name])
        for name, filename in [
            ("Linear Interp.", "__LINEAR__"),
            ("SwinV2", "fuxi_3yr_ep10.json"),
            ("ModAFNO", "modafno_3yr_ep10.json"),
            ("S-DYff", "sdyff_3yr_ep10.json"),
            ("PixelAttn-VFI", "atm_vfi_3yr_ep10_matched.json"),
            ("WeatherDCAE-14M", "weatherdcae_14m_3yr_ep10_matched.json"),
            ("WeatherBridge", "weatherbridge_14m_3yr_ep10.json"),
        ]
    ]


def load_6h_perchannel(path: Path) -> np.ndarray:
    """Returns rmse[τ, channel] with τ in {1..5}.
    Falls back to per_hour structure (e.g. OOD 2021 schema)."""
    method = "model"
    if str(path).endswith("__LINEAR__"):
        path = path.parent / "weatherbridge_pp3_14m_6yr_ep8.json"
        method = "bilinear"
    d = json.load(open(path))
    if "per_tau" in d:
        out = np.full((5, 24), np.nan)
        for ti, tau in enumerate(range(1, 6)):
            entry = d["per_tau"][str(tau)][method]
            for ci, channel in enumerate(CHANNELS_ORDER):
                out[ti, ci] = entry[f"rmse_norm_{channel}"]
        return out
    if "rmse_model_norm" in d:
        return np.asarray(d["rmse_model_norm"])
    if "per_hour" in d:
        out = np.zeros((5, 24))
        for ti, tau in enumerate(range(1, 6)):
            e = d["per_hour"][str(tau)].get("model", {})
            for ci, ch in enumerate(CHANNELS_ORDER):
                # Try rmse_norm_<CH> first, then rmse_<CH> (un-normed → normalise)
                key_norm = f"rmse_norm_{ch}"
                key_raw = f"rmse_{ch}"
                if key_norm in e:
                    out[ti, ci] = e[key_norm]
                elif key_raw in e:
                    out[ti, ci] = e[key_raw]
        return out
    raise KeyError(f"Could not find RMSE data in {path}")


def load_6h_perchannel_ci(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Load asymptotic CI low/high arrays mirroring load_6h_perchannel.

    Returns (lo, hi) each shaped (5, 24). Expects the schema produced by
    ``tools/eval/asymptotic_ci_ood2021.py``: keys
    ``rmse_<CH>_ci_low`` and ``rmse_<CH>_ci_high`` under
    ``per_hour[τ].model``. Returns NaN arrays if no CI keys are present so
    the caller can skip the shaded band silently."""
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


def load_12h_perchannel(path: Path) -> np.ndarray:
    """Returns rmse[τ, channel] with τ in {1..11} from per_tau dict.
    If path is the sentinel "__LINEAR__", load the linear baseline from
    a reference 12h JSON (per_tau[..].bilinear sub-entry)."""
    if str(path).endswith("__LINEAR__"):
        ref = path.parent / "weatherbridge_14m_3yr_ep10.json"
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


def class_avg(rmse_tau_ch: np.ndarray, group_chs: List[str]) -> np.ndarray:
    """Average over channels in a class. rmse[T, 24], returns rmse[T]."""
    idx = [CHANNELS_ORDER.index(c) for c in group_chs if c in CHANNELS_ORDER]
    return rmse_tau_ch[:, idx].mean(axis=1)


def _stds_canonical() -> dict:
    """Canonical 24-channel standard deviations used by the evaluator."""
    stats = load_channel_stats(
        ROOT / "data" / "json_stats_0p5.nc",
        ROOT / "data" / "surface_stats_0p5.json",
    )
    return {
        channel: float(std)
        for channel, std in zip(stats.channel_names, stats.std)
    }


# Same conventions as paper_units / make_fig_specific_channels_phys.
_PHYS_UNITS = {
    "T":  "K",  "U":  r"m s$^{-1}$",  "V":  r"m s$^{-1}$",
    "Q":  r"g kg$^{-1}$",            "Z":  r"m$^2$ s$^{-2}$",
    "t2m": "K",  "u10": r"m s$^{-1}$", "v10": r"m s$^{-1}$",
    "mslp": "Pa",
}
_PHYS_SCALE = {"Q": 1000.0}      # kg/kg → g/kg; everything else ×1


def _phys_class_avg(rmse_norm_tau_ch: np.ndarray, group_chs: List[str],
                    stds: dict) -> np.ndarray:
    """Average over channels in a class, in physical units (only valid for
    classes that share a unit: T/U/V/Q/Z). Returns rmse[T]."""
    idx = [CHANNELS_ORDER.index(c) for c in group_chs if c in CHANNELS_ORDER]
    fam = group_chs[0][0]
    scale = _PHYS_SCALE.get(fam, 1.0)
    sigma = np.array([stds[CHANNELS_ORDER[i]] for i in idx])  # (4,)
    phys = rmse_norm_tau_ch[:, idx] * sigma[None, :] * scale  # (T, 4)
    return phys.mean(axis=1)


def plot_perclass_phys(loader, models, taus, out_pdf: Path, title: str,
                       metrics_dir: Path, unseen_taus=None) -> None:
    """Physical-unit version of plot_perclass. Pressure-level classes use
    a single shared unit; the Surface "class" is broken into 4 dedicated
    panels (t2m, u10, v10, mslp), each in its own unit. Layout: 3x3 grid:
      T  U  V
      Q  Z  t2m
      u10 v10 mslp"""
    stds = _stds_canonical()
    panel_specs = [
        ("T",   "Temperature, PL avg",          ["T1000","T925","T850","T700"]),
        ("U",   "Zonal wind U, PL avg",         ["U1000","U925","U850","U700"]),
        ("V",   "Meridional wind V, PL avg",    ["V1000","V925","V850","V700"]),
        ("Q",   "Specific humidity Q, PL avg",  ["Q1000","Q925","Q850","Q700"]),
        ("Z",   "Geopotential Z, PL avg",       ["Z1000","Z925","Z850","Z700"]),
        ("t2m", "Temperature at 2 m",            ["t2m"]),
        ("u10", "Zonal wind at 10 m",            ["u10"]),
        ("v10", "Meridional wind at 10 m",       ["v10"]),
        ("mslp","Mean sea-level pressure",       ["mslp"]),
    ]
    fig, axes = plt.subplots(3, 3, figsize=(13.5, 10), sharex=True)
    axes = axes.flat
    legend_handles = []
    for ai, (key, chlabel, chs) in enumerate(panel_specs):
        ax = axes[ai]
        if unseen_taus:
            for ut in unseen_taus:
                ax.axvspan(ut - 0.4, ut + 0.4, color="#ffd6d6", alpha=0.5,
                           zorder=0)
        for name, fname, color, marker in models:
            p = metrics_dir / fname
            if fname != "__LINEAR__" and not p.exists():
                continue
            arr_norm = loader(p)                      # already norm
            y = _phys_class_avg(arr_norm, chs, stds)
            ls = "--" if name == "Linear Interp." else "-"
            lw = 1.8 if name in {"WeatherBridge", "WeatherDCAE-14M"} else 1.3
            (line,) = ax.plot(taus, y, marker=marker, color=color,
                              linestyle=ls, linewidth=lw, markersize=5,
                              label=name)
            if ai == 0:
                legend_handles.append(line)
        ax.set_title(f"{chlabel}  [{_PHYS_UNITS[key]}]", fontsize=11)
        ax.grid(alpha=0.3)
        if ai >= 6:
            ax.set_xlabel("Interior hour τ (h)")
        if ai % 3 == 0:
            ax.set_ylabel("RMSE")
    if unseen_taus:
        from matplotlib.patches import Patch
        legend_handles.append(Patch(facecolor="#ffd6d6", alpha=0.5,
                                    label="held-out τ"))
    fig.suptitle(title, fontsize=13, fontweight="bold", y=0.995)
    fig.legend(handles=legend_handles,
               loc="lower center", ncol=min(len(legend_handles), 5),
               bbox_to_anchor=(0.5, -0.02), fontsize=14, frameon=False)
    fig.tight_layout(rect=[0, 0.04, 1, 0.97])
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, bbox_inches="tight")
    print(f"  wrote {out_pdf}")
    plt.close(fig)


def plot_perclass(loader, models, taus, out_pdf: Path, title: str,
                  metrics_dir: Path, unseen_taus=None,
                  ci_dir: Path = None, ci_loader=None) -> None:
    """Render a 2x3 grid of per-channel-class RMSE vs τ.
    unseen_taus: list of integer τ to shade as held-out (e.g. [2,4] for 6h
    sparse-τ protocol, [4,6,8] for 12h oddskip).
    ci_dir / ci_loader: if both provided, draw shaded 95% CI bands per
    model using ``ci_loader(ci_dir/fname) -> (lo[τ,ch], hi[τ,ch])``. The
    band is the channel-averaged lo/hi over the class group."""
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.2), sharex=True)
    axes = axes.flat
    legend_handles = []
    for ai, (cls, chs) in enumerate(CHANNEL_CLASSES.items()):
        ax = axes[ai]
        # shade unseen-τ vertical bands first (so lines draw on top)
        if unseen_taus:
            for ut in unseen_taus:
                ax.axvspan(ut - 0.4, ut + 0.4, color="#ffd6d6", alpha=0.5,
                           zorder=0)
        for name, fname, color, marker in models:
            p = metrics_dir / fname
            if fname != "__LINEAR__" and not p.exists():
                continue
            arr = loader(p)
            y = class_avg(arr, chs)
            # Optional shaded 95% CI band (asymptotic, RMSE/sqrt(2N))
            if ci_dir is not None and ci_loader is not None:
                pci = ci_dir / fname
                if pci.exists():
                    lo_full, hi_full = ci_loader(pci)
                    if not np.all(np.isnan(lo_full)):
                        y_lo = class_avg(lo_full, chs)
                        y_hi = class_avg(hi_full, chs)
                        ax.fill_between(taus, y_lo, y_hi, color=color,
                                        alpha=0.18, linewidth=0, zorder=1)
            ls = "--" if name == "Linear Interp." else "-"
            lw = 1.8 if name in {"WeatherBridge", "WeatherDCAE-14M"} else 1.3
            (line,) = ax.plot(taus, y, marker=marker, color=color,
                              linestyle=ls, linewidth=lw, markersize=5,
                              label=name, zorder=3)
            if ai == 0:
                legend_handles.append(line)
        ax.set_title(CLASS_TITLE[cls], fontsize=11)
        ax.grid(alpha=0.3)
        if ai >= 3:
            ax.set_xlabel("Interior hour τ (h)")
        if ai % 3 == 0:
            ax.set_ylabel("Normalised RMSE")
    if unseen_taus:
        # legend entry for held-out shading
        from matplotlib.patches import Patch
        legend_handles.append(Patch(facecolor="#ffd6d6", alpha=0.5,
                                    label="held-out τ"))
    fig.suptitle(title, fontsize=13, fontweight="bold", y=0.995)
    fig.legend(handles=legend_handles,
               loc="lower center", ncol=min(len(legend_handles), 5),
               bbox_to_anchor=(0.5, -0.02), fontsize=14, frameon=False)
    fig.tight_layout(rect=[0, 0.04, 1, 0.97])
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, bbox_inches="tight")
    print(f"  wrote {out_pdf}")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["6h", "12h", "ood2021", "all"], default="all")
    args = p.parse_args()

    out_dir = ROOT / "paper" / "figs"

    if args.mode in ("6h", "all"):
        plot_perclass(
            loader=load_6h_perchannel,
            models=models_6h(),
            taus=[1, 2, 3, 4, 5],
            out_pdf=out_dir / "fig_perclass_6h_2020.pdf",
            title="6 h horizon — normalised RMSE by channel class, 2020 test year",
            metrics_dir=ROOT / "metrics" / "journal_unified" / "6h_2020",
            unseen_taus=[2, 4],
        )
        plot_perclass_phys(
            loader=load_6h_perchannel,
            models=models_6h(),
            taus=[1, 2, 3, 4, 5],
            out_pdf=out_dir / "fig_perclass_6h_2020_phys.pdf",
            title="6 h horizon — RMSE in physical units by channel class, "
                  "2020 test year",
            metrics_dir=ROOT / "metrics" / "journal_unified" / "6h_2020",
            unseen_taus=[2, 4],
        )

    if args.mode in ("12h", "all"):
        plot_perclass(
            loader=load_12h_perchannel,
            models=models_12h(),
            taus=list(range(1, 12)),
            out_pdf=out_dir / "fig_perclass_12h_2020.pdf",
            title="12 h horizon — normalised RMSE by channel class, 2020 oddskip",
            metrics_dir=ROOT / "metrics" / "journal_unified" / "12h_2020",
            unseen_taus=[4, 6, 8],
        )
        plot_perclass_phys(
            loader=load_12h_perchannel,
            models=models_12h(),
            taus=list(range(1, 12)),
            out_pdf=out_dir / "fig_perclass_12h_2020_phys.pdf",
            title="12 h horizon — RMSE in physical units by channel class, "
                  "2020 oddskip",
            metrics_dir=ROOT / "metrics" / "journal_unified" / "12h_2020",
            unseen_taus=[4, 6, 8],
        )

    if args.mode in ("ood2021", "all"):
        plot_perclass(
            loader=load_6h_perchannel,
            models=models_ood(),
            taus=[1, 2, 3, 4, 5],
            out_pdf=out_dir / "fig_perclass_ood2021.pdf",
            title="6 h horizon — OOD 2021 normalised RMSE by channel class",
            metrics_dir=ROOT / "metrics" / "journal_unified" / "6h_2021",
            unseen_taus=[2, 4],
        )


if __name__ == "__main__":
    main()
