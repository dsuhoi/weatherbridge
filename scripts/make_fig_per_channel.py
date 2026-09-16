#!/usr/bin/env python3
"""Per-channel RMSE line plots — compact replacement for tables 3-4.

Two panels per figure:
  left  = 20 pressure-level channels grouped by variable
          (T1000 T925 T850 T700 | U... | V... | Q... | Z...)
  right = 4 surface channels (t2m, u10, v10, mslp)

Y = normalised RMSE averaged over the seen interior hours per horizon.

Generates:
  paper/figs/fig_per_channel_6h.pdf   (6 h, 2020 test, all 7 architectures)
  paper/figs/fig_per_channel_12h.pdf  (12 h, 2020 oddskip)
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import List, Tuple
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from paper_plot_style import MODEL_COLORS, MODEL_MARKERS

ROOT = Path(__file__).resolve().parent.parent

CHANNELS_ORDER = [
    "T1000", "T925", "T850", "T700",
    "U1000", "U925", "U850", "U700",
    "V1000", "V925", "V850", "V700",
    "Q1000", "Q925", "Q850", "Q700",
    "Z1000", "Z925", "Z850", "Z700",
    "t2m", "u10", "v10", "mslp",
]
PL_CHANNELS = CHANNELS_ORDER[:20]
SURF_CHANNELS = CHANNELS_ORDER[20:]
PL_GROUPS = [("T", 0), ("U", 4), ("V", 8), ("Q", 12), ("Z", 16)]


def models_6h() -> List[Tuple[str, str, str, str]]:
    # drop it from the main per-channel breakdown.
    return [
        (name, filename, MODEL_COLORS[name], MODEL_MARKERS[name])
        for name, filename in [
            ("Linear Interp.", "bilinear_24ch.json"),
            ("SwinV2", "fuxi_24ch_6yr_ep8_unified.json"),
            ("ModAFNO", "modafno_24ch_6yr_ep8_unified.json"),
            ("S-DYff", "sdyff_24ch_6yr_ep8_unified.json"),
            ("ATM-VFI", "atm_vfi_v2_135only_unified.json"),
            ("WeatherDCAE-14M", "deep_noskip_14M_3yr_ep8.json"),
            ("WeatherBridge", "wb_flow_135_unified.json"),
        ]
    ]


def models_12h() -> List[Tuple[str, str, str, str]]:
    # ModAFNO dropped on 12h (catastrophic +127.5% inflation collapses the
    # y-scale and makes the other curves unreadable). WeatherDCAE-Skip
    # dropped as an ablation.
    return [
        (name, filename, MODEL_COLORS[name], MODEL_MARKERS[name])
        for name, filename in [
            ("Linear Interp.", "__BILINEAR__"),
            ("SwinV2", "FuXi_3yr_12h_fibo.json"),
            ("S-DYff", "SDyff_3yr_12h_fibo.json"),
            ("ATM-VFI", "ATM-VFI_bs16_compute_matched_3yr_12h.json"),
            ("WeatherDCAE-14M", "DC-AE_NoSkip_3yr_12h_fibo.json"),
            ("WeatherBridge", "WeatherBridge_PP3_3yr_12h.json"),
        ]
    ]


def load_6h(path: Path) -> np.ndarray:
    """Returns rmse_per_channel averaged over τ."""
    d = json.load(open(path))
    if "rmse_model_norm" in d:
        arr = np.asarray(d["rmse_model_norm"])  # (5, 24)
        return arr.mean(axis=0)                  # (24,)
    if "per_hour" in d:
        out = np.zeros((5, 24))
        for ti, tau in enumerate(range(1, 6)):
            e = d["per_hour"][str(tau)].get("model", {})
            for ci, ch in enumerate(CHANNELS_ORDER):
                out[ti, ci] = e.get(f"rmse_norm_{ch}", e.get(f"rmse_{ch}", np.nan))
        return out.mean(axis=0)
    raise KeyError(f"no RMSE in {path}")


def load_12h(path: Path) -> np.ndarray:
    """Returns rmse_per_channel averaged over all τ ∈ {1..11}."""
    if str(path).endswith("__BILINEAR__"):
        ref = path.parent / "WeatherDCAE_NoSkip_6yr_12h.json"
        d = json.load(open(ref))
        pt = d["per_tau"]
        out = np.zeros((11, 24))
        for ti, tau in enumerate(range(1, 12)):
            e = pt[str(tau)]["bilinear"]
            for ci, ch in enumerate(CHANNELS_ORDER):
                out[ti, ci] = e.get(f"rmse_norm_{ch}", np.nan)
        return out.mean(axis=0)
    d = json.load(open(path))
    pt = d["per_tau"]
    out = np.zeros((11, 24))
    for ti, tau in enumerate(range(1, 12)):
        e = pt[str(tau)]["model"]
        for ci, ch in enumerate(CHANNELS_ORDER):
            out[ti, ci] = e.get(f"rmse_norm_{ch}", np.nan)
    return out.mean(axis=0)


def plot_per_channel(loader, models, metrics_dir, out_pdf, title):
    """2-panel figure: PL channels (left) + surface (right)."""
    fig, (ax_pl, ax_sf) = plt.subplots(
        1, 2, figsize=(13, 4.4), gridspec_kw=dict(width_ratios=[5, 1.2])
    )

    handles = []
    for name, fname, color, marker in models:
        p = metrics_dir / fname
        if fname != "__BILINEAR__" and not p.exists():
            continue
        vec = loader(p)
        ls = "--" if name == "Linear Interp." else "-"
        lw = 2.0 if name.startswith(("WeatherBridge", "WeatherDCAE")) else 1.3
        (line,) = ax_pl.plot(range(20), vec[:20], color=color, marker=marker,
                             linestyle=ls, linewidth=lw, markersize=5, label=name)
        ax_sf.plot(range(4), vec[20:], color=color, marker=marker,
                   linestyle=ls, linewidth=lw, markersize=5)
        handles.append(line)

    # PL: x labels per channel + group separators
    ax_pl.set_xticks(range(20))
    ax_pl.set_xticklabels([c[1:] for c in PL_CHANNELS], rotation=0, fontsize=8)
    for gname, gi in PL_GROUPS:
        ax_pl.axvline(gi - 0.5, color="black", alpha=0.15, lw=0.6)
        ax_pl.text(gi + 1.5, ax_pl.get_ylim()[1], gname,
                    ha="center", va="top", fontsize=11, fontweight="bold")
    ax_pl.set_xlabel("Pressure level (hPa)", fontsize=10)
    ax_pl.set_ylabel("Normalised RMSE (avg over τ)", fontsize=10)
    ax_pl.set_title("Pressure-level channels — T, U, V, Q, Z", fontsize=11)
    ax_pl.grid(alpha=0.3)
    ax_pl.set_yscale("log")

    # Surface
    ax_sf.set_xticks(range(4))
    ax_sf.set_xticklabels(SURF_CHANNELS, fontsize=9)
    ax_sf.set_xlabel("Surface", fontsize=10)
    ax_sf.set_title("Surface", fontsize=11)
    ax_sf.grid(alpha=0.3)
    ax_sf.set_yscale("log")

    fig.suptitle(title, fontsize=12, fontweight="bold", y=0.995)
    fig.legend(handles=handles,
               loc="lower center", ncol=min(len(handles), 4),
               bbox_to_anchor=(0.5, -0.06), fontsize=12, frameon=False)
    fig.tight_layout(rect=[0, 0.07, 1, 0.95])
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, bbox_inches="tight")
    print(f"  wrote {out_pdf}")
    plt.close(fig)


def main() -> None:
    plot_per_channel(
        load_6h, models_6h(),
        ROOT / "metrics" / "eval_6h_2020_paper_leaderboard",
        ROOT / "paper" / "figs" / "fig_per_channel_6h.pdf",
        "6 h horizon, 2020 — normalised RMSE per channel",
    )
    plot_per_channel(
        load_12h, models_12h(),
        ROOT / "metrics" / "eval_12h_2020_ep10_normalized",
        ROOT / "paper" / "figs" / "fig_per_channel_12h.pdf",
        "12 h horizon, 2020 oddskip — normalised RMSE per channel",
    )


if __name__ == "__main__":
    main()
