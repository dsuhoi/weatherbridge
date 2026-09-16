#!/usr/bin/env python3
"""Specific-channel line graphs in DENORMALISED physical units.

Counterpart to `make_fig_specific_channels.py` that displays RMSE in the
field's native units (K, m s⁻¹, Pa, m² s⁻²) rather than dimensionless
normalised RMSE. The unit per channel is encoded in the panel title and
used to choose the y-axis label.

Reads `rmse_<CH>` entries directly from the metric JSONs (already in
physical units, see `tools/eval/batch_eval_memmap.py`). When that key is
missing, falls back to `rmse_norm_<CH>` × stored channel std.

Outputs:
  paper/images/fig_fields_6h_thermodynamic_phys.pdf
  paper/images/fig_fields_6h_wind_phys.pdf
  paper/images/fig_fields_6h_mass_surface_phys.pdf
  paper/images/fig_channels_body_6h_phys.pdf
  paper/images/fig_channels_app_6h_phys.pdf
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
from typing import List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
try:
    from scripts.paper_plot_style import (
        COLUMN_WIDTH_IN,
        MODEL_COLORS,
        MODEL_LINESTYLES,
        MODEL_MARKERS,
        add_panel_labels,
        use_paper_rc,
    )
except ModuleNotFoundError:  # Direct execution: python scripts/<file>.py
    from paper_plot_style import (
        COLUMN_WIDTH_IN,
        MODEL_COLORS,
        MODEL_LINESTYLES,
        MODEL_MARKERS,
        add_panel_labels,
        use_paper_rc,
    )
from weather_time_interp.normalization import load_channel_stats

ROOT = Path(__file__).resolve().parent.parent
METRICS_ROOT = Path(
    os.environ.get(
        "WTI_JOURNAL_METRICS_ROOT",
        ROOT / "metrics" / "journal_unified",
    )
)

# Channel unit map (paper-friendly LaTeX). Used in y-axis labels.
UNITS = {
    "T": r"K",
    "U": r"m s$^{-1}$",
    "V": r"m s$^{-1}$",
    "Q": r"g kg$^{-1}$",
    "Z": r"m$^2$ s$^{-2}$",
    "t2m": r"K",
    "u10": r"m s$^{-1}$",
    "v10": r"m s$^{-1}$",
    "mslp": r"Pa",
}

# Native ERA5 storage units → display units. Q is stored as kg/kg in the σ
# map, so the displayed g/kg requires ×1000. Other channels are 1:1.
UNIT_SCALE = {
    "T": 1.0, "U": 1.0, "V": 1.0, "Z": 1.0,
    "Q": 1000.0,        # kg/kg → g/kg
    "t2m": 1.0, "u10": 1.0, "v10": 1.0, "mslp": 1.0,
}

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
APP_CHANNELS = []
for lvl in [1000, 925, 700]:
    for var, varname in [("T", "Temperature"), ("U", "Zonal wind"),
                         ("V", "Meridional wind"), ("Q", "Specific humidity"),
                         ("Z", "Geopotential")]:
        APP_CHANNELS.append((f"{var}{lvl}", f"{varname}, {lvl} hPa"))

CHANNELS_ORDER = [
    "T1000", "T925", "T850", "T700",
    "U1000", "U925", "U850", "U700",
    "V1000", "V925", "V850", "V700",
    "Q1000", "Q925", "Q850", "Q700",
    "Z1000", "Z925", "Z850", "Z700",
    "t2m", "u10", "v10", "mslp",
]
CIDX = {c: i for i, c in enumerate(CHANNELS_ORDER)}
_CHANNEL_LABELS = dict(BODY_CHANNELS + APP_CHANNELS)
ALL_CHANNELS = [(channel, _CHANNEL_LABELS[channel]) for channel in CHANNELS_ORDER]
THERMODYNAMIC_CHANNELS = [
    (channel, _CHANNEL_LABELS[channel])
    for channel in (
        "T1000", "T925", "T850", "T700",
        "Q1000", "Q925", "Q850", "Q700",
    )
]
WIND_CHANNELS = [
    (channel, _CHANNEL_LABELS[channel])
    for channel in (
        "U1000", "U925", "U850", "U700",
        "V1000", "V925", "V850", "V700",
        "u10", "v10",
    )
]
MASS_SURFACE_CHANNELS = [
    (channel, _CHANNEL_LABELS[channel])
    for channel in ("Z1000", "Z925", "Z850", "Z700", "t2m", "mslp")
]


def _channel_family(ch: str) -> str:
    """Map a channel key to its UNITS family."""
    if ch in ("t2m", "u10", "v10", "mslp"):
        return ch
    return ch[0]  # T1000 -> T, etc.


def models_6h(main_panel: bool = False) -> List[Tuple[str, str, str, str]]:
    models = [
        (name, filename, MODEL_COLORS[name], MODEL_MARKERS[name])
        for name, filename in [
            ("Linear Interp.", "__LINEAR__"),
            ("SwinV2", "fuxi_24ch_6yr_ep8.json"),
            ("ModAFNO", "modafno_24ch_6yr_ep8.json"),
            ("S-DYff", "sdyff_24ch_6yr_ep8.json"),
            ("S-DYff-ENS", "sdyff_ens21.json"),
            ("PixelAttn-VFI", "atm_vfi_6yr_ep8_matched.json"),
            ("WeatherDCAE-14M", "weatherdcae_14m_6yr_ep8_matched.json"),
            ("WeatherBridge", "weatherbridge_pp3_14m_6yr_ep8.json"),
        ]
    ]
    if main_panel:
        excluded = {"SwinV2", "ModAFNO"}
        return [model for model in models if model[0] not in excluded]
    return models


def _canonical_stds() -> dict:
    """Single canonical standard-deviation map used by all models."""
    stats = load_channel_stats(
        ROOT / "data" / "json_stats_0p5.nc",
        ROOT / "data" / "surface_stats_0p5.json",
    )
    return {
        channel: float(std)
        for channel, std in zip(stats.channel_names, stats.std)
    }


_STDS = None


def _get_canonical_stds() -> dict:
    """Load channel scales only for legacy metrics without physical RMSE."""
    global _STDS
    if _STDS is None:
        _STDS = _canonical_stds()
    return _STDS


def load_6h_phys(path: Path) -> np.ndarray:
    """Returns rmse[τ, channel] in PHYSICAL units (τ ∈ 1..5).

    Uses `rmse_norm_<CH> × σ_canonical[<CH>]`. The previously stored
    `rmse_model_phys` field was denormalised inconsistently across channels
    (mslp/Z off by 30-70×, Q by ~5000×) — see audit notes in OVERNIGHT_SUMMARY.
    """
    method = "model"
    if str(path).endswith("__LINEAR__"):
        path = path.parent / "weatherbridge_pp3_14m_6yr_ep8.json"
        method = "bilinear"
    d = json.load(open(path))
    out = np.full((5, 24), np.nan)
    if "per_tau" in d:
        for ti, tau in enumerate(range(1, 6)):
            entry = d["per_tau"][str(tau)][method]
            for ci, channel in enumerate(CHANNELS_ORDER):
                out[ti, ci] = _phys_from_entry(entry, channel)
        return out
    # Schema A: top-level rmse_model_norm (5, 24) + channels list
    if "rmse_model_norm" in d:
        stds = _get_canonical_stds()
        arr = np.asarray(d["rmse_model_norm"])
        chs = d.get("channels", CHANNELS_ORDER)
        for ci, ch in enumerate(CHANNELS_ORDER):
            if ch in chs:
                scale = UNIT_SCALE[_channel_family(ch)]
                out[:, ci] = arr[:, chs.index(ch)] * stds[ch] * scale
        return out
    # Schema B: per_hour[tau].model.rmse_norm_<CH>
    if "per_hour" in d:
        stds = _get_canonical_stds()
        for ti, tau in enumerate(range(1, 6)):
            e = d["per_hour"][str(tau)].get("model", {})
            for ci, ch in enumerate(CHANNELS_ORDER):
                v = e.get(f"rmse_norm_{ch}")
                if v is not None:
                    scale = UNIT_SCALE[_channel_family(ch)]
                    out[ti, ci] = v * stds[ch] * scale
        return out
    raise KeyError(f"No normalised RMSE in {path}")


def plot_grid_phys(
    loader,
    models,
    taus,
    channels,
    ncols,
    out_pdf,
    title,
    metrics_dir,
    unseen_taus=None,
    clip_model=None,
    height_ratio=0.78,
    compact_channel_titles=None,
) -> None:
    missing = [
        metrics_dir / filename
        for _, filename, _, _ in models
        if not filename.startswith("__") and not (metrics_dir / filename).is_file()
    ]
    if missing:
        joined = "\n".join(str(path) for path in missing)
        raise FileNotFoundError(
            f"refusing an incomplete comparison; missing metrics:\n{joined}"
        )
    indices = {
        json.loads((metrics_dir / filename).read_text())
        .get("evaluation_protocol", {})
        .get("index_sha256")
        for _, filename, _, _ in models
        if not filename.startswith("__")
    }
    if None in indices or len(indices) != 1:
        raise ValueError(
            "refusing an unmatched comparison; models use different "
            "or missing window indices"
        )
    nrows = (len(channels) + ncols - 1) // ncols
    # Drawn at the manuscript column width so the figure is placed at scale one
    # and every point size survives into the PDF above the 5 pt legibility floor.
    use_paper_rc(7.0)
    panel_w = COLUMN_WIDTH_IN / ncols
    fig_height = nrows * panel_w * height_ratio
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(COLUMN_WIDTH_IN, fig_height),
                             sharex=True)
    # Channel keys keep long field names from crossing adjacent panel titles.
    if compact_channel_titles is None:
        compact_channel_titles = True
    title_size = 6.0 if compact_channel_titles else 6.6
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
        unit = UNITS[_channel_family(chkey)]
        series = []
        for name, fname, color, marker in models:
            p = metrics_dir / fname
            # __BILINEAR__ etc. placeholders are resolved inside loader;
            # only skip the file-exists check for normal model JSONs.
            arr = loader(p)
            y = arr[:, cidx]
            series.append((name, color, marker, y))

        y_cap = None
        if clip_model:
            reference = [
                y[np.isfinite(y)]
                for name, _, _, y in series
                if name != clip_model and np.isfinite(y).any()
            ]
            if reference:
                y_cap = 1.08 * max(float(values.max()) for values in reference)

        for name, color, marker, y in series:
            plot_y = y.copy()
            clipped = np.zeros_like(plot_y, dtype=bool)
            if name == clip_model and y_cap is not None:
                clipped = np.isfinite(plot_y) & (plot_y > y_cap)
                plot_y[clipped] = y_cap
            ls = MODEL_LINESTYLES[name]
            lw = 1.2 if name in {"WeatherBridge", "WeatherDCAE-14M"} else 0.9
            label = {"S-DYff": "S-DYff (N=1)",
                     "S-DYff-ENS": "S-DYff-ENS (N=21)"}.get(name, name)
            (line,) = ax.plot(taus, plot_y, marker=marker, color=color,
                              linestyle=ls, linewidth=lw, markersize=2.4, label=label)
            if clipped.any():
                clipped_taus = np.asarray(taus)[clipped]
                ax.scatter(
                    clipped_taus,
                    np.full(clipped_taus.shape, y_cap),
                    marker=r"$\uparrow$",
                    s=22,
                    color=color,
                    zorder=5,
                    clip_on=False,
                )
            if ai == 0:
                legend_handles.append(line)
        heading = chkey if compact_channel_titles else chlabel
        ax.set_title(f"{heading}  [{unit}]", fontsize=title_size)
        ax.grid(alpha=0.3, lw=0.4)
        if len(taus) <= 5:
            ax.set_xticks(taus)
        else:
            ax.set_xticks(taus[::2])
            ax.set_xticks(taus[1::2], minor=True)
            ax.grid(axis="x", which="minor", alpha=0.3, lw=0.4)
            ax.set_xlim(min(taus) - 0.4, max(taus) + 0.4)
        if y_cap is not None:
            ax.set_ylim(top=y_cap * 1.04)
        ax.tick_params(axis="both", labelsize=6.0, width=0.5, length=2.2)
        if ai // ncols == nrows - 1:
            ax.set_xlabel("τ (h)", fontsize=6.6)
        if ai % ncols == 0:
            ax.set_ylabel("RMSE", fontsize=6.6)
    for ax in axes_flat[len(channels):]:
        ax.set_visible(False)
    add_panel_labels(list(axes.flat)[:len(channels)], inside=True, fontsize=6.6)
    if unseen_taus:
        legend_handles.append(Patch(facecolor="#ffd6d6", alpha=0.5,
                                    label="held-out τ"))
    fig.suptitle(title, fontsize=7.6, fontweight="bold", y=1.0)
    # The legend needs a fixed physical strip, so its reserved fraction has to
    # shrink as the grid gets taller rather than stay a constant proportion.
    legend_rows = -(-len(legend_handles) // 3)
    legend_in = 0.16 + 0.13 * legend_rows
    fig.tight_layout(rect=[0, legend_in / fig_height, 1, 1 - 0.16 / fig_height])
    fig.legend(handles=legend_handles, loc="lower center",
               ncol=min(len(legend_handles), 3),
               bbox_to_anchor=(0.5, 0.0), fontsize=6.6, frameon=False,
               handlelength=1.6, columnspacing=1.4)
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, bbox_inches="tight")
    print(f"  wrote {out_pdf}")
    plt.close(fig)


def models_12h(main_panel: bool = False) -> List[Tuple[str, str, str, str]]:
    models = [
        (name, filename, MODEL_COLORS[name], MODEL_MARKERS[name])
        for name, filename in [
            ("Linear Interp.", "__LINEAR__"),
            ("SwinV2", "fuxi_3yr_ep10.json"),
            ("ModAFNO", "modafno_3yr_ep10.json"),
            ("S-DYff", "sdyff_3yr_ep10.json"),
            ("S-DYff-ENS", "sdyff_ens21.json"),
            ("PixelAttn-VFI", "atm_vfi_3yr_ep10_matched.json"),
            ("WeatherDCAE-14M", "weatherdcae_14m_3yr_ep10_matched.json"),
            ("WeatherBridge", "weatherbridge_14m_3yr_ep10.json"),
        ]
    ]
    if main_panel:
        excluded = {"SwinV2", "ModAFNO"}
        return [model for model in models if model[0] not in excluded]
    return models


def models_ood() -> List[Tuple[str, str, str, str]]:
    return models_6h()


def _phys_from_entry(e: dict, ch: str) -> float:
    """Resolve a physical-unit RMSE for channel `ch` from a per-τ entry,
    handling three schemas:
      (a) rmse_phys_<CH> directly (12h schema — already physical, σ matches
          canonical so it is consistent across models),
      (b) rmse_norm_<CH> × σ_canonical (any schema with norm RMSE),
      (c) rmse_<CH> directly (OOD 2021 schema, already physical).
    Applies the Q kg/kg → g/kg ×1000 scale uniformly."""
    scale = UNIT_SCALE[_channel_family(ch)]
    v = e.get(f"rmse_phys_{ch}")
    if v is not None:
        return float(v) * scale
    v = e.get(f"rmse_norm_{ch}")
    if v is not None:
        return float(v) * _get_canonical_stds()[ch] * scale
    v = e.get(f"rmse_{ch}")
    if v is not None:
        return float(v) * scale
    return float("nan")


def load_12h_phys(path: Path) -> np.ndarray:
    """Returns rmse[τ, channel] in physical units (τ ∈ 1..11) from a 12h
    eval JSON via per_tau[tau].model schema."""
    if str(path).endswith("__LINEAR__"):
        ref = path.parent / "weatherbridge_14m_3yr_ep10.json"
        d = json.load(open(ref))
        pt = d["per_tau"]
        out = np.full((11, 24), np.nan)
        for ti, tau in enumerate(range(1, 12)):
            e = pt[str(tau)]["bilinear"]
            for ci, ch in enumerate(CHANNELS_ORDER):
                out[ti, ci] = _phys_from_entry(e, ch)
        return out
    d = json.load(open(path))
    pt = d["per_tau"]
    out = np.full((11, 24), np.nan)
    for ti, tau in enumerate(range(1, 12)):
        e = pt[str(tau)]["model"]
        for ci, ch in enumerate(CHANNELS_ORDER):
            out[ti, ci] = _phys_from_entry(e, ch)
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["6h", "12h", "ood2021", "extensions", "all"], default="all")
    args = p.parse_args()
    out = ROOT / "paper" / "images"

    if args.mode in ("6h", "all"):
        md = METRICS_ROOT / "6h_2020"
        plot_grid_phys(load_6h_phys, models_6h(), [1, 2, 3, 4, 5],
                       THERMODYNAMIC_CHANNELS, ncols=3,
                       out_pdf=out / "fig_fields_6h_thermodynamic_phys.pdf",
                       title="6 h, 2020 — temperature and humidity RMSE",
                       metrics_dir=md, unseen_taus=[2, 4],
                       height_ratio=0.95, compact_channel_titles=True)
        plot_grid_phys(load_6h_phys, models_6h(), [1, 2, 3, 4, 5],
                       WIND_CHANNELS, ncols=3,
                       out_pdf=out / "fig_fields_6h_wind_phys.pdf",
                       title="6 h, 2020 — wind-component RMSE",
                       metrics_dir=md, unseen_taus=[2, 4],
                       height_ratio=0.82, compact_channel_titles=True)
        plot_grid_phys(load_6h_phys, models_6h(), [1, 2, 3, 4, 5],
                       MASS_SURFACE_CHANNELS, ncols=3,
                       out_pdf=out / "fig_fields_6h_mass_surface_phys.pdf",
                       title="6 h, 2020 — geopotential and surface-field RMSE",
                       metrics_dir=md, unseen_taus=[2, 4],
                       height_ratio=1.30, compact_channel_titles=True)
        plot_grid_phys(load_6h_phys, models_6h(), [1, 2, 3, 4, 5],
                       BODY_CHANNELS, ncols=3,
                       out_pdf=out / "fig_channels_body_6h_phys.pdf",
                       title="6 h, 2020 — RMSE in physical units (850 hPa + surface)",
                       metrics_dir=md, unseen_taus=[2, 4])
        plot_grid_phys(load_6h_phys, models_6h(), [1, 2, 3, 4, 5],
                       APP_CHANNELS, ncols=3,
                       out_pdf=out / "fig_channels_app_6h_phys.pdf",
                       title="6 h, 2020 — RMSE in physical units (all PL channels)",
                       metrics_dir=md, unseen_taus=[2, 4])

    if args.mode in ("12h", "all"):
        md = METRICS_ROOT / "12h_2020"
        plot_grid_phys(load_12h_phys, models_12h(), list(range(1, 12)),
                       BODY_CHANNELS, ncols=3,
                       out_pdf=out / "fig_channels_body_12h_phys.pdf",
                       title="12 h, 2020 — RMSE in physical units "
                             "(850 hPa + surface)",
                       metrics_dir=md, unseen_taus=[4, 6, 8],
                       clip_model="ModAFNO")
        plot_grid_phys(load_12h_phys, models_12h(), list(range(1, 12)),
                       APP_CHANNELS, ncols=3,
                       out_pdf=out / "fig_channels_app_12h_phys.pdf",
                       title="12 h, 2020 — RMSE in physical units "
                             "(all PL channels)",
                       metrics_dir=md, unseen_taus=[4, 6, 8],
                       clip_model="ModAFNO")

    if args.mode in ("ood2021", "all"):
        md = METRICS_ROOT / "6h_2021"
        plot_grid_phys(
            load_6h_phys,
            models_ood(),
            [1, 2, 3, 4, 5],
            BODY_CHANNELS,
            ncols=3,
            out_pdf=out / "fig_channels_body_ood2021.pdf",
            title="6 h, 2021 transfer audit — RMSE in physical units",
            metrics_dir=md,
            unseen_taus=[2, 4],
        )
        plot_grid_phys(
            load_6h_phys,
            models_ood(),
            [1, 2, 3, 4, 5],
            APP_CHANNELS,
            ncols=3,
            out_pdf=out / "fig_channels_app_ood2021.pdf",
            title="6 h, 2021 transfer audit — pressure-level RMSE in physical units",
            metrics_dir=md,
            unseen_taus=[2, 4],
        )

    if args.mode in ("extensions", "all"):
        for horizon, year in ((12, 2021), (6, 2022), (12, 2022)):
            plot_grid_phys(
                load_6h_phys if horizon == 6 else load_12h_phys,
                models_6h() if horizon == 6 else models_12h(),
                list(range(1, horizon)), ALL_CHANNELS, ncols=4,
                out_pdf=out / f"fig_channels_all_{horizon}h_{year}_phys.pdf",
                title=f"{horizon} h, {year}"
                      + (" (48 anchor windows)" if year == 2022 else " (full year)")
                      + " - RMSE in physical units",
                metrics_dir=METRICS_ROOT / f"{horizon}h_{year}",
                unseen_taus=[2, 4] if horizon == 6 else [4, 6, 8],
                clip_model="ModAFNO" if horizon == 12 else None,
                height_ratio=0.82,
            )


if __name__ == "__main__":
    main()
