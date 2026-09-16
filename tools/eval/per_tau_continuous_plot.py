#!/usr/bin/env python3
"""Fig 9 generator: RMSE(τ) with seen vs unseen τ highlight for H2 hypothesis.

Reads per-model JSON files emitted by ``batch_eval_12h_memmap.py`` and
produces two figures:

  * ``fig9_continuous_tau.{png,pdf}`` — RMSE(τ) for τ ∈ {1, ..., 11}
    with grey shading + red ``x`` markers on the held-out (unseen) τ values
    (default ``{4, 6, 8}``).
  * ``fig9_continuous_tau_degradation.{png,pdf}`` — bar chart of
    ``(unseen_mean / seen_mean − 1) × 100`` per model, sorted ascending
    so the most-robust model appears first.

Inputs
------
``--metrics-dir`` should contain ``<name>.json`` files in the schema produced
by ``batch_eval_12h_memmap.py`` (each with ``per_tau`` and the canonical
``seen_tau`` / ``unseen_tau`` lists).

Each line on the main figure averages ``rmse_norm_<channel>`` across all
channels (``channel_names``). The bilinear and bicubic baseline lines are
extracted from the **first** model JSON found (they are identical across
files, modulo small batch-grouping differences).

Usage
-----
::

    python tools/eval/per_tau_continuous_plot.py \
        --metrics-dir metrics/eval_12h_2020_pre_ep10 \
        --out-dir figs/
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


# Default colour / style mapping by ``paper_tag`` (canonical name in the
# registry). Anything unmapped falls through to the ``--cmap`` fallback.
_DEFAULT_STYLE: Dict[str, Dict[str, Any]] = {
    "WeatherDCAE": {"color": "tab:red", "linestyle": "-", "marker": "o"},
    "DC-AE Skip": {"color": "tab:blue", "linestyle": "-", "marker": "s"},
    "DC-AE Skip (3yr)": {"color": "tab:cyan", "linestyle": "-", "marker": "s"},
    "DC-AE NoSkip (3yr)": {"color": "tab:pink", "linestyle": "-", "marker": "o"},
    "SwinV2": {"color": "tab:green", "linestyle": "-", "marker": "^"},
    "ModAFNO": {"color": "tab:purple", "linestyle": "-", "marker": "D"},
    "S-DYff": {"color": "tab:orange", "linestyle": "-", "marker": "v"},
    "ATM-VFI": {"color": "tab:brown", "linestyle": "-", "marker": "P"},
}

_BASELINE_STYLES: Dict[str, Dict[str, Any]] = {
    "bilinear": {"color": "0.4", "linestyle": "--", "marker": None, "label": "bilinear"},
    "bicubic": {"color": "0.6", "linestyle": ":", "marker": None, "label": "bicubic"},
}


def _avg_rmse_norm_per_tau(
    per_tau: Dict[str, Dict[str, Dict[str, float]]],
    method: str,
    channel_names: Sequence[str],
) -> Dict[int, float]:
    """Return ``{tau: mean over channels of rmse_norm}``."""
    out: Dict[int, float] = {}
    for tau_str, by_method in per_tau.items():
        block = by_method.get(method, {})
        if not block:
            continue
        vals = []
        for name in channel_names:
            key = f"rmse_norm_{name}"
            if key in block:
                vals.append(float(block[key]))
        if vals:
            out[int(tau_str)] = float(np.mean(vals))
    return out


def _load_model_json(path: Path) -> Dict[str, Any]:
    with open(path) as f:
        return json.load(f)


def _resolve_label(payload: Dict[str, Any], fallback_name: str) -> str:
    tag = payload.get("paper_tag")
    # If paper_tag is a generic timestamp ("12h_2020", "12h_2020_ep10", etc.),
    # fall back to the file stem so each model gets a unique label.
    if isinstance(tag, str) and tag and not tag.startswith("12h_2020"):
        return tag
    # Map common file stems to short display labels.
    stem_to_label = {
        "WeatherDCAE_NoSkip_6yr_12h": "WeatherDCAE",
        "DC-AE_NoSkip_3yr_12h_fibo": "DC-AE NoSkip (3yr)",
        "DC-AE_Skip_3yr_12h_fibo": "DC-AE Skip (3yr)",
        "FuXi_3yr_12h_fibo": "SwinV2",
        "ModAFNO_3yr_12h_fibo": "ModAFNO",
        "SDyff_3yr_12h_fibo": "S-DYff",
        "ATM-VFI_3yr_12h_fibo": "ATM-VFI",
    }
    return stem_to_label.get(fallback_name, fallback_name)


def _resolve_style(label: str, index: int, cmap_name: str) -> Dict[str, Any]:
    if label in _DEFAULT_STYLE:
        return dict(_DEFAULT_STYLE[label])
    import matplotlib.pyplot as plt  # lazy

    cmap = plt.get_cmap(cmap_name)
    return {
        "color": cmap(index % cmap.N),
        "linestyle": "-",
        "marker": "o",
    }


def _compute_degradation(
    per_tau_curve: Dict[int, float],
    seen_tau: Sequence[int],
    unseen_tau: Sequence[int],
) -> Optional[float]:
    seen_vals = [per_tau_curve[t] for t in seen_tau if t in per_tau_curve]
    unseen_vals = [per_tau_curve[t] for t in unseen_tau if t in per_tau_curve]
    if not seen_vals or not unseen_vals:
        return None
    seen_mean = float(np.mean(seen_vals))
    unseen_mean = float(np.mean(unseen_vals))
    if seen_mean <= 0.0:
        return None
    return (unseen_mean / seen_mean - 1.0) * 100.0


def _plot_continuous_tau(
    *,
    curves: List[Tuple[str, Dict[int, float], Dict[str, Any]]],
    baseline_curves: Dict[str, Dict[int, float]],
    seen_tau: Sequence[int],
    unseen_tau: Sequence[int],
    eval_tau: Sequence[int],
    out_dir: Path,
    figsize: Tuple[float, float],
) -> None:
    import matplotlib.pyplot as plt  # lazy

    fig, ax = plt.subplots(figsize=figsize)

    # Shaded vertical bands for unseen τ.
    for t in unseen_tau:
        ax.axvspan(t - 0.4, t + 0.4, color="0.85", alpha=0.5, zorder=0)

    # Baseline lines (extracted from the first model JSON).
    for bname, style in _BASELINE_STYLES.items():
        curve = baseline_curves.get(bname)
        if not curve:
            continue
        xs = sorted(curve.keys())
        ys = [curve[x] for x in xs]
        ax.plot(
            xs,
            ys,
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=1.4,
            label=style["label"],
            zorder=2,
        )

    # Model lines with seen/unseen markers.
    for label, curve, style in curves:
        xs = sorted(curve.keys())
        ys = [curve[x] for x in xs]
        ax.plot(
            xs,
            ys,
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=1.8,
            label=label,
            zorder=3,
        )
        # Seen markers (circles).
        seen_x = [x for x in xs if x in seen_tau]
        seen_y = [curve[x] for x in seen_x]
        if seen_x:
            ax.scatter(
                seen_x,
                seen_y,
                marker=style.get("marker", "o"),
                facecolors="white",
                edgecolors=style["color"],
                linewidths=1.4,
                s=42,
                zorder=4,
            )
        # Unseen markers (red ×).
        unseen_x = [x for x in xs if x in unseen_tau]
        unseen_y = [curve[x] for x in unseen_x]
        if unseen_x:
            ax.scatter(
                unseen_x,
                unseen_y,
                marker="x",
                color="red",
                linewidths=1.8,
                s=70,
                zorder=5,
            )

    ax.set_xlabel(r"$\tau$ (hours)")
    ax.set_ylabel("Avg RMSE (normalised units) over 24 channels")
    ax.set_xticks(list(eval_tau))
    title = (
        "RMSE vs interpolation step τ for the 12 h window\n"
        f"Held-out τ ∈ {{{', '.join(str(t) for t in unseen_tau)}}} — "
        "measures continuous-time generalization"
    )
    ax.set_title(title)
    ax.grid(True, linestyle=":", alpha=0.45)
    ax.legend(loc="upper left", framealpha=0.9, fontsize=9, ncol=2)

    out_dir.mkdir(parents=True, exist_ok=True)
    png_path = out_dir / "fig9_continuous_tau.png"
    pdf_path = out_dir / "fig9_continuous_tau.pdf"
    fig.tight_layout()
    fig.savefig(png_path, dpi=200)
    fig.savefig(pdf_path)
    plt.close(fig)
    print(f"saved {png_path}")
    print(f"saved {pdf_path}")


def _plot_degradation_bar(
    *,
    degradations: List[Tuple[str, Optional[float]]],
    out_dir: Path,
    figsize: Tuple[float, float],
) -> Optional[str]:
    import matplotlib.pyplot as plt  # lazy

    valid = [(lbl, v) for lbl, v in degradations if v is not None]
    if not valid:
        print("  [warn] no valid degradation values; skipping bar chart")
        return None
    # Sort ascending: lower degradation = more robust.
    valid.sort(key=lambda kv: kv[1])  # type: ignore[arg-type]
    labels = [lbl for lbl, _ in valid]
    values = [v for _, v in valid]  # type: ignore[misc]

    fig, ax = plt.subplots(figsize=figsize)
    colors = ["tab:green" if v <= 0 else "tab:red" for v in values]
    ax.bar(labels, values, color=colors)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel(r"Degradation on unseen $\tau$ (%)")
    ax.set_title(
        "Continuous-τ generalization gap\n"
        r"$(\overline{\mathrm{RMSE}}_{\mathrm{unseen}} / "
        r"\overline{\mathrm{RMSE}}_{\mathrm{seen}} - 1) \times 100$"
    )
    ax.tick_params(axis="x", rotation=20)
    ax.grid(True, axis="y", linestyle=":", alpha=0.45)

    out_dir.mkdir(parents=True, exist_ok=True)
    png_path = out_dir / "fig9_continuous_tau_degradation.png"
    pdf_path = out_dir / "fig9_continuous_tau_degradation.pdf"
    fig.tight_layout()
    fig.savefig(png_path, dpi=200)
    fig.savefig(pdf_path)
    plt.close(fig)
    print(f"saved {png_path}")
    print(f"saved {pdf_path}")
    most_robust = labels[0]
    return most_robust


def _build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--metrics-dir",
        default="metrics/eval_12h_2020_pre_ep10",
        help="Directory with per-model JSONs from batch_eval_12h_memmap.py.",
    )
    ap.add_argument(
        "--out-dir",
        default="figs",
        help="Output directory for the generated PNG/PDF files.",
    )
    ap.add_argument(
        "--cmap",
        default="tab10",
        help="Matplotlib colormap for fallback line colors (default: tab10).",
    )
    ap.add_argument(
        "--figsize",
        default="9,5",
        help="Comma-separated width,height (inches) for the main figure.",
    )
    ap.add_argument(
        "--bar-figsize",
        default="7,4",
        help="Comma-separated width,height (inches) for the degradation bar plot.",
    )
    ap.add_argument(
        "--include",
        default="",
        help=(
            "Comma-separated list of paper_tag/file-stem names to include "
            "(default: include all JSONs found)."
        ),
    )
    ap.add_argument(
        "--exclude",
        default="",
        help="Comma-separated list of names to exclude.",
    )
    return ap


def main() -> None:
    ap = _build_arg_parser()
    args = ap.parse_args()

    metrics_dir = Path(args.metrics_dir)
    if not metrics_dir.is_dir():
        raise SystemExit(f"--metrics-dir does not exist: {metrics_dir}")

    json_paths = sorted(metrics_dir.glob("*.json"))
    if not json_paths:
        raise SystemExit(f"no *.json found in {metrics_dir}")

    include = {x.strip() for x in args.include.split(",") if x.strip()}
    exclude = {x.strip() for x in args.exclude.split(",") if x.strip()}

    curves: List[Tuple[str, Dict[int, float], Dict[str, Any]]] = []
    seen_tau: List[int] = []
    unseen_tau: List[int] = []
    eval_tau: List[int] = []
    channel_names: List[str] = []
    baseline_curves: Dict[str, Dict[int, float]] = {}
    degradations: List[Tuple[str, Optional[float]]] = []

    for idx, path in enumerate(json_paths):
        payload = _load_model_json(path)
        per_tau = payload.get("per_tau", {})
        if not per_tau:
            print(f"  [skip] {path}: no per_tau block")
            continue
        names_here = payload.get("channel_names") or []
        if not channel_names:
            channel_names = list(names_here)
            seen_tau = list(payload.get("seen_tau", []))
            unseen_tau = list(payload.get("unseen_tau", []))
            eval_tau = sorted(int(t) for t in per_tau.keys())
            for bname in _BASELINE_STYLES:
                baseline_curves[bname] = _avg_rmse_norm_per_tau(
                    per_tau, bname, channel_names
                )

        label = _resolve_label(payload, path.stem)
        if include and label not in include and path.stem not in include:
            continue
        if label in exclude or path.stem in exclude:
            continue

        style = _resolve_style(label, idx, args.cmap)
        model_curve = _avg_rmse_norm_per_tau(per_tau, "model", names_here or channel_names)
        if not model_curve:
            print(f"  [skip] {path}: empty model curve")
            continue
        curves.append((label, model_curve, style))
        deg = _compute_degradation(model_curve, seen_tau, unseen_tau)
        degradations.append((label, deg))
        if deg is not None:
            print(f"  {label}: degradation = {deg:+.2f}% on unseen τ")
        else:
            print(f"  {label}: degradation = N/A (missing τ values)")

    if not curves:
        raise SystemExit("no eligible model curves to plot")

    figsize = tuple(float(x) for x in args.figsize.split(","))
    bar_figsize = tuple(float(x) for x in args.bar_figsize.split(","))

    out_dir = Path(args.out_dir)
    _plot_continuous_tau(
        curves=curves,
        baseline_curves=baseline_curves,
        seen_tau=seen_tau,
        unseen_tau=unseen_tau,
        eval_tau=eval_tau,
        out_dir=out_dir,
        figsize=figsize,  # type: ignore[arg-type]
    )
    most_robust = _plot_degradation_bar(
        degradations=degradations,
        out_dir=out_dir,
        figsize=bar_figsize,  # type: ignore[arg-type]
    )

    print("\n=== Summary ===")
    print(f"  seen τ:   {seen_tau}")
    print(f"  unseen τ: {unseen_tau}")
    for lbl, deg in degradations:
        if deg is None:
            print(f"  {lbl:>22s}: degradation = N/A")
        else:
            print(f"  {lbl:>22s}: degradation = {deg:+.2f}%")
    if most_robust:
        print(f"\n  Most robust to held-out τ: {most_robust}")


if __name__ == "__main__":
    main()
