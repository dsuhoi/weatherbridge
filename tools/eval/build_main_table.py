#!/usr/bin/env python3
"""Build canonical paper Tab 1: headline 0.5° 6yr results.

Columns: Model | Params | Latency (ms) | h=1..5 (Δ%) | Mean ± CI
Rows: numerical baselines (sorted by quality), then ML models.

Numbers come from:
  - metrics/eval_6h_2020_paper_leaderboard/    (4 ML models)
  - metrics/eval_0p5_2020_numeric/ (5 numeric methods)
  - Bootstrap CI on the fly
  - Inference cost: hard-coded from A4 (regenerate via tools/eval/inference_cost.py)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


# Model display info (ordered by paper narrative)
NUMERIC = [
    ("hermite_advection",  "Hermite-advection",         "metrics/eval_0p5_2020_numeric"),
    ("settls",             "SETTLS (Hortal 2002)",      "metrics/eval_0p5_2020_numeric"),
    ("semi_lagrangian",    "Semi-Lagrangian",            "metrics/eval_0p5_2020_numeric"),
    ("bilinear",           "Bilinear (reference)",       "metrics/eval_0p5_2020_numeric"),
]
ML = [
    # (stem, disp, params_M, latency_ms, sub)
    ("dcae_skip_0p5_6yr_pad",    r"\textbf{DC-AE Skip (PAD) 6yr}", 14.4, 74.8,  "metrics/eval_6h_2020_paper_leaderboard"),
    ("modafno_0p5_6yr",          "ModAFNO 6yr",                    47.1, 94.4,  "metrics/eval_6h_2020_paper_leaderboard"),
    ("sdyff_dyffusion_0p5_6yr",  "S-DYff DYffusion 6yr",          101.3, 89.7,  "metrics/eval_6h_2020_paper_leaderboard"),
    ("fuxi_0p5_6yr",             "FuXi SwinV2 6yr",                 7.9, 24.6,  "metrics/eval_6h_2020_paper_leaderboard"),
]
HOURS = [1, 2, 3, 4, 5]


def per_h_delta_from_bicubic(d, h: int):
    """Per-channel Δ% vs the 'bicubic' field stored inside the JSON."""
    deltas = []
    for cn, mv in d['per_hour'][str(h)]['model'].items():
        bv = float(d['per_hour'][str(h)]['bicubic'][cn])
        if bv > 0:
            deltas.append((float(mv) - bv) / bv * 100.0)
    return np.array(deltas)


def per_h_delta_from_bilinear(d, bil, h: int):
    """Per-channel Δ% vs explicit bilinear JSON."""
    deltas = []
    for cn, mv in d['per_hour'][str(h)]['model'].items():
        bv = float(bil['per_hour'][str(h)]['model'][cn])
        if bv > 0:
            deltas.append((float(mv) - bv) / bv * 100.0)
    return np.array(deltas)


def bootstrap_ci(values, B: int = 2000, seed: int = 0):
    if len(values) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    n = len(values)
    means = np.array([values[rng.integers(0, n, size=n)].mean() for _ in range(B)])
    return float(values.mean()), float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def fmt_delta(d):
    if not np.isfinite(d):
        return "---"
    sign = "+" if d > 0 else ""
    color = "ForestGreen" if d < 0 else ("BrickRed" if d > 0 else "black")
    return rf"\textcolor{{{color}}}{{{sign}{d:.1f}}}"


def fmt_ci(point, lo, hi):
    if not np.isfinite(point):
        return "---"
    sign = "+" if point > 0 else ""
    color = "ForestGreen" if point < 0 else ("BrickRed" if point > 0 else "black")
    hw = max(point - lo, hi - point)
    return rf"\textcolor{{{color}}}{{{sign}{point:.1f}}}\,\textcolor{{black!50}}{{$\pm${hw:.1f}}}"


def main():
    # Load bilinear ref for numeric models
    bil = json.loads(Path("metrics/eval_0p5_2020_numeric/bilinear.json").read_text())

    lines = []
    lines.append(r"% Main paper table: headline 0.5° 6yr results (auto-generated)")
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{Main results on 0.5$^\circ$ ERA5 (360$\times$720 grid, 24 fields, "
                 r"6-hour interpolation, test on 2020). $\Delta$RMSE\% is per-hour change vs "
                 r"bicubic baseline; numerical methods compared to bilinear (bicubic degenerates "
                 r"with 2 anchors). Mean column reports bootstrap 95\% CI over 24 fields "
                 r"($B{=}2000$). Latency is per-sample on A100 80GB, fp32, batch=1.}")
    lines.append(r"\label{tab:main_0p5_6yr}")
    lines.append(r"\small")
    lines.append(r"\setlength{\tabcolsep}{4pt}")
    lines.append(r"\begin{tabular}{lrrrrrrrr}")
    lines.append(r"\toprule")
    lines.append(r"\textbf{Method} & \textbf{Params} & \textbf{Lat.} & "
                 r"$h{=}1$ & $h{=}2$ & $h{=}3$ & $h{=}4$ & $h{=}5$ & \textbf{Mean$\pm$95\%CI} \\")
    lines.append(r"  & (M) & (ms) & \multicolumn{5}{c}{$\Delta$RMSE\% (lower = better)} & \\")
    lines.append(r"\midrule")
    lines.append(r"\multicolumn{9}{l}{\textit{Numerical baselines (vs bilinear)}} \\")
    lines.append(r"\midrule")
    for stem, disp, sub in NUMERIC:
        p = Path(sub) / f"{stem}.json"
        if not p.exists():
            continue
        d = json.loads(p.read_text())
        per_h_cells = []
        all_deltas = []
        for h in HOURS:
            if stem == "bilinear":
                cells = "0.0"
                per_h_cells.append(cells)
                all_deltas.extend([0.0] * 24)
                continue
            deltas = per_h_delta_from_bilinear(d, bil, h)
            per_h_cells.append(fmt_delta(deltas.mean()))
            all_deltas.extend(deltas.tolist())
        if stem == "bilinear":
            ci_cell = r"0.0 (ref)"
        else:
            pt, lo, hi = bootstrap_ci(np.array(all_deltas), B=2000, seed=hash(stem) & 0xFFFF)
            ci_cell = fmt_ci(pt, lo, hi)
        lines.append(f"{disp} & --- & 0.14 & " + " & ".join(per_h_cells) + f" & {ci_cell} \\\\")
    lines.append(r"\midrule")
    lines.append(r"\multicolumn{9}{l}{\textit{ML models (vs bicubic; trained 6 years 2014--2019, 8 epochs)}} \\")
    lines.append(r"\midrule")
    for stem, disp, params_M, lat_ms, sub in ML:
        p = Path(sub) / f"{stem}.json"
        if not p.exists():
            continue
        d = json.loads(p.read_text())
        per_h_cells = []
        all_deltas = []
        for h in HOURS:
            deltas = per_h_delta_from_bicubic(d, h)
            per_h_cells.append(fmt_delta(deltas.mean()))
            all_deltas.extend(deltas.tolist())
        pt, lo, hi = bootstrap_ci(np.array(all_deltas), B=2000, seed=hash(stem) & 0xFFFF)
        ci_cell = fmt_ci(pt, lo, hi)
        lines.append(f"{disp} & {params_M:.1f} & {lat_ms:.0f} & " +
                     " & ".join(per_h_cells) + f" & {ci_cell} \\\\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table*}")
    out = Path("paper/01_main_table_0p5_6yr.tex")
    out.write_text("\n".join(lines) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
