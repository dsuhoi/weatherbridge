#!/usr/bin/env python3
"""Build Appendix table + pgfplots figure: 6yr 0.5° model comparison per surface
variable, per interior hour h=1..5.

Reads metrics/eval_6h_2020_paper_leaderboard/*.json (full-coverage = 192 windows) and emits:
  paper/A1_surface_per_hour_table.tex   — one block per surface var, 4 models × h=1..5
  paper/A2_surface_per_hour_plot.tex    — 7-panel groupplot, x=h, y=Δ% vs bicubic

Surface variables: t2m, u10, v10, mslp, sst, tcc, tcwv.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np

# Models in display order (input JSON stem → display name + colour)
MODEL_ORDER = [
    ("dcae_skip_0p5_6yr_pad", r"\textbf{DC-AE Skip (PAD)}", "ForestGreen!80!black", "*"),
    ("fuxi_0p5_6yr",          "SwinV2",                     "blue!70!black",        "triangle*"),
    ("sdyff_dyffusion_0p5_6yr","S-DYff DYffusion",          "orange!90!black",      "diamond*"),
    ("modafno_0p5_6yr",       "ModAFNO",                    "red!80!black",         "square*"),
]

SURF_VARS = ["t2m", "u10", "v10", "mslp", "sst", "tcc", "tcwv"]
HOURS = [1, 2, 3, 4, 5]


def load_per_hour(p: Path) -> Dict[int, Dict[str, Dict[str, float]]]:
    d = json.loads(p.read_text())
    return {int(k): v for k, v in d["per_hour"].items()}


def get_rmse(per_hour, which: str, name: str, h: int):
    if h not in per_hour:
        return float("nan")
    sub = per_hour[h].get(which, {})
    return float(sub.get(f"rmse_{name}", float("nan")))


def fmt_delta(d: float) -> str:
    if not np.isfinite(d):
        return "---"
    sign = "+" if d > 0 else ""
    color = "ForestGreen" if d < 0 else "BrickRed"
    return rf"\textcolor{{{color}}}{{{sign}{d:.1f}}}"


def build_table(model_data: Dict[str, Dict[int, dict]], out: Path):
    lines = []
    lines.append(r"% Appendix table: 6yr 0.5° per-surface-var ΔRMSE vs bicubic (auto-generated)")
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{6-year-trained 0.5$^\circ$ models on 2020 test: per-surface-variable "
                 r"$\Delta$RMSE vs bicubic baseline at interior hours $h{=}1\ldots5$. "
                 r"Values are \% change (lower / negative = model beats baseline).}")
    lines.append(r"\label{tab:appendix_surface_6yr}")
    lines.append(r"\small")
    lines.append(r"\setlength{\tabcolsep}{6pt}")
    lines.append(r"\begin{tabular}{llrrrrr}")
    lines.append(r"\toprule")
    lines.append(r"\textbf{Var} & \textbf{Model} & "
                 r"$h{=}1$ & $h{=}2$ & $h{=}3$ & $h{=}4$ & $h{=}5$ \\")
    lines.append(r"\midrule")

    for vi, var in enumerate(SURF_VARS):
        if vi > 0:
            lines.append(r"\midrule")
        for mi, (stem, disp, _color, _mark) in enumerate(MODEL_ORDER):
            per_hour = model_data.get(stem)
            cells = []
            for h in HOURS:
                if per_hour is None:
                    cells.append("---")
                    continue
                m = get_rmse(per_hour, "model", var, h)
                b = get_rmse(per_hour, "bicubic", var, h)
                if not (np.isfinite(m) and np.isfinite(b) and b > 0):
                    cells.append("---")
                else:
                    cells.append(fmt_delta((m - b) / b * 100))
            var_cell = rf"\texttt{{{var}}}" if mi == 0 else ""
            lines.append(f"{var_cell} & {disp} & " + " & ".join(cells) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table*}")

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    print(f"wrote {out}")


def build_plot(model_data: Dict[str, Dict[int, dict]], out: Path):
    """Groupplot 7 panels (one per surface var) + a single external legend on top.

    Uses `legend to name` to render a horizontal legend ABOVE the group of plots,
    so legend never overlaps any panel.
    """
    lines = []
    lines.append(r"% Appendix figure: 6yr 0.5° per-surface-var per-hour RMSE Δ% (auto-generated)")
    lines.append(r"\documentclass[border=2pt]{standalone}")
    lines.append(r"\usepackage[dvipsnames]{xcolor}")
    lines.append(r"\usepackage{tikz}")
    lines.append(r"\usepackage{pgfplots}")
    lines.append(r"\usetikzlibrary{calc}")
    lines.append(r"\usepgfplotslibrary{groupplots}")
    lines.append(r"\pgfplotsset{compat=1.18}")
    lines.append(r"\begin{document}")
    lines.append(r"\begin{tikzpicture}")
    lines.append(r"\begin{groupplot}[")
    lines.append(r"  group style={group size=4 by 2, horizontal sep=1.6cm, vertical sep=1.5cm},")
    lines.append(r"  width=5.5cm, height=4.2cm,")
    lines.append(r"  xlabel={Hour $h$}, xtick={1,2,3,4,5}, xticklabels={1,2,3,4,5},")
    lines.append(r"  ylabel={$\Delta$RMSE \% vs bicubic},")
    lines.append(r"  ylabel near ticks,")
    lines.append(r"  grid=major, grid style={gray!20},")
    lines.append(r"]")

    for vi, var in enumerate(SURF_VARS):
        lines.append(rf"\nextgroupplot[title={{\texttt{{{var}}}}}, xmin=0.5, xmax=5.5]")
        for stem, disp, color, mark in MODEL_ORDER:
            per_hour = model_data.get(stem)
            if per_hour is None:
                continue
            coords = []
            for h in HOURS:
                m = get_rmse(per_hour, "model", var, h)
                b = get_rmse(per_hour, "bicubic", var, h)
                if not (np.isfinite(m) and np.isfinite(b) and b > 0):
                    continue
                d = (m - b) / b * 100
                coords.append(f"({h}, {d:.2f})")
            if not coords:
                continue
            lines.append(rf"\addplot[color={color}, mark={mark}, mark size=2pt, line width=1.0pt] "
                         f"coordinates {{ {' '.join(coords)} }};")

    # 8th panel: dedicated legend panel (hidden axis, visible legend at center).
    lines.append(r"\nextgroupplot[hide axis, axis lines=none, xmin=0, xmax=1, ymin=0, ymax=1,")
    lines.append(r"  legend style={")
    lines.append(r"    at={(0.5,0.5)}, anchor=center,")
    lines.append(r"    draw=black!30, fill=white, font=\small,")
    lines.append(r"    legend cell align=left, row sep=2pt,")
    lines.append(r"  }]")
    for stem, disp, color, mark in MODEL_ORDER:
        lines.append(rf"\addlegendimage{{color={color}, mark={mark}, mark size=2.5pt, line width=1.0pt}}")
        lines.append(rf"\addlegendentry{{{disp}}}")
    lines.append(r"\addplot[only marks, mark=none, opacity=0] coordinates {(0.5,0.5)};")

    lines.append(r"\end{groupplot}")
    lines.append(r"\end{tikzpicture}")
    lines.append(r"\end{document}")

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    print(f"wrote {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", default="metrics/eval_6h_2020_paper_leaderboard")
    ap.add_argument("--out-table", default="paper/A1_surface_per_hour_table.tex")
    ap.add_argument("--out-plot", default="paper/A2_surface_per_hour_plot.tex")
    args = ap.parse_args()

    in_dir = Path(args.in_dir)
    model_data: Dict[str, Dict[int, dict]] = {}
    for stem, _disp, _c, _m in MODEL_ORDER:
        p = in_dir / f"{stem}.json"
        if p.exists():
            model_data[stem] = load_per_hour(p)
        else:
            print(f"[skip] {p} not found")

    if not model_data:
        raise SystemExit(f"no JSONs found in {in_dir}")

    build_table(model_data, Path(args.out_table))
    build_plot(model_data, Path(args.out_plot))


if __name__ == "__main__":
    main()
