#!/usr/bin/env python3
"""Build per-channel x per-hour ΔRMSE-vs-bicubic LaTeX table from 0.5° eval JSONs.

Reads metrics/eval_6h_2020_paper_leaderboard/*.json (output of batch_eval_memmap.py) and
emits paper/10_0p5_rmse_per_channel.tex.

Aggregation: pressure-level channels are averaged by variable family (T, U, V, Q, Z).
Each row = (model, channel-group), columns = h=1..5, values = (RMSE_model - RMSE_bicubic)/RMSE_bicubic * 100.
Negative = model beats bicubic.

Usage:
    python tools/eval/build_0p5_rmse_table.py \
        --in-dir metrics/eval_6h_2020_paper_leaderboard \
        --out paper/10_0p5_rmse_per_channel.tex
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np

# Models in display order — file basename (no .json) -> display name
MODEL_ORDER = [
    ("dcae_skip_0p5_pad", r"\textbf{DC-AE Skip (PAD)}"),
    ("fuxi_0p5_6yr",      "FuXi SwinV2 (6yr)"),
    ("fuxi_0p5_2018",     "FuXi SwinV2 (1yr)"),
    ("modafno_0p5_2018",  "ModAFNO (1yr)"),
]

# Channel groups (display label -> matching channel names from the JSON)
PL_VARS = ["T", "U", "V", "Q", "Z"]
PL_LEVELS = [1000, 925, 850, 700]

GROUPS: List[tuple] = []
for v in PL_VARS:
    GROUPS.append((v, [f"{v}{lvl}" for lvl in PL_LEVELS]))
# Surface — keep individual since they're distinct physics
for s in ["t2m", "u10", "v10", "mslp", "sst", "tcc"]:
    GROUPS.append((s, [s]))


def load_per_hour(json_path: Path) -> Dict[int, Dict[str, Dict[str, float]]]:
    d = json.loads(json_path.read_text())
    return {int(k): v for k, v in d["per_hour"].items()}


def group_rmse(per_hour: Dict[int, Dict[str, Dict[str, float]]],
               which: str, names: List[str], hours: List[int]) -> Dict[int, float]:
    """Mean RMSE across `names` per hour for `which` in {'model','bilinear','bicubic'}."""
    out = {}
    for h in hours:
        if h not in per_hour:
            out[h] = float("nan")
            continue
        sub = per_hour[h].get(which, {})
        vals = []
        for n in names:
            key = f"rmse_{n}"
            if key in sub:
                vals.append(float(sub[key]))
        out[h] = float(np.mean(vals)) if vals else float("nan")
    return out


def fmt_delta(d: float) -> str:
    if not np.isfinite(d):
        return "---"
    sign = "+" if d > 0 else ""
    color = "ForestGreen" if d < 0 else "BrickRed"
    return rf"\textcolor{{{color}}}{{{sign}{d:.1f}}}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", default="metrics/eval_6h_2020_paper_leaderboard")
    ap.add_argument("--out", default="paper/10_0p5_rmse_per_channel.tex")
    ap.add_argument("--hours", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    args = ap.parse_args()

    in_dir = Path(args.in_dir)
    hours = args.hours

    # Gather per-model data
    model_data = {}
    for stem, _disp in MODEL_ORDER:
        p = in_dir / f"{stem}.json"
        if not p.exists():
            print(f"[skip] {p} not found")
            continue
        model_data[stem] = load_per_hour(p)

    if not model_data:
        raise SystemExit(f"no JSONs found in {in_dir}")

    # Build LaTeX
    n_h = len(hours)
    col_spec = "ll" + "r" * n_h
    header_h = " & ".join([rf"$h{{=}}{h}$" for h in hours])
    lines = []
    lines.append(r"% 0.5° per-channel x per-hour ΔRMSE vs bicubic (auto-generated)")
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{Per-channel $\Delta$RMSE vs bicubic baseline at 0.5$^\circ$ on 2020 test set, "
                 r"interior hours $h{=}1..5$. Values are \% change (lower / negative is better). "
                 r"Pressure-level rows show mean across levels (1000, 925, 850, 700 hPa). "
                 r"Bicubic with 2 anchors degenerates to bilinear, so this is the cheapest spline baseline.}")
    lines.append(r"\label{tab:rmse_per_channel_0p5}")
    lines.append(r"\small")
    lines.append(r"\setlength{\tabcolsep}{4pt}")
    lines.append(rf"\begin{{tabular}}{{{col_spec}}}")
    lines.append(r"\toprule")
    lines.append(rf"\textbf{{Model}} & \textbf{{Var}} & {header_h} \\")
    lines.append(r"\midrule")

    first = True
    for stem, disp in MODEL_ORDER:
        if stem not in model_data:
            continue
        per_hour = model_data[stem]
        if not first:
            lines.append(r"\midrule")
        first = False
        for gi, (label, names) in enumerate(GROUPS):
            model_h = group_rmse(per_hour, "model", names, hours)
            bic_h = group_rmse(per_hour, "bicubic", names, hours)
            deltas = []
            for h in hours:
                m, b = model_h[h], bic_h[h]
                if not (np.isfinite(m) and np.isfinite(b) and b > 0):
                    deltas.append(float("nan"))
                else:
                    deltas.append((m - b) / b * 100.0)
            row_model = disp if gi == 0 else ""
            cells = " & ".join(fmt_delta(d) for d in deltas)
            lines.append(f"{row_model} & \\texttt{{{label}}} & {cells} \\\\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table*}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    print(f"wrote {out} ({len(lines)} lines)")


if __name__ == "__main__":
    main()
