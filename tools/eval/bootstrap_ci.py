#!/usr/bin/env python3
"""Bootstrap confidence intervals for per-hour RMSE on existing eval JSONs.

Resamples the fixed 24-field protocol with replacement (B=2000 by default) to get 95% CI on the
mean per-hour Δ% vs bicubic. Channel-level bootstrap; tighter sample-level CI
requires per-window error logging (next iteration).

Outputs paper/A3_bootstrap_ci_table.tex with 95% CI intervals.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


def _deterministic_seed(label: str, base_seed: int = 0) -> int:
    """Return a stable 32-bit seed derived from `label` + `base_seed`.

    Python's builtin ``hash()`` is randomised per-process (PYTHONHASHSEED),
    so it must not be used for reproducible RNG seeding. We use MD5 over the
    UTF-8 bytes and XOR with the user-supplied base seed.
    """
    digest = hashlib.md5(label.encode("utf-8")).digest()[:8]
    return (int.from_bytes(digest, "big") ^ base_seed) & 0xFFFFFFFF

ML_MODELS = [
    ("dcae_skip_0p5_6yr_pad",    r"\textbf{DC-AE Skip (PAD) 6yr}",  "metrics/eval_6h_2020_paper_leaderboard"),
    ("sdyff_dyffusion_0p5_6yr",  "S-DYff DYffusion 6yr",            "metrics/eval_6h_2020_paper_leaderboard"),
    ("modafno_0p5_6yr",          "ModAFNO 6yr",                      "metrics/eval_6h_2020_paper_leaderboard"),
    ("fuxi_0p5_6yr",             "FuXi SwinV2 6yr",                  "metrics/eval_6h_2020_paper_leaderboard"),
]
NUM_BASELINES = [
    ("hermite_advection",  "Hermite-advection",     "metrics/eval_0p5_2020_numeric"),
    ("hermite_diffusion",  "Hermite + diffusion",   "metrics/eval_0p5_2020_numeric"),
    ("settls",             "SETTLS",                 "metrics/eval_0p5_2020_numeric"),
    ("semi_lagrangian",    "Semi-Lagrangian",        "metrics/eval_0p5_2020_numeric"),
]
HOURS = [1, 2, 3, 4, 5]


def per_channel_deltas(d_model, d_ref, h: int) -> np.ndarray:
    """Per-channel Δ% vs reference (bicubic for ML, bilinear for numeric)."""
    deltas = []
    for ch_name, val in d_model['per_hour'][str(h)]['model'].items():
        ref_val = d_ref['per_hour'][str(h)]['model'][ch_name]
        m = float(val); r = float(ref_val)
        if r > 0:
            deltas.append((m - r) / r * 100.0)
    return np.array(deltas, dtype=np.float64)


def bootstrap_ci(values: np.ndarray, B: int = 2000, alpha: float = 0.05, seed: int = 0):
    """Channel-level bootstrap CI for mean(values)."""
    rng = np.random.default_rng(seed)
    n = len(values)
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    means = np.empty(B, dtype=np.float64)
    for b in range(B):
        idx = rng.integers(0, n, size=n)
        means[b] = values[idx].mean()
    point = float(values.mean())
    lo = float(np.quantile(means, alpha / 2.0))
    hi = float(np.quantile(means, 1.0 - alpha / 2.0))
    return point, lo, hi


def fmt_ci(point: float, lo: float, hi: float) -> str:
    if not np.isfinite(point):
        return "---"
    sign = "+" if point > 0 else ""
    color = "ForestGreen" if point < 0 else "BrickRed"
    half_width = max(point - lo, hi - point)
    return rf"\textcolor{{{color}}}{{{sign}{point:.1f}}}\,\textcolor{{black!50}}{{$\pm${half_width:.1f}}}"


def load_json(path: Path):
    return json.loads(path.read_text())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="paper/A3_bootstrap_ci_table.tex")
    ap.add_argument("--bootstraps", type=int, default=2000)
    args = ap.parse_args()

    # ML models use bicubic from their own JSON as reference; numeric use bilinear (in 'model' field of bilinear.json)
    ml_data = {}
    for stem, _, sub in ML_MODELS:
        p = Path(sub) / f"{stem}.json"
        if p.exists():
            ml_data[stem] = load_json(p)
        else:
            print(f"[skip] {p}")
    bil_for_num = None
    num_data = {}
    bil_path = Path("metrics/eval_0p5_2020_numeric/bilinear.json")
    if bil_path.exists():
        bil_for_num = load_json(bil_path)
    for stem, _, sub in NUM_BASELINES:
        p = Path(sub) / f"{stem}.json"
        if p.exists():
            num_data[stem] = load_json(p)

    # Build table
    lines = []
    lines.append(r"% Bootstrap 95% CIs on per-hour Δ\% RMSE (auto-generated)")
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    lines.append(rf"\caption{{Bootstrap 95\% confidence intervals on per-hour $\Delta$RMSE (\%) "
                 rf"for 0.5$^\circ$ models on 2020 test. Field-level resampling (24 fields), "
                 rf"$B={args.bootstraps}$ replicates. ML models compared vs bicubic baseline; "
                 rf"numerical baselines vs bilinear. Negative = better than baseline.}}")
    lines.append(r"\label{tab:bootstrap_ci_6yr}")
    lines.append(r"\small")
    lines.append(r"\setlength{\tabcolsep}{5pt}")
    lines.append(r"\begin{tabular}{lrrrrr}")
    lines.append(r"\toprule")
    lines.append(r"\textbf{Method} & $h{=}1$ & $h{=}2$ & $h{=}3$ & $h{=}4$ & $h{=}5$ \\")
    lines.append(r"\midrule")
    lines.append(r"\multicolumn{6}{l}{\textit{ML models (vs bicubic baseline)}} \\")
    lines.append(r"\midrule")
    for stem, disp, sub in ML_MODELS:
        if stem not in ml_data:
            continue
        d = ml_data[stem]
        # bicubic reference: from same JSON, in 'bicubic' key
        # We resampled channels but need to compute per-channel Δ
        row_cells = []
        for h in HOURS:
            ch_names = list(d['per_hour'][str(h)]['model'].keys())
            deltas = []
            for cn in ch_names:
                bic_key = cn  # 'rmse_T1000'
                m = float(d['per_hour'][str(h)]['model'][cn])
                b = float(d['per_hour'][str(h)]['bicubic'][bic_key])
                if b > 0:
                    deltas.append((m - b) / b * 100.0)
            point, lo, hi = bootstrap_ci(np.array(deltas), B=args.bootstraps, seed=_deterministic_seed(stem + str(h)))
            row_cells.append(fmt_ci(point, lo, hi))
        lines.append(f"{disp} & " + " & ".join(row_cells) + r" \\")
    lines.append(r"\midrule")
    lines.append(r"\multicolumn{6}{l}{\textit{Numerical baselines (vs bilinear)}} \\")
    lines.append(r"\midrule")
    for stem, disp, sub in NUM_BASELINES:
        if stem not in num_data or bil_for_num is None:
            continue
        d = num_data[stem]
        row_cells = []
        for h in HOURS:
            ch_names = list(d['per_hour'][str(h)]['model'].keys())
            deltas = []
            for cn in ch_names:
                m = float(d['per_hour'][str(h)]['model'][cn])
                b = float(bil_for_num['per_hour'][str(h)]['model'][cn])
                if b > 0:
                    deltas.append((m - b) / b * 100.0)
            point, lo, hi = bootstrap_ci(np.array(deltas), B=args.bootstraps, seed=_deterministic_seed(stem + str(h)))
            row_cells.append(fmt_ci(point, lo, hi))
        lines.append(f"{disp} & " + " & ".join(row_cells) + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table*}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
