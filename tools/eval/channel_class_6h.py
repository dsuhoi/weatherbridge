#!/usr/bin/env python3
"""Per-channel-class breakdown of the 6h headline (24ch) leaderboard.

Mirrors tools/eval/channel_class_12h.py but for the 6h ``fast`` JSONs at
metrics/eval_6h_2020_paper_leaderboard.  24ch JSONs use ``channels`` + ``tau_hours`` +
``rmse_model_norm`` (list-of-lists OR dict-of-lists).

The 24-channel set in 6h JSONs has 4 surface channels (t2m, u10, v10, mslp)
and 20 pressure-level channels (T/U/V/Q/Z @ 1000,925,850,700).

Outputs:
  - metrics/eval_6h_2020_paper_leaderboard/channel_class_breakdown_24ch.json
  - paper/tab_channel_class_6h.tex
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

CHANNEL_CLASSES: Dict[str, List[str]] = {
    "T_PL":     ["T1000", "T925", "T850", "T700"],
    "U_PL":     ["U1000", "U925", "U850", "U700"],
    "V_PL":     ["V1000", "V925", "V850", "V700"],
    "Q_PL":     ["Q1000", "Q925", "Q850", "Q700"],
    "Z_PL":     ["Z1000", "Z925", "Z850", "Z700"],
    "Surface":  ["t2m", "u10", "v10", "mslp"],
}

CLASS_DISPLAY = {
    "T_PL":    r"$T$ (PL)",
    "U_PL":    r"$u$ (PL)",
    "V_PL":    r"$v$ (PL)",
    "Q_PL":    r"$q$ (PL)",
    "Z_PL":    r"$z$ (PL)",
    "Surface": "Surface",
}

TAG_DISPLAY = {
    "weatherdcae_noskip_24ch_6yr_ep10": "WeatherDCAE NoSkip (6yr, ep10)",
    "atm_vfi_24ch_6yr":                 "ATM-VFI 24ch (6yr)",
    "atm_vfi_v2_static_24ch_6yr":       "ATM-VFI v2 static (6yr)",
    "atm_vfi_v2_135only_rmse":          "ATM-VFI v2 (135 only)",
    "corrdiff_fm_bilinear_24ch":        "CorrDiff-FM / Bilinear",
    "corrdiff_fm_dcae_skip_24ch":       "CorrDiff-FM / DC-AE Skip",
}


def _extract_rows(d: dict) -> Tuple[np.ndarray, List[str], List[int]]:
    """Return (n_tau, 24) matrix and (channels, used_taus) for a 24ch JSON."""
    channels = d.get("channels") or []
    tau_hours = d.get("tau_hours") or [1, 2, 3, 4, 5]
    rmse = d.get("rmse_model_norm")
    if not rmse:
        return np.empty((0, 0)), channels, []
    rows = []
    used_taus = []
    if isinstance(rmse, list):
        for i, tau in enumerate(tau_hours):
            row = rmse[i]
            if any(v is None for v in row):
                continue
            rows.append(row)
            used_taus.append(tau)
    elif isinstance(rmse, dict):
        for tau in sorted(int(k) for k in rmse.keys()):
            row = rmse[str(tau)]
            if any(v is None for v in row):
                continue
            rows.append(row)
            used_taus.append(tau)
    return np.array(rows, dtype=np.float64), channels, used_taus


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics-dir", default="metrics/eval_6h_2020_paper_leaderboard")
    ap.add_argument("--out-json", default="metrics/eval_6h_2020_paper_leaderboard/channel_class_breakdown_24ch.json")
    ap.add_argument("--out-tex", default="paper/tab_channel_class_6h.tex")
    args = ap.parse_args()

    metrics_dir = Path(args.metrics_dir)

    paths = []
    for p in sorted(metrics_dir.glob("*.json")):
        if p.name.startswith("bootstrap_ci_") or p.name.startswith("channel_class_breakdown"):
            continue
        try:
            d = json.load(open(p))
        except Exception:
            continue
        if "channels" in d and "tau_hours" in d:
            paths.append(p)

    records = []
    for p in paths:
        d = json.load(open(p))
        M, channels, used = _extract_rows(d)
        if M.size == 0:
            continue
        ch_idx = {c: i for i, c in enumerate(channels)}
        class_to_vals: Dict[str, List[float]] = {cls: [] for cls in CHANNEL_CLASSES}
        for cls, ch_list in CHANNEL_CLASSES.items():
            for c in ch_list:
                if c not in ch_idx:
                    continue
                col = M[:, ch_idx[c]]
                class_to_vals[cls].extend(float(v) for v in col)
        class_means = {cls: float(np.mean(vs)) if vs else None
                       for cls, vs in class_to_vals.items()}
        class_stds = {cls: float(np.std(vs, ddof=1)) if len(vs) > 1 else None
                      for cls, vs in class_to_vals.items()}
        records.append({
            "model_stem": p.stem,
            "display_name": TAG_DISPLAY.get(p.stem, p.stem),
            "per_class_mean": class_means,
            "per_class_std": class_stds,
            "n_tau": M.shape[0],
            "n_channels": len(channels),
        })

    records.sort(key=lambda r: np.mean([v for v in r["per_class_mean"].values()
                                        if v is not None]))

    payload = {
        "schema_version": 1,
        "channel_classes": CHANNEL_CLASSES,
        "metric": "mean(rmse_norm) over τ∈{1..5} per channel-class (24ch headline schema)",
        "records": records,
    }
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(payload, indent=2))
    print(f"saved {args.out_json}")

    # LaTeX table
    class_order = list(CHANNEL_CLASSES.keys())
    best_per_class: Dict[str, float] = {}
    for cls in class_order:
        vals = [r["per_class_mean"].get(cls) for r in records]
        vals = [v for v in vals if v is not None]
        if vals:
            best_per_class[cls] = min(vals)

    lines = []
    lines.append("% Auto-generated by tools/eval/channel_class_6h.py")
    lines.append("% 24-channel 6h headline; mirrors paper/tab_channel_class_12h.tex.")
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{Per-channel-class mean RMSE$_{\mathrm{norm}}$ for the "
                 r"6\,h \emph{24-channel headline} leaderboard on 2020 ERA5 "
                 r"0.5\textdegree (averaged over $\tau \in \{1..5\}$). PL classes "
                 r"group $(z=1000, 925, 850, 700)$ levels (Z geopotential, "
                 r"$T$ temperature, $u/v$ winds, $q$ specific humidity); Surface "
                 r"groups $t2m, u10, v10, mslp$. Class-best in \textbf{bold}.}")
    lines.append(r"\label{tab:channel_class_6h}")
    lines.append(r"\small")
    lines.append(r"\setlength{\tabcolsep}{5pt}")
    lines.append(r"\begin{tabular}{l" + "r" * (len(class_order) + 1) + "}")
    lines.append(r"\toprule")
    head_cells = " & ".join(f"\\textbf{{{CLASS_DISPLAY[c]}}}" for c in class_order)
    lines.append(r"\textbf{Model} & " + head_cells + r" & \textbf{Mean} \\")
    lines.append(r"\midrule")
    for r in records:
        cells = []
        per_class = r["per_class_mean"]
        all_vals = [v for v in per_class.values() if v is not None]
        for cls in class_order:
            v = per_class.get(cls)
            if v is None:
                cells.append("--")
                continue
            s = f"{v:.3f}"
            if abs(v - best_per_class.get(cls, 1e9)) < 1e-9:
                s = r"\textbf{" + s + "}"
            cells.append(s)
        mean_str = f"{np.mean(all_vals):.3f}" if all_vals else "--"
        lines.append(f"{r['display_name']} & " + " & ".join(cells)
                     + f" & {mean_str} \\\\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table*}")

    Path(args.out_tex).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_tex).write_text("\n".join(lines) + "\n")
    print(f"saved {args.out_tex}")

    print("\n=== Class-best (lowest mean RMSE_norm per class, 6h) ===")
    for cls in class_order:
        best_rec = min((r for r in records if r["per_class_mean"].get(cls) is not None),
                       key=lambda r: r["per_class_mean"][cls])
        v = best_rec["per_class_mean"][cls]
        print(f"  {cls:8s}  {best_rec['display_name']:34s}  RMSE_norm={v:.4f}")


if __name__ == "__main__":
    main()
