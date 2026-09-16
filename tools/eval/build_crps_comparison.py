#!/usr/bin/env python3
"""Build paper/tab_crps_comparison.tex from CRPS JSONs + RMSE baselines.

Reads:
  metrics/crps_0p5_2020/corrdiff_fm_weatherdcae.json   (6h CorrDiff ens N=16 CRPS)
  metrics/crps_0p5_2020/sdyff_24ch_6yr.json            (6h S-DYff ens CRPS)
  metrics/crps_12h_2020/corrdiff_fm_weatherdcae_12h_6yr.json  (12h CorrDiff CRPS)
  metrics/eval_6h_2020_paper_leaderboard/bilinear_24ch.json        (6h bilinear det)
  metrics/eval_6h_2020_paper_leaderboard/weatherdcae_noskip_24ch_6yr_ep8.json   (6h det WeatherDCAE)
  metrics/eval_12h_2020_ep10/WeatherDCAE_NoSkip_6yr_12h.json        (12h det WeatherDCAE)
  metrics/eval_12h_2020_numeric/bilinear.json                       (12h det bilinear)
  metrics/eval_12h_2020_ep10/corrdiff_fm_weatherdcae_12h_6yr_ens16.json (12h CorrDiff ens RMSE)

Outputs:
  paper/tab_crps_comparison.tex with per-tau avg over 24 channels:
    - Bilinear 6h / 12h   (RMSE only — deterministic)
    - WeatherDCAE NoSkip det 6h / 12h (RMSE only — deterministic)
    - CorrDiff/WeatherDCAE ensemble N=16 6h / 12h (ensemble-mean RMSE + CRPS)
    - S-DYff ensemble N=16 (ensemble-mean RMSE + CRPS)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def load_norm_avg_per_tau(path, key_options=("rmse_model_norm", "crps_norm"), n_tau=5):
    """Load schema where the file stores ``rmse_model_norm: [[ch_vals] * n_tau]``.

    Returns a list of length n_tau with channel-averaged norm values.
    """
    d = json.load(open(path))
    for k in key_options:
        if k in d:
            return [float(np.mean(d[k][t])) for t in range(n_tau)]
    raise KeyError(f"None of {key_options} in {path}")


def load_12h_per_tau_avg_rmse(path, taus=(2, 3, 5, 8, 11), block="model"):
    """Load 12h schema where ``per_tau[str(tau)][block] = {rmse_norm_<ch>: val, ...}``.

    Returns a list of length len(taus) with channel-averaged norm RMSE.
    block can be "model" or "bilinear".
    """
    d = json.load(open(path))
    per_tau = d.get("per_tau") or d.get("per_hour")
    if per_tau is None:
        raise KeyError(f"no per_tau / per_hour in {path}")
    out = []
    for tau in taus:
        cell = per_tau.get(str(tau), {}).get(block, {})
        norms = [v for k, v in cell.items() if k.startswith("rmse_norm_")]
        out.append(float(np.mean(norms)) if norms else float("nan"))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--repo",
        default=str(Path(__file__).resolve().parents[2]),
        help="Repository root (defaults to the checkout containing this script)",
    )
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    repo = Path(args.repo)
    out_path = Path(args.out) if args.out else repo / "paper/tab_crps_comparison.tex"

    # 6h sources
    bilinear_rmse_path = repo / "metrics/eval_6h_2020_paper_leaderboard/bilinear_24ch.json"
    wdcae_rmse_path = repo / "metrics/eval_6h_2020_paper_leaderboard/weatherdcae_noskip_24ch_6yr_ep8.json"
    corrdiff_rmse_path = repo / "metrics/eval_6h_2020_paper_leaderboard/corrdiff_fm_weatherdcae_24ch_3yr_ep10.json"
    corrdiff_crps_path = repo / "metrics/crps_0p5_2020/corrdiff_fm_weatherdcae.json"
    # Use the latest N=16 ensemble S-DYff metrics (see context: sdyff_24ch_6yr_ens16.json)
    sdyff_rmse_path_ens16 = repo / "metrics/eval_6h_2020_paper_leaderboard/sdyff_24ch_6yr_ens16.json"
    sdyff_crps_path_ens16 = repo / "metrics/crps_0p5_2020/sdyff_24ch_6yr_ens16.json"
    # Fallback to N=8 if N=16 not yet on this machine.
    sdyff_rmse_path = sdyff_rmse_path_ens16 if sdyff_rmse_path_ens16.exists() else \
        repo / "metrics/eval_6h_2020_paper_leaderboard/sdyff_24ch_6yr_ens8.json"
    sdyff_crps_path = sdyff_crps_path_ens16 if sdyff_crps_path_ens16.exists() else \
        repo / "metrics/crps_0p5_2020/sdyff_24ch_6yr.json"
    sdyff_label = "S-DYff 6yr (ens N=16 MC-dropout)" if sdyff_rmse_path_ens16.exists() \
        else "S-DYff 6yr (ens N=8 MC-dropout)"

    # 12h sources — note the WeatherDCAE_NoSkip_6yr_12h.json includes a "bilinear"
    # block alongside the "model" block with rmse_norm_*, so we use it for both.
    wdcae_12h_path = repo / "metrics/eval_12h_2020_ep10/WeatherDCAE_NoSkip_6yr_12h.json"
    bilinear_12h_path = wdcae_12h_path  # same file, different block
    corrdiff_12h_rmse_path = repo / "metrics/eval_12h_2020_ep10/corrdiff_fm_weatherdcae_12h_6yr_ens16.json"
    corrdiff_12h_crps_path = repo / "metrics/crps_12h_2020/corrdiff_fm_weatherdcae_12h_6yr.json"

    rows = []  # 6h section: (name, rmse[5], crps[5], kind)

    # === 6h rows ===
    if bilinear_rmse_path.exists():
        bil = load_norm_avg_per_tau(bilinear_rmse_path, ("rmse_model_norm", "rmse_bilinear_norm"))
        rows.append(("Bilinear 6h", bil, [None] * 5, "det"))
    if wdcae_rmse_path.exists():
        wd = load_norm_avg_per_tau(wdcae_rmse_path)
        rows.append(("WeatherDCAE NoSkip 6h (det)", wd, [None] * 5, "det"))
    if corrdiff_rmse_path.exists() and corrdiff_crps_path.exists():
        rmse = load_norm_avg_per_tau(corrdiff_rmse_path)
        crps = load_norm_avg_per_tau(corrdiff_crps_path, ("crps_norm",))
        rows.append(("CorrDiff/FM/WeatherDCAE 6h (ens N=16)", rmse, crps, "ens"))
    else:
        rows.append(("CorrDiff/FM/WeatherDCAE 6h (ens N=16)", [None]*5, [None]*5, "pending"))
    if sdyff_rmse_path.exists() and sdyff_crps_path.exists():
        rmse = load_norm_avg_per_tau(sdyff_rmse_path)
        crps = load_norm_avg_per_tau(sdyff_crps_path, ("crps_norm",))
        rows.append((sdyff_label, rmse, crps, "ens"))
    else:
        rows.append((sdyff_label, [None]*5, [None]*5, "pending"))

    # === 12h rows (representative taus 2,4,6,8,10 to match 5-column layout) ===
    TAUS_12H = (2, 4, 6, 8, 10)
    rows_12h = []
    if bilinear_12h_path.exists():
        # bilinear 12h uses ``per_hour[tau]["bilinear"]`` structure
        bil12 = load_12h_per_tau_avg_rmse(bilinear_12h_path, taus=TAUS_12H, block="bilinear")
        rows_12h.append(("Bilinear 12h", bil12, [None]*5, "det"))
    if wdcae_12h_path.exists():
        wd12 = load_12h_per_tau_avg_rmse(wdcae_12h_path, taus=TAUS_12H, block="model")
        rows_12h.append(("WeatherDCAE NoSkip 12h (det)", wd12, [None]*5, "det"))
    if corrdiff_12h_rmse_path.exists() and corrdiff_12h_crps_path.exists():
        # 12h ens RMSE/CRPS use eval_ensemble_crps.py schema: list[n_tau-1][24].
        # n_tau for 12h is 11 (taus 1..11). Index in list = tau-1.
        rmse12_all = load_norm_avg_per_tau(corrdiff_12h_rmse_path, ("rmse_model_norm",), n_tau=11)
        crps12_all = load_norm_avg_per_tau(corrdiff_12h_crps_path, ("crps_norm",), n_tau=11)
        rmse12 = [rmse12_all[t-1] for t in TAUS_12H]
        crps12 = [crps12_all[t-1] for t in TAUS_12H]
        rows_12h.append(("CorrDiff/FM/WeatherDCAE 12h (ens N=16)", rmse12, crps12, "ens"))
    else:
        rows_12h.append(("CorrDiff/FM/WeatherDCAE 12h (ens N=16)", [None]*5, [None]*5, "pending"))

    # Render LaTeX
    def fmt(v):
        return "--" if v is None else f"{v:.4f}"

    lines = []
    lines.append("% Auto-generated by tools/eval/build_crps_comparison.py")
    lines.append("% Source: metrics/eval_6h_2020_paper_leaderboard/*.json + metrics/crps_0p5_2020/*.json")
    lines.append("% +     metrics/eval_12h_2020_ep10/*.json + metrics/crps_12h_2020/*.json")
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{\textbf{Ensemble RMSE and lat-weighted CRPS at 6\,h and 12\,h interpolation, $0.5^\circ$, 2020 test (normalised scale).}")
    lines.append(r"RMSE shown is on the ensemble mean for stochastic models, equal to per-sample RMSE for")
    lines.append(r"deterministic baselines. CRPS uses the fair-Hersbach estimator, lat-weighted; lower is better")
    lines.append(r"for both metrics. Ensemble size $N$ noted in parentheses. 6\,h section shows $\tau{\in}\{1{,}\dots{,}5\}$h;")
    lines.append(r"12\,h section shows $\tau{\in}\{2{,}4{,}6{,}8{,}10\}$h.}")
    lines.append(r"\label{tab:crps-comparison}")
    lines.append(r"\small")
    lines.append(r"\setlength{\tabcolsep}{4pt}")
    lines.append(r"\begin{tabular}{lrrrrr@{\hspace{8pt}}rrrrr}")
    lines.append(r"\toprule")
    lines.append(r"& \multicolumn{5}{c}{\textbf{Ens.\ RMSE$_\sigma$}} & \multicolumn{5}{c}{\textbf{CRPS$_\sigma$}} \\")
    lines.append(r"\cmidrule(lr){2-6} \cmidrule(lr){7-11}")
    lines.append(r"\textbf{Method} & $\tau{=}1$ & $\tau{=}2$ & $\tau{=}3$ & $\tau{=}4$ & $\tau{=}5$ & $\tau{=}1$ & $\tau{=}2$ & $\tau{=}3$ & $\tau{=}4$ & $\tau{=}5$ \\")
    lines.append(r"\midrule")
    lines.append(r"\multicolumn{11}{l}{\textit{6\,h interpolation window ($\tau$ in hours, $\tau{\in}\{1,\dots,5\}$)}} \\")
    lines.append(r"\midrule")
    for name, rmse, crps, kind in rows:
        rmse_cells = " & ".join(fmt(v) for v in rmse)
        crps_cells = " & ".join(fmt(v) for v in crps)
        lines.append(f"{name} & {rmse_cells} & {crps_cells} \\\\")
    lines.append(r"\midrule")
    lines.append(r"\multicolumn{11}{l}{\textit{12\,h interpolation window ($\tau$ in hours, $\tau{\in}\{2,4,6,8,10\}$)}} \\")
    lines.append(r"\midrule")
    for name, rmse, crps, kind in rows_12h:
        rmse_cells = " & ".join(fmt(v) for v in rmse)
        crps_cells = " & ".join(fmt(v) for v in crps)
        lines.append(f"{name} & {rmse_cells} & {crps_cells} \\\\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n")
    print(f"wrote {out_path}")
    print()
    print("6h rows:")
    for name, rmse, crps, kind in rows:
        print(f"  [{kind:7s}] {name}: rmse={[None if v is None else round(v,4) for v in rmse]}, crps={[None if v is None else round(v,4) for v in crps]}")
    print("12h rows:")
    for name, rmse, crps, kind in rows_12h:
        print(f"  [{kind:7s}] {name}: rmse={[None if v is None else round(v,4) for v in rmse]}, crps={[None if v is None else round(v,4) for v in crps]}")


if __name__ == "__main__":
    main()
