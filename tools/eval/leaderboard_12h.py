#!/usr/bin/env python3
"""Build 12h leaderboard bar chart + LaTeX table from metrics/eval_12h_2020_pre_ep10/*.json.

Inputs:
  - --metrics-dir metrics/eval_12h_2020_pre_ep10       (neural model JSONs)
  - --numeric-dir metrics/eval_12h_2020_numeric (bilinear/semi_lagrangian/... JSONs)

Outputs:
  - figs/fig_leaderboard_12h.{png,pdf}
  - paper/tab_main_12h.tex

Each JSON's grand RMSE = mean over τ ∈ {1..11} of mean over channels of
``rmse_norm_<ch>``. Numerical JSONs have ``per_hour`` block (legacy 6h schema).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


# Display labels (paper_tag preferred over filename stem)
TAG_TO_DISPLAY = {
    "WeatherBridge_PP3_3yr_12h": ("WeatherBridge", 14.3, "2017-19"),
    "DC-AE_NoSkip_3yr_12h_fibo": ("DC-AE NoSkip (3yr)", 14.4, "2017-19"),
    "DC-AE_Skip_3yr_12h_fibo": ("DC-AE Skip (3yr)", 14.4, "2017-19"),
    "FuXi_3yr_12h_fibo": ("SwinV2", 8.0, "2017-19"),
    "ModAFNO_3yr_12h_fibo": ("ModAFNO", 151.7, "2017-19"),
    "SDyff_3yr_12h_fibo": ("S-DYff", 85.5, "2017-19"),
    "ATM-VFI_3yr_12h_fibo": ("ATM-VFI", 11.7, "2017-19"),
}


def _avg_per_tau_norm(per_tau: dict, method: str, channels: List[str]) -> Dict[int, float]:
    out: Dict[int, float] = {}
    for tau_s, by_m in per_tau.items():
        block = by_m.get(method, {})
        vals = [block.get(f"rmse_norm_{c}") for c in channels]
        vals = [v for v in vals if v is not None]
        if vals:
            out[int(tau_s)] = float(np.mean(vals))
    return out


def _summary_for_model(path: Path, seen=(1, 2, 3, 5, 7, 9, 10, 11),
                       unseen=(4, 6, 8)) -> Dict:
    d = json.load(open(path))
    channels = d["channel_names"]
    per_tau_model = _avg_per_tau_norm(d["per_tau"], "model", channels)
    per_tau_bil = _avg_per_tau_norm(d["per_tau"], "bilinear", channels)
    per_tau_bic = _avg_per_tau_norm(d["per_tau"], "bicubic", channels)
    seen_vals = [per_tau_model[t] for t in seen if t in per_tau_model]
    unseen_vals = [per_tau_model[t] for t in unseen if t in per_tau_model]
    return {
        "name": path.stem,
        "per_tau": per_tau_model,
        "per_tau_bil": per_tau_bil,
        "per_tau_bic": per_tau_bic,
        "rmse_tau3": per_tau_model.get(3),
        "rmse_tau6": per_tau_model.get(6),
        "rmse_tau9": per_tau_model.get(9),
        "rmse_avg_all": float(np.mean(list(per_tau_model.values()))),
        "rmse_avg_seen": float(np.mean(seen_vals)) if seen_vals else None,
        "rmse_avg_unseen": float(np.mean(unseen_vals)) if unseen_vals else None,
    }


def _summary_for_numerical(path: Path,
                           seen=(1, 2, 3, 5, 7, 9, 10, 11),
                           unseen=(4, 6, 8)) -> Dict:
    d = json.load(open(path))
    channels = d["channel_names"]
    # Numerical JSON uses ``per_hour`` schema with method sub-blocks, and the
    # channel RMSE keys are named ``rmse_<ch>`` (no ``_norm_`` prefix). The
    # ``model`` sub-block carries the numerical method's own prediction.
    per_tau_src = d.get("per_tau") or d.get("per_hour", {})
    per_tau_model: Dict[int, float] = {}
    for tau_s, by_m in per_tau_src.items():
        block = by_m.get("model", by_m)
        vals = []
        for c in channels:
            v = block.get(f"rmse_norm_{c}", block.get(f"rmse_{c}"))
            if v is not None:
                vals.append(float(v))
        if vals:
            per_tau_model[int(tau_s)] = float(np.mean(vals))
    seen_vals = [per_tau_model[t] for t in seen if t in per_tau_model]
    unseen_vals = [per_tau_model[t] for t in unseen if t in per_tau_model]
    return {
        "name": path.stem,
        "per_tau": per_tau_model,
        "rmse_tau3": per_tau_model.get(3),
        "rmse_tau6": per_tau_model.get(6),
        "rmse_tau9": per_tau_model.get(9),
        "rmse_avg_all": float(np.mean(list(per_tau_model.values()))) if per_tau_model else None,
        "rmse_avg_seen": float(np.mean(seen_vals)) if seen_vals else None,
        "rmse_avg_unseen": float(np.mean(unseen_vals)) if unseen_vals else None,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics-dir", default="metrics/eval_12h_2020_pre_ep10")
    ap.add_argument("--numeric-dir", default="metrics/eval_12h_2020_numeric")
    ap.add_argument("--figs-out", default="figs/fig_leaderboard_12h")
    ap.add_argument("--tex-out", default="paper/tab_main_12h.tex")
    args = ap.parse_args()

    models = []
    excluded_stems = {"WeatherDCAE_NoSkip_6yr_12h"}
    for p in sorted(Path(args.metrics_dir).glob("*.json")):
        if p.stem in excluded_stems:
            continue
        try:
            s = _summary_for_model(p)
            models.append(s)
        except Exception as e:
            print(f"  skip {p.name}: {e}")

    numerics = []
    for p in sorted(Path(args.numeric_dir).glob("*.json")):
        try:
            s = _summary_for_numerical(p)
            numerics.append(s)
        except Exception as e:
            print(f"  skip {p.name}: {e}")

    # Sort: best (lowest grand RMSE) first.
    models.sort(key=lambda d: d["rmse_avg_all"])
    numerics.sort(key=lambda d: d["rmse_avg_all"] if d["rmse_avg_all"] else 1e9)

    # ───────────────── bar chart ─────────────────
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 5))
    labels = []
    vals = []
    colors = []
    for m in models:
        disp = TAG_TO_DISPLAY.get(m["name"], (m["name"], 0.0, "?"))
        labels.append(disp[0])
        vals.append(m["rmse_avg_all"])
        colors.append("tab:blue")
    for n in numerics:
        labels.append(n["name"])
        vals.append(n["rmse_avg_all"] if n["rmse_avg_all"] else float("nan"))
        colors.append("0.65")

    ypos = np.arange(len(labels))
    bars = ax.barh(ypos, vals, color=colors, edgecolor="0.2", height=0.7)
    ax.set_yticks(ypos)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Normalised RMSE (avg over τ∈{1..11}, 24 channels, 2020 4d/mo)")
    ax.set_title("12 h interpolation leaderboard — 2020 ERA5 0.5°")
    ax.invert_yaxis()  # best at top
    ax.axvline(min(vals), color="red", linewidth=0.6, alpha=0.4)
    for bar, v in zip(bars, vals):
        if not np.isnan(v):
            ax.text(v + 0.05, bar.get_y() + bar.get_height() / 2, f"{v:.3f}",
                    va="center", fontsize=8)
    plt.tight_layout()
    out_png = Path(args.figs_out + ".png")
    out_pdf = Path(args.figs_out + ".pdf")
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=150)
    plt.savefig(out_pdf)
    plt.close()
    print(f"saved {out_png}")
    print(f"saved {out_pdf}")

    # ───────────────── LaTeX table ─────────────────
    tex_path = Path(args.tex_out)
    tex_path.parent.mkdir(parents=True, exist_ok=True)

    def _fmt(v, prec=3):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return "--"
        return f"{v:.{prec}f}"

    best_all = min(m["rmse_avg_all"] for m in models)
    rows = []
    for m in models:
        disp, params_m, train_yrs = TAG_TO_DISPLAY.get(
            m["name"], (m["name"], 0.0, "?")
        )
        seen_v = m["rmse_avg_seen"]
        unseen_v = m["rmse_avg_unseen"]
        delta = (unseen_v / seen_v - 1.0) * 100 if seen_v and unseen_v else None
        avg_all = m["rmse_avg_all"]
        avg_all_str = _fmt(avg_all)
        if abs(avg_all - best_all) < 1e-9:
            avg_all_str = r"\textbf{" + avg_all_str + "}"
        row = (
            f"{disp} & {params_m:.1f} & {train_yrs} & "
            f"{_fmt(m['rmse_tau3'])} & {_fmt(m['rmse_tau6'])} & "
            f"{_fmt(m['rmse_tau9'])} & {_fmt(m['rmse_avg_seen'])} & "
            f"{_fmt(m['rmse_avg_unseen'])} & {_fmt(delta, 1)}\\% \\\\"
        )
        rows.append(row)

    num_rows = []
    for n in numerics:
        seen_v = n["rmse_avg_seen"]
        unseen_v = n["rmse_avg_unseen"]
        delta = (unseen_v / seen_v - 1.0) * 100 if seen_v and unseen_v else None
        row = (
            f"\\textit{{{n['name']}}} & -- & -- & "
            f"{_fmt(n['rmse_tau3'])} & {_fmt(n['rmse_tau6'])} & "
            f"{_fmt(n['rmse_tau9'])} & {_fmt(n['rmse_avg_seen'])} & "
            f"{_fmt(n['rmse_avg_unseen'])} & {_fmt(delta, 1)}\\% \\\\"
        )
        num_rows.append(row)

    tex = (
        r"\begin{table}[t]" + "\n"
        r"\centering" + "\n"
        r"\caption{12\,h interpolation leaderboard on 2020 ERA5 0.5\textdegree{} (4 days/month, 24 channels). "
        r"Avg seen / unseen split: $\tau{\in}\{1,2,3,5,7,9,10,11\}$ seen, $\tau{\in}\{4,6,8\}$ held out. "
        r"$\Delta\%$ is unseen vs seen relative gap. Lowest RMSE in \textbf{bold}.}" + "\n"
        r"\label{tab:main-12h}" + "\n"
        r"\small" + "\n"
        r"\begin{tabular}{lrrrrrrrr}" + "\n"
        r"\toprule" + "\n"
        r"Model & Params (M) & Train yrs & RMSE@$\tau$=3 & RMSE@$\tau$=6 & RMSE@$\tau$=9 & RMSE seen & RMSE unseen & $\Delta\%$ \\" + "\n"
        r"\midrule" + "\n"
        + "\n".join(rows) + "\n"
        r"\midrule" + "\n"
        r"\multicolumn{9}{l}{\textit{Numerical baselines}} \\" + "\n"
        + "\n".join(num_rows) + "\n"
        r"\bottomrule" + "\n"
        r"\end{tabular}" + "\n"
        r"\end{table}" + "\n"
    )
    tex_path.write_text(tex)
    print(f"saved {tex_path}")

    # Console summary
    print("\n=== 12 h leaderboard (avg over τ ∈ {1..11}, normalised RMSE) ===")
    for m in models:
        disp = TAG_TO_DISPLAY.get(m["name"], (m["name"],))[0]
        print(f"  {disp:30s}  RMSE_avg={m['rmse_avg_all']:.4f}  seen={m['rmse_avg_seen']:.4f}  unseen={m['rmse_avg_unseen']:.4f}")
    print("\n=== Numerical baselines ===")
    for n in numerics:
        v = n["rmse_avg_all"]
        if v is not None:
            print(f"  {n['name']:30s}  RMSE_avg={v:.4f}  seen={n['rmse_avg_seen']:.4f}  unseen={n['rmse_avg_unseen']:.4f}")
        else:
            print(f"  {n['name']:30s}  (incompatible JSON schema)")


if __name__ == "__main__":
    main()
