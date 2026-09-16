#!/usr/bin/env python3
"""Compile leaderboard + visualizations for all model evals on 2020.

Scans metrics/eval_<year>/*.json and produces:
  - leaderboard.md (overall + per-channel AVG)
  - rmse_per_hour_per_channel.png (line plots: model RMSE vs hour for each of 5 chans)
  - delta_pct_bar.png (bar chart Δ% vs bilinear, per-channel summary)
  - heatmap_per_model_per_channel.png (heatmap models × channels)
  - (optional) energy_spectra.png if --energy-spectra-npz given

Usage:
    python tools/plot_leaderboard.py --metrics-dir metrics/eval_2020 \
        --out-dir docs/results/leaderboard_2020
"""
from __future__ import annotations
import argparse, json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

CHANNELS = [
    ("T", "rmse_temperature"),
    ("U", "rmse_u_component_of_wind"),
    ("V", "rmse_v_component_of_wind"),
    ("Q", "rmse_specific_humidity"),
    ("Z", "rmse_geopotential"),
]
CHAN_FULL_NAMES = {
    "T": "temperature (K)",
    "U": "u-wind (m/s)",
    "V": "v-wind (m/s)",
    "Q": "specific humidity (kg/kg)",
    "Z": "geopotential (m²/s²)",
}


def collect(metrics_dir: Path):
    """Returns dict: {name: per_hour_dict}."""
    runs = {}
    for f in sorted(metrics_dir.glob("*.json")):
        name = f.stem
        try:
            data = json.load(open(f))
        except Exception as e:
            print(f"[warn] skip {name}: {e}")
            continue
        # Skip bilinear-only JSON (no model channel)
        if "per_hour" not in data:
            continue
        runs[name] = data["per_hour"]
    return runs


def get(per_hour: dict, h: str, source: str, key: str) -> float | None:
    """Safe getter; source ∈ {'bilinear','bicubic','model'}, key e.g. 'rmse'."""
    return per_hour.get(h, {}).get(source, {}).get(key)


def hour_keys(per_hour: dict) -> list[str]:
    return sorted(per_hour.keys(), key=int)


def avg_metric(per_hour: dict, source: str, key: str, hours: list[str]) -> float:
    vals = [get(per_hour, h, source, key) for h in hours]
    vals = [v for v in vals if v is not None]
    return float(np.mean(vals)) if vals else float("nan")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--metrics-dir", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--include-h6", action="store_true",
                   help="include hour 6 in AVG (endpoint of 6h gap; usually excluded)")
    p.add_argument("--strict-interior", action="store_true", default=False,
                   help="Exclude τ=1 and τ=(delta-1) — keep only deep interpolation hours. "
                        "For 6h gap this means {2,3,4}; for 12h means {2..10}. "
                        "Endpoints (h=0, h=delta) are always excluded.")
    p.add_argument("--energy-spectra-npz", default=None)
    args = p.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    runs = collect(Path(args.metrics_dir))
    if not runs:
        print("[err] no eval JSONs found in", args.metrics_dir)
        return
    print(f"[info] found {len(runs)} model runs: {list(runs)}")

    # Sanity: pick first run to get hour set
    sample = next(iter(runs.values()))
    all_hours = hour_keys(sample)

    # WARNING: each run computes its own bilinear/bicubic on the test windows it
    # sampled. With different --samples-per-date, the bilinear baseline values
    # differ between runs. We always compare a run's MODEL vs its OWN bilinear
    # (the Δ% is intra-run consistent), but the absolute RMSE numbers across runs
    # are NOT directly comparable.
    # Endpoints (h=0, h=delta) always excluded — those are exact bilinear matches.
    # Default: keep all middle hours (1..delta-1). Strict-interior: drop h=1 and h=delta-1 too.
    last = max(int(h) for h in all_hours)
    if args.include_h6:
        avg_hours = [h for h in all_hours if 1 <= int(h)]
    elif args.strict_interior:
        # Deep-interpolation only — drop near-endpoint hours where bilinear is naturally good
        avg_hours = [h for h in all_hours if 2 <= int(h) <= last - 2]
        if not avg_hours:
            print(f"[warn] strict-interior empty for hours {all_hours}, falling back to (1..last-1)")
            avg_hours = [h for h in all_hours if 1 <= int(h) < last]
    else:
        avg_hours = [h for h in all_hours if 1 <= int(h) < last]
    print(f"[info] hours present: {all_hours}; AVG window: {avg_hours}")

    # ─── 1. Leaderboard table ────────────────────────────────────────────
    # IMPORTANT: each run's `bilinear` field was computed on its own sampled
    # windows; absolute RMSE across runs is NOT directly comparable. We compare
    # each MODEL vs its OWN intra-run BILINEAR (Δ% is sample-invariant).
    table_rows = []
    for name, per_hour in runs.items():
        bi_avg = avg_metric(per_hour, "bilinear", "rmse", avg_hours)
        mo_avg = avg_metric(per_hour, "model", "rmse", avg_hours)
        d = (mo_avg - bi_avg) / bi_avg * 100
        per_ch = {}
        for sh, k in CHANNELS:
            ch_bi = avg_metric(per_hour, "bilinear", k, avg_hours)
            ch_mo = avg_metric(per_hour, "model", k, avg_hours)
            per_ch[sh] = (ch_mo, (ch_mo - ch_bi) / ch_bi * 100 if ch_bi > 0 else 0.0, ch_bi)
        table_rows.append((name, mo_avg, d, per_ch, bi_avg))

    md = [f"# Leaderboard ({Path(args.metrics_dir).name})\n",
          f"AVG hours: {avg_hours} (n={len(avg_hours)}).",
          "Each row: RMSE compared against THIS run's own bilinear baseline (Δ% is intra-run, sample-invariant).",
          "**Absolute RMSE across rows is NOT directly comparable** if runs used different `--samples-per-date`.\n",
          "## Per-model intra-run Δ% vs bilinear\n",
          "| Model | bilin RMSE | model RMSE | Δ% | T Δ% | U Δ% | V Δ% | Q Δ% | Z Δ% |",
          "|---|---|---|---|---|---|---|---|---|"]
    # Sort by Δ% (best first)
    table_rows.sort(key=lambda r: r[2])
    for name, mo_avg, d, per_ch, bi_avg in table_rows:
        deltas_str = " | ".join(f"{per_ch[sh][1]:+.1f}%" for sh in ["T", "U", "V", "Q", "Z"])
        row = f"| {name} | {bi_avg:.4f} | {mo_avg:.4f} | **{d:+.1f}%** | {deltas_str} |"
        md.append(row)

    md_path = out / "leaderboard.md"
    md_path.write_text("\n".join(md))
    print(f"[ok] wrote {md_path}")

    # ─── 2. Line plots: RMSE vs hour per channel ─────────────────────────
    fig, axes = plt.subplots(1, len(CHANNELS), figsize=(4 * len(CHANNELS), 3.6), constrained_layout=True)
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(runs) + 2, 10)))
    for ci, (sh, k) in enumerate(CHANNELS):
        ax = axes[ci]
        h_int = [int(h) for h in all_hours]
        bi = [get(sample, h, "bilinear", k) for h in all_hours]
        ax.plot(h_int, bi, "k--", marker="s", lw=2, label="bilinear")
        for run_i, (name, per_hour) in enumerate(runs.items()):
            mo = [get(per_hour, h, "model", k) for h in all_hours]
            ax.plot(h_int, mo, "-", marker="o", color=colors[run_i + 2], label=name[:18])
        ax.set_title(f"{sh} — {CHAN_FULL_NAMES[sh]}")
        ax.set_xlabel("hour τ")
        ax.set_ylabel("RMSE")
        ax.grid(alpha=0.3)
        if ci == 0:
            ax.legend(fontsize=7, loc="upper center", ncol=1)
    plt.suptitle(f"Per-channel RMSE vs interpolation hour (test {Path(args.metrics_dir).name})")
    fig.savefig(out / "rmse_per_hour_per_channel.png", dpi=120, bbox_inches="tight")
    print(f"[ok] wrote {out / 'rmse_per_hour_per_channel.png'}")
    plt.close(fig)

    # ─── 3. Bar chart: Δ% vs bilinear per (model, channel) ───────────────
    model_names = [name for name, *_ in table_rows]
    chan_short = [sh for sh, _ in CHANNELS]
    deltas = np.zeros((len(model_names), len(chan_short)))
    for mi, (name, _, _, per_ch, _) in enumerate(table_rows):
        for ci, sh in enumerate(chan_short):
            deltas[mi, ci] = per_ch[sh][1]

    fig, ax = plt.subplots(figsize=(max(8, 1.2 * len(model_names)), 5), constrained_layout=True)
    x = np.arange(len(model_names))
    width = 0.16
    for ci, sh in enumerate(chan_short):
        ax.bar(x + (ci - 2) * width, deltas[:, ci], width, label=sh)
    ax.set_xticks(x)
    ax.set_xticklabels([n.replace("exp_", "")[:24] for n in model_names], rotation=45, ha="right")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_ylabel("Δ% RMSE vs bilinear (negative = better)")
    ax.set_title(f"Per-channel improvement over bilinear (test {Path(args.metrics_dir).name}, h={'..'.join((avg_hours[0], avg_hours[-1]))})")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3, axis="y")
    fig.savefig(out / "delta_pct_bar.png", dpi=120, bbox_inches="tight")
    print(f"[ok] wrote {out / 'delta_pct_bar.png'}")
    plt.close(fig)

    # ─── 4. Heatmap (models × channels) ──────────────────────────────────
    fig, ax = plt.subplots(figsize=(max(6, len(chan_short) * 1.0), max(4, 0.4 * len(model_names))),
                           constrained_layout=True)
    vmax = max(50, np.abs(deltas).max())
    im = ax.imshow(deltas, cmap="RdYlGn_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(chan_short)))
    ax.set_xticklabels(chan_short)
    ax.set_yticks(range(len(model_names)))
    ax.set_yticklabels([n.replace("exp_", "")[:24] for n in model_names])
    for mi in range(len(model_names)):
        for ci in range(len(chan_short)):
            v = deltas[mi, ci]
            ax.text(ci, mi, f"{v:+.0f}%", ha="center", va="center",
                    color="white" if abs(v) > 25 else "black", fontsize=9)
    plt.colorbar(im, ax=ax, label="Δ% vs bilinear")
    ax.set_title(f"Per-channel Δ% per model (test {Path(args.metrics_dir).name})")
    fig.savefig(out / "heatmap_per_model_per_channel.png", dpi=120, bbox_inches="tight")
    print(f"[ok] wrote {out / 'heatmap_per_model_per_channel.png'}")
    plt.close(fig)

    # ─── 5. Energy spectra (optional) ────────────────────────────────────
    if args.energy_spectra_npz and Path(args.energy_spectra_npz).exists():
        data = np.load(args.energy_spectra_npz, allow_pickle=True)
        k = data["k"]
        truth = data["truth"]
        bilinear = data["bilinear"]
        labels = data["channel_labels"].tolist() if "channel_labels" in data else None
        model_keys = [kk for kk in data.files
                      if kk not in {"k", "truth", "bilinear", "channel_labels", "hour", "delta_t", "n_samples"}]
        n_show = min(8, len(labels) if labels else truth.shape[0])
        fig, axes = plt.subplots(2, 4, figsize=(16, 8), constrained_layout=True)
        for i in range(n_show):
            ax = axes.flat[i]
            ax.loglog(k[1:], truth[i, 1:], "k-", lw=2, label="truth")
            ax.loglog(k[1:], bilinear[i, 1:], "b--", label="bilinear")
            for mk in model_keys:
                ax.loglog(k[1:], data[mk][i, 1:], "-", lw=1, alpha=0.85, label=mk[:14])
            lab = labels[i] if labels else f"ch{i}"
            ax.set_title(lab)
            ax.set_xlabel("wavenumber k")
            ax.set_ylabel("E(k)")
            ax.grid(alpha=0.3, which="both")
            if i == 0:
                ax.legend(fontsize=7, loc="lower left")
        plt.suptitle(f"Energy spectra @ h={int(data['hour']) if 'hour' in data else '?'}")
        fig.savefig(out / "energy_spectra.png", dpi=120, bbox_inches="tight")
        print(f"[ok] wrote {out / 'energy_spectra.png'}")
        plt.close(fig)

    print(f"\n[done] all artifacts in {out}/")


if __name__ == "__main__":
    main()
