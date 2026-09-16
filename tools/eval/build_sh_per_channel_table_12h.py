#!/usr/bin/env python3
"""Build the per-channel SH HF-preservation table for the 12h paper.

Reads ``metrics/sh_spectra_12h_ep10_24ch/<model>_tau<τ>.npz`` (output of
``tools/eval/sh_energy_spectra_12h.py --channels all``) and writes:

  1. ``metrics/sh_spectra_12h_ep10_24ch/per_channel_breakdown.json``
     — per (model, τ, channel) R_HF / R_LF / R_total ratios and aggregated
     means.
  2. ``paper/tab_sh_per_channel_12h.tex`` — LaTeX table with all 24 channels
     grouped by class (4 PL variables × 4 levels + 4 surface) and an
     aggregate summary row per pressure level / per surface.

Each entry in the .npz contains ``ell``, ``pred_El``, ``gt_El``,
``channel_names``, ``tau``. We accept any subset of channels — channels
absent from the npz are skipped with a warning.

Run::

    python tools/eval/build_sh_per_channel_table_12h.py \\
        --npz-dir metrics/sh_spectra_12h_ep10_24ch \\
        --out-json metrics/sh_spectra_12h_ep10_24ch/per_channel_breakdown.json \\
        --out-tex paper/tab_sh_per_channel_12h.tex \\
        --hf-frac 0.5
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np


# Display ordering for the LaTeX table.
PL_VARS = ["T", "U", "V", "Q", "Z"]
PL_LEVELS = [1000, 925, 850, 700]
SURF_VARS = ["t2m", "u10", "v10", "mslp"]
ALL_24 = [f"{v}{lvl}" for v in PL_VARS for lvl in PL_LEVELS] + list(SURF_VARS)


# Model stem → paper display name.
MODEL_DISPLAY = {
    "refine": "Refine ablation",
    "flow_spectral": "WeatherBridge",
    "weatherdcae_14m": "WeatherDCAE-14M",
    "pixelattn_vfi": "PixelAttn-VFI",
    "fuxi": "SwinV2",
    "modafno": "ModAFNO",
    "sdyff": "S-DYff",
    "linear": "Linear Interp.",
}

# Column order for the table (kept identical to the existing 5-channel
# version so reviewers can diff rows directly).
TABLE_MODEL_ORDER = [
    "WeatherBridge",
    "WeatherDCAE-14M",
    "PixelAttn-VFI",
    "SwinV2",
    "ModAFNO",
    "S-DYff",
    "Linear Interp.",
]


def _ratios(pred_El: np.ndarray, gt_El: np.ndarray, ell_lo: int) -> Dict[str, float]:
    eps = 1e-30
    r_hf = float(pred_El[ell_lo:].sum() / max(float(gt_El[ell_lo:].sum()), eps))
    r_lf = float(pred_El[:ell_lo].sum() / max(float(gt_El[:ell_lo].sum()), eps))
    r_tot = float(pred_El.sum() / max(float(gt_El.sum()), eps))
    return {"R_HF": r_hf, "R_LF": r_lf, "R_total": r_tot}


def _load_all_npz(npz_dir: Path, ell_min: int, taus: set[int]):
    """Return dict[model_stem][tau][channel] -> {R_HF, R_LF, R_total}."""
    per_entry: Dict[str, Dict[int, Dict[str, Dict[str, float]]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    lmax_seen = None
    for p in sorted(npz_dir.glob("*.npz")):
        d = np.load(p, allow_pickle=False)
        if "model_name" not in d.files or "tau" not in d.files:
            continue
        model = str(d["model_name"])
        tau = int(d["tau"])
        if tau not in taus or model == "weatherbridge":
            continue
        metadata = json.loads(str(d["metadata_json"].item()))
        if (
            metadata.get("schema_version") != 6
            or metadata.get("sht_grid")
            != "wb2_0p25_2x2_block_average_v1_latitude_strip_area_sht"
            or metadata.get("sample_strategy") != "all_valid_anchor_windows"
        ):
            raise ValueError(f"{p}: table requires dense schema-v6 spectra")
        channels = [str(c) for c in d["channel_names"]]
        ell = d["ell"]
        lmax_i = int(ell.max())
        if not 1 <= ell_min <= lmax_i:
            raise ValueError(f"ell_min={ell_min} is invalid for {p}")
        if lmax_seen is None:
            lmax_seen = lmax_i
        elif lmax_seen != lmax_i:
            print(f"  warn: lmax mismatch in {p.name} ({lmax_i} vs {lmax_seen})")
        for ci, ch in enumerate(channels):
            r = _ratios(d["pred_El"][ci], d["gt_El"][ci], ell_min)
            per_entry[model][tau][ch] = r
    return per_entry, lmax_seen


def _avg_over_taus(per_entry):
    """Reduce per_entry[stem][tau][ch] -> per_entry_mean[stem][ch] (mean over τ)."""
    out: Dict[str, Dict[str, Dict[str, float]]] = {}
    for stem, by_tau in per_entry.items():
        agg: Dict[str, List[Dict[str, float]]] = defaultdict(list)
        for tau, by_ch in by_tau.items():
            for ch, r in by_ch.items():
                agg[ch].append(r)
        out[stem] = {
            ch: {k: float(np.mean([r[k] for r in rs])) for k in rs[0]}
            for ch, rs in agg.items()
        }
    return out


def _fmt_ratio(x: float) -> str:
    return f"{x:.3f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz-dir", default="metrics/journal_spectra_v6/12h_2020")
    ap.add_argument("--out-json",
                    default="metrics/journal_spectra_v6/12h_2020/per_channel_breakdown.json")
    ap.add_argument("--out-tex", default="paper/tab_sh_per_channel_12h.tex")
    ap.add_argument("--ell-min", type=int, default=80)
    ap.add_argument("--taus", default="1,2,3,4,5,6,7,8,9,10,11")
    ap.add_argument("--horizon", type=int, choices=(6, 12), default=12)
    ap.add_argument("--label", default="tab:sh_per_channel_12h")
    args = ap.parse_args()

    npz_dir = Path(args.npz_dir)
    out_json = Path(args.out_json)
    out_tex = Path(args.out_tex)

    taus = {int(value) for value in args.taus.split(",")}
    per_entry, lmax = _load_all_npz(npz_dir, args.ell_min, taus)
    if not per_entry:
        raise SystemExit(f"No NPZ artefacts found in {npz_dir}")

    avg = _avg_over_taus(per_entry)
    ell_lo = args.ell_min

    # JSON payload.
    stems_present = sorted(avg.keys())
    display_names = [MODEL_DISPLAY.get(s, s) for s in stems_present]
    all_channels = sorted({ch for stem in avg for ch in avg[stem]},
                          key=lambda c: ALL_24.index(c) if c in ALL_24 else 999)
    R_HF = {ch: {MODEL_DISPLAY.get(s, s): avg[s].get(ch, {}).get("R_HF")
                 for s in stems_present}
            for ch in all_channels}
    R_LF = {ch: {MODEL_DISPLAY.get(s, s): avg[s].get(ch, {}).get("R_LF")
                 for s in stems_present}
            for ch in all_channels}
    R_total = {ch: {MODEL_DISPLAY.get(s, s): avg[s].get(ch, {}).get("R_total")
                    for s in stems_present}
               for ch in all_channels}

    payload = {
        "schema_version": 2,
        "metric": ("R_HF = sum(E_pred[l>=l_lo])/sum(E_gt[l>=l_lo]); "
                   "R_LF = sum(E_pred[l<l_lo])/sum(E_gt[l<l_lo]); "
                   "R_total = sum(E_pred)/sum(E_gt)"),
        "hf_lo": ell_lo,
        "lmax": lmax,
        "taus": sorted(taus),
        "averaging": "uniform mean over available τ per (model, channel)",
        f"{args.horizon}h": {
            "model_stems": stems_present,
            "display_names": display_names,
            "channels": all_channels,
            "n_channels": len(all_channels),
            "R_HF": R_HF,
            "R_LF": R_LF,
            "R_total": R_total,
        },
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2, default=str))
    print(f"saved {out_json}  ({len(all_channels)} channels × {len(stems_present)} models)")

    # ---------------- LaTeX table ----------------
    # Pick the columns that are actually present.
    cols = [m for m in TABLE_MODEL_ORDER if m in display_names]
    if not cols:
        cols = display_names  # fallback to whatever we have

    def best_worst(ch: str):
        """Closest-to-1 = best, farthest-from-1 = worst (per row)."""
        vals = []
        for m in cols:
            v = R_HF[ch].get(m)
            if v is not None:
                vals.append((m, v))
        if not vals:
            return None, None
        best = min(vals, key=lambda x: abs(x[1] - 1.0))[0]
        worst = max(vals, key=lambda x: abs(x[1] - 1.0))[0]
        return best, worst

    def row(ch: str) -> str:
        best, worst = best_worst(ch)
        cells = [ch]
        for m in cols:
            v = R_HF[ch].get(m)
            if v is None:
                cells.append("--")
            else:
                s = _fmt_ratio(v)
                if m == best:
                    s = r"\textbf{" + s + "}"
                if m == worst:
                    s = r"\underline{" + s + "}"
                cells.append(s)
        return " & ".join(cells) + r" \\"

    def agg_row(label: str, ch_subset: List[str]) -> str:
        cells = [label]
        for m in cols:
            xs = [R_HF[ch][m] for ch in ch_subset
                  if R_HF.get(ch, {}).get(m) is not None]
            cells.append(_fmt_ratio(float(np.mean(xs))) if xs else "--")
        return " & ".join(cells) + r" \\"

    lines: List[str] = []
    lines.append("% Auto-generated by tools/eval/build_sh_per_channel_table_12h.py")
    lines.append(
        f"% Source: {npz_dir}/*.npz  "
        f"({len(cols)} displayed models × {len(all_channels)} channels)"
    )
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    lines.append(
        r"\caption{Per-channel spherical-harmonic high-frequency preservation "
        r"ratio $R_{\mathrm{HF}} = \sum_{\ell \geq " + str(ell_lo) +
        r"} E_{\mathrm{pred}}(\ell) / \sum_{\ell \geq " + str(ell_lo) +
        r"} E_{\mathrm{GT}}(\ell)$ for the " + str(args.horizon) +
        r"\,h leaderboard, averaged over $\tau \in \{" +
        ", ".join(str(value) for value in sorted(taus)) +
        r"\}$ across all 24 prognostic channels "
        r"(5 PL variables $\times$ 4 levels = 20 + 4 surface). "
        r"$R_{\mathrm{HF}} = 1$ = perfect spectral fidelity; "
        r"values $<1$ indicate HF energy under-prediction. "
        r"Per-row \textbf{best} (closest to 1.0) and \underline{worst} "
        r"(farthest from 1.0) highlighted. Block means aggregate over the "
        r"4 PL levels per variable; the surface block mean uses the 4 "
        r"surface fields.}")
    lines.append(r"\label{" + args.label + "}")
    lines.append(r"\small")
    lines.append(r"\setlength{\tabcolsep}{4pt}")
    align = "l" + "r" * len(cols)
    lines.append(r"\resizebox{\textwidth}{!}{%")
    lines.append(r"\begin{tabular}{" + align + r"}")
    lines.append(r"\toprule")
    header = ["Channel"] + cols
    lines.append(" & ".join(header) + r" \\")
    lines.append(r"\midrule")

    # PL blocks: T, U, V, Q, Z (each: 4 level rows + block mean).
    for v in PL_VARS:
        chs = [f"{v}{lvl}" for lvl in PL_LEVELS if f"{v}{lvl}" in all_channels]
        if not chs:
            continue
        for ch in chs:
            lines.append(row(ch))
        lines.append(agg_row(rf"\textit{{mean({v}*)}}", chs))
        lines.append(r"\midrule")

    # Surface block.
    surf_present = [c for c in SURF_VARS if c in all_channels]
    for ch in surf_present:
        lines.append(row(ch))
    if surf_present:
        lines.append(agg_row(r"\textit{mean(surface)}", surf_present))
        lines.append(r"\midrule")

    # Grand mean.
    lines.append(agg_row(r"\textbf{mean (24ch)}", all_channels))
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"}")
    lines.append(r"\end{table*}")
    out_tex.parent.mkdir(parents=True, exist_ok=True)
    out_tex.write_text("\n".join(lines) + "\n")
    print(f"saved {out_tex}  ({len(cols)} columns × {len(all_channels) + len(PL_VARS) + 2} rows)")


if __name__ == "__main__":
    main()
