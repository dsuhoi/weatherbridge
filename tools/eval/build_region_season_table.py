#!/usr/bin/env python3
"""Build paper/A5_region_season_table.tex from region_season eval JSONs."""
from __future__ import annotations

import json
import statistics
from pathlib import Path

MODELS = [
    ("dcae_skip_0p5_6yr_pad",       r"WeatherDCAE-Skip"),
    ("atm_vfi_v2_135only",          "ATM-VFI v2"),
    ("sdyff_dyffusion_0p5_6yr",     "S-DYff"),
    ("modafno_0p5_6yr",             "ModAFNO"),
    ("fuxi_0p5_6yr",                "SwinV2"),
]
REGIONS = [
    ("tropics",        "Tropics ($30°$S--$30°$N)"),
    ("midlatitudes_N", "Mid-lat N ($30°$--$60°$)"),
    ("midlatitudes_S", "Mid-lat S ($-60°$--$-30°$)"),
    ("polar_N",        "Polar N ($60°$--$90°$)"),
    ("polar_S",        "Polar S ($-90°$--$-60°$)"),
]
SEASONS = [("DJF","DJF"),("MAM","MAM"),("JJA","JJA"),("SON","SON")]
HOURS = ["1","2","3","4","5"]


def mean_delta(d_model, region, season):
    deltas = []
    for h in HOURS:
        node = d_model['per_region_season_hour'][region][season].get(h, {})
        if not node:
            continue
        for cn, mv in node['model'].items():
            bv = node['bilinear'][cn]
            if bv > 0:
                deltas.append((mv - bv) / bv * 100.0)
    return statistics.mean(deltas) if deltas else float("nan")


def fmt(d):
    if d != d:  # NaN
        return "---"
    sign = "+" if d > 0 else ""
    color = "ForestGreen" if d < 0 else "BrickRed"
    return rf"\textcolor{{{color}}}{{{sign}{d:.1f}}}"


def main():
    data = {}
    for stem, _ in MODELS:
        p = Path(f"metrics/region_season_2020/{stem}.json")
        if p.exists():
            data[stem] = json.loads(p.read_text())
    lines = []
    lines.append(r"% Region × season breakdown (auto-generated)")
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{Per-region $\times$ per-season $\Delta$RMSE\% vs bilinear baseline "
                 r"for 0.5$^\circ$ models on the 2020 test year (mean over the fixed 24-field "
                 r"protocol and interior hours "
                 r"$h{=}1\ldots5$). Regions use area-weighted means; seasons are DJF (Dec–Feb), "
                 r"MAM (Mar–May), JJA (Jun–Aug), SON (Sep–Nov).}")
    lines.append(r"\label{tab:region_season_6yr}")
    lines.append(r"\small")
    lines.append(r"\setlength{\tabcolsep}{4pt}")
    lines.append(r"\begin{tabular}{ll" + "r" * len(SEASONS) + "r}")
    lines.append(r"\toprule")
    seas_hdr = " & ".join(f"\\textbf{{{s_disp}}}" for _, s_disp in SEASONS)
    lines.append(rf"\textbf{{Region}} & \textbf{{Model}} & {seas_hdr} & \textbf{{Mean}} \\")
    lines.append(r"\midrule")
    first = True
    for ri, (r, r_disp) in enumerate(REGIONS):
        if not first:
            lines.append(r"\midrule")
        first = False
        for mi, (stem, disp) in enumerate(MODELS):
            if stem not in data:
                continue
            d = data[stem]
            cells = []
            season_vals = []
            for s, _ in SEASONS:
                val = mean_delta(d, r, s)
                season_vals.append(val)
                cells.append(fmt(val))
            mean_val = statistics.mean([v for v in season_vals if v == v]) if any(v == v for v in season_vals) else float("nan")
            row_region = r_disp if mi == 0 else ""
            lines.append(f"{row_region} & {disp} & " + " & ".join(cells) + f" & {fmt(mean_val)} \\\\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table*}")
    out = Path("paper/A5_region_season_table.tex")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
