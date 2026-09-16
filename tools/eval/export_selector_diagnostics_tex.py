#!/usr/bin/env python3
"""Export compact selector and vector-SHT diagnostics from Supplementary Data 1."""
from __future__ import annotations

import argparse
import csv
import hashlib
import os
from pathlib import Path


REFERENCE = {
    (6, 2020): "weatherdcae_14m_6yr",
    (6, 2021): "weatherdcae_14m_6yr",
    (12, 2020): "weatherdcae_14m",
    (12, 2021): "weatherdcae_14m",
}


def _read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _one(rows: list[dict[str, str]], **filters: str) -> dict[str, str]:
    matches = [
        row
        for row in rows
        if all(row.get(key) == value for key, value in filters.items())
    ]
    if len(matches) != 1:
        raise ValueError(f"expected one row for {filters}, found {len(matches)}")
    return matches[0]


def render_selector_table(rows: list[dict[str, str]], source_hash: str) -> str:
    body = []
    for horizon, year in REFERENCE:
        reference = REFERENCE[(horizon, year)]
        common = {
            "candidate": "flow_spectral",
            "reference": reference,
            "horizon_hours": str(horizon),
            "year": str(year),
        }
        curvature = _one(
            rows,
            **common,
            family="temporal_curvature",
            metric="temporal_curvature",
            scope="aggregate",
        )
        physical = [
            row
            for row in rows
            if all(row.get(key) == value for key, value in common.items())
            and row["family"] == "physical"
        ]
        if len(physical) != 7:
            raise ValueError(f"expected seven physical diagnostics for {common}")
        significant_worse = [
            row["metric"]
            for row in physical
            if float(row["delta_left_minus_right"]) > 0
            and float(row["p_holm_two_sided"]) <= 0.05
        ]
        if significant_worse != ["lower_tropospheric_moisture_bias"]:
            raise ValueError(f"unexpected physical gate failures for {common}")
        body.append(
            (
                horizon,
                year,
                float(curvature["left"]),
                float(curvature["right"]),
                float(curvature["delta_left_minus_right"]),
                float(curvature["ci95_low"]),
                float(curvature["ci95_high"]),
            )
        )
    lines = [
        f"% Source SHA-256: {source_hash}",
        r"\begin{table}[!ht]",
        r"\centering",
        (
            r"\caption{Selector diagnostics for WeatherBridge against "
            r"WeatherDCAE-14M. Lower is better. Curvature intervals use paired "
            r"seven-day blocks ($n=53$). In the physical diagnostics "
            r"(Supplementary Data~1), lower-tropospheric moisture "
            r"bias is the sole Holm-significant regression in every row.}"
        ),
        r"\label{tab:selector-diagnostics}",
        r"\small",
        r"\setlength{\tabcolsep}{3pt}",
        r"\begin{tabular}{@{}lccc@{}}",
        r"\toprule",
        (
            r"Interval, year & \shortstack{WeatherBridge\\curvature} & "
            r"\shortstack{WeatherDCAE\\curvature} & "
            r"$\Delta$ [95\% CI] \\"
        ),
        r"\midrule",
    ]
    for horizon, year, left, right, delta, low, high in body:
        lines.append(
            f"{horizon}\\,h, {year} & {left:.5f} & {right:.5f} & "
            f"{delta:+.5f} [{low:+.5f}, {high:+.5f}]" + r" \\"
        )
    lines.extend((r"\bottomrule", r"\end{tabular}", r"\end{table}"))
    return "\n".join(lines) + "\n"


def render_vector_table(rows: list[dict[str, str]], source_hash: str) -> str:
    body = []
    for horizon, taus in ((6, (2, 3)), (12, (5, 8))):
        for tau in taus:
            for component in ("spheroidal", "toroidal"):
                row = _one(
                    rows,
                    candidate="flow_spectral",
                    reference="weatherdcae_14m",
                    horizon_hours=str(horizon),
                    year="2020",
                    family="vector_spectral",
                    metric="signed_cospectrum",
                    tau=str(tau),
                    channel=f"wind850:{component}",
                )
                body.append(
                    (
                        horizon,
                        tau,
                        component,
                        float(row["left"]),
                        float(row["right"]),
                        float(row["delta_left_minus_right"]),
                        float(row["ci95_low"]),
                        float(row["ci95_high"]),
                    )
                )
    lines = [
        f"% Source SHA-256: {source_hash}",
        r"\begin{table}[!ht]",
        r"\centering",
        (
            r"\caption{Vector-SHT signed cospectrum for 850-hPa wind in 2020. "
            r"One is ideal; higher is better. Query hours include one held-out "
            r"and one trained time per interval. Intervals are paired over "
            r"calendar-week blocks. Supplementary Data~1 contains the complete "
            r"multiplicity-corrected vector family.}"
        ),
        r"\label{tab:vector-sht}",
        r"\small",
        r"\begin{tabular}{@{}rrlccc@{}}",
        r"\toprule",
        r"Interval & $\tau$ & Component & WeatherBridge & WeatherDCAE & $\Delta$ [95\% CI] \\",
        r"\midrule",
    ]
    for horizon, tau, component, left, right, delta, low, high in body:
        lines.append(
            f"{horizon}\\,h & {tau} & {component.capitalize()} & {left:.3f} & "
            f"{right:.3f} & {delta:+.3f} [{low:+.3f}, {high:+.3f}]" + r" \\"
        )
    lines.extend((r"\bottomrule", r"\end{tabular}", r"\end{table}"))
    return "\n".join(lines) + "\n"


def _write(path: Path, contents: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(contents)
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--statistics",
        type=Path,
        default=Path("paper/supplementary_data_1_statistics.csv"),
    )
    parser.add_argument(
        "--selector-out",
        type=Path,
        default=Path("paper/tab_selector_diagnostics.tex"),
    )
    parser.add_argument(
        "--vector-out",
        type=Path,
        default=Path("paper/tab_vector_sht.tex"),
    )
    args = parser.parse_args()
    rows = _read(args.statistics)
    source_hash = hashlib.sha256(args.statistics.read_bytes()).hexdigest()
    _write(args.selector_out, render_selector_table(rows, source_hash))
    _write(args.vector_out, render_vector_table(rows, source_hash))
    print(f"wrote {args.selector_out} and {args.vector_out}")


if __name__ == "__main__":
    main()
