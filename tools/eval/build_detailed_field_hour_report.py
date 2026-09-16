#!/usr/bin/env python3
"""Build field-by-hour tables from the detailed benchmark artifact tree."""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np

DISPLAY_NAMES = {
    "weatherbridge_detail": "Detail-bypass ablation",
    "refine": "Refine",
    "flow_spectral": "WeatherBridge",
    "flow": "Flow",
    "weatherdcae_14m_6yr": "WeatherDCAE-14M (6-year)",
    "weatherdcae_14m": "WeatherDCAE-14M",
    "linear": "Linear interpolation",
}

MODEL_ALIASES = {
    "flow_detail_8ep": "weatherbridge_detail",
    "flow_spectral_8ep": "flow_spectral",
}


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    columns = list(rows[0])
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _model_jsons(full_year: Path) -> Iterable[Path]:
    for path in sorted(full_year.glob("*.json")):
        if path.name.startswith("paired_"):
            continue
        try:
            payload = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if "per_tau" in payload and "channel_names" in payload:
            yield path


def collect_field_rows(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for horizon_dir in sorted(root.glob("*h")):
        horizon = horizon_dir.name
        for year_dir in sorted(horizon_dir.glob("20??")):
            full_year = year_dir / "full_year"
            linear_added = False
            for path in _model_jsons(full_year):
                payload = json.loads(path.read_text())
                model = path.stem
                channels = payload["channel_names"]
                seen = {int(value) for value in payload.get("seen_tau", [])}
                for tau_text, methods in payload["per_tau"].items():
                    tau = int(tau_text)
                    methods_to_add = [("model", model)]
                    if not linear_added and "bilinear" in methods:
                        methods_to_add.append(("bilinear", "linear"))
                    for method, output_name in methods_to_add:
                        values = methods[method]
                        for field in channels:
                            rows.append(
                                {
                                    "horizon": horizon,
                                    "year": int(year_dir.name),
                                    "tau_hour": tau,
                                    "tau_seen_in_training": tau in seen,
                                    "field": field,
                                    "model": output_name,
                                    "rmse_norm": values[f"rmse_norm_{field}"],
                                    "rmse_phys": values[f"rmse_phys_{field}"],
                                    "acc": values.get(f"acc_{field}", ""),
                                }
                            )
                linear_added = linear_added or any(
                    "bilinear" in methods for methods in payload["per_tau"].values()
                )
    return rows


def collect_relative_rows(
    field_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int, int, str], dict[str, dict[str, Any]]] = (
        defaultdict(dict)
    )
    for row in field_rows:
        key = (row["horizon"], row["year"], row["tau_hour"], row["field"])
        grouped[key][row["model"]] = row
    rows = []
    for key, models in sorted(grouped.items()):
        candidate = models.get("weatherbridge_detail")
        if candidate is None:
            continue
        for baseline, base in sorted(models.items()):
            if baseline == "weatherbridge_detail":
                continue
            cand_rmse = float(candidate["rmse_norm"])
            base_rmse = float(base["rmse_norm"])
            cand_acc = candidate["acc"]
            base_acc = base["acc"]
            rows.append(
                {
                    "horizon": key[0],
                    "year": key[1],
                    "tau_hour": key[2],
                    "field": key[3],
                    "baseline": baseline,
                    "candidate_rmse_norm": cand_rmse,
                    "baseline_rmse_norm": base_rmse,
                    "rmse_relative_delta_pct": 100.0 * (cand_rmse / base_rmse - 1.0),
                    "candidate_rmse_wins": cand_rmse < base_rmse,
                    "acc_delta": (
                        float(cand_acc) - float(base_acc)
                        if cand_acc != "" and base_acc != ""
                        else ""
                    ),
                }
            )
    return rows


def collect_spectral_rows(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for horizon_dir in sorted(root.glob("*h")):
        for year_dir in sorted(horizon_dir.glob("20??")):
            spectra_dir = year_dir / "spectra"
            for path in sorted(spectra_dir.glob("*_tau*.npz")):
                if path.name.startswith("paired_"):
                    continue
                with np.load(path, allow_pickle=False) as arrays:
                    raw_model = str(arrays["model_name"].item())
                    model = MODEL_ALIASES.get(raw_model, raw_model)
                    tau = int(arrays["tau"].item())
                    channels = [str(value) for value in arrays["channel_names"]]
                    ratio = np.asarray(arrays["window_hf_energy_ratio"], dtype=float)
                    shape = np.asarray(
                        arrays["window_hf_log_shape_error"], dtype=float
                    )
                    coherence = np.asarray(
                        arrays["window_hf_coherence"], dtype=float
                    )
                    for index, field in enumerate(channels):
                        field_ratio = ratio[:, index]
                        rows.append(
                            {
                                "horizon": horizon_dir.name,
                                "year": int(year_dir.name),
                                "tau_hour": tau,
                                "field": field,
                                "model": model,
                                "n_windows": ratio.shape[0],
                                "hf_energy_ratio": float(field_ratio.mean()),
                                "hf_energy_log_error": float(
                                    np.abs(np.log(np.clip(field_ratio, 1e-12, None))).mean()
                                ),
                                "hf_log_shape_error": float(shape[:, index].mean()),
                                "hf_coherence": float(coherence[:, index].mean()),
                            }
                        )
    return rows


def collect_spectral_relative_rows(
    spectral_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int, int, str], dict[str, dict[str, Any]]] = (
        defaultdict(dict)
    )
    for row in spectral_rows:
        key = (row["horizon"], row["year"], row["tau_hour"], row["field"])
        grouped[key][row["model"]] = row
    rows = []
    for key, models in sorted(grouped.items()):
        candidate = models.get("weatherbridge_detail")
        if candidate is None:
            continue
        for baseline, base in sorted(models.items()):
            if baseline == "weatherbridge_detail":
                continue
            rows.append(
                {
                    "horizon": key[0],
                    "year": key[1],
                    "tau_hour": key[2],
                    "field": key[3],
                    "baseline": baseline,
                    "energy_log_error_delta": float(
                        candidate["hf_energy_log_error"]
                    )
                    - float(base["hf_energy_log_error"]),
                    "energy_error_wins": float(
                        candidate["hf_energy_log_error"]
                    )
                    < float(base["hf_energy_log_error"]),
                    "shape_error_delta": float(candidate["hf_log_shape_error"])
                    - float(base["hf_log_shape_error"]),
                    "shape_error_wins": float(candidate["hf_log_shape_error"])
                    < float(base["hf_log_shape_error"]),
                    "coherence_delta": float(candidate["hf_coherence"])
                    - float(base["hf_coherence"]),
                    "coherence_wins": float(candidate["hf_coherence"])
                    > float(base["hf_coherence"]),
                }
            )
    return rows


def collect_physics_rows(root: Path) -> list[dict[str, Any]]:
    rows = []
    for horizon_dir in sorted(root.glob("*h")):
        for year_dir in sorted(horizon_dir.glob("20??")):
            for path in sorted(year_dir.glob("physics_*.json")):
                payload = json.loads(path.read_text())
                model = payload["model_name"]
                for tau, methods in payload["per_tau"].items():
                    value = methods[model]
                    rows.append(
                        {
                            "horizon": horizon_dir.name,
                            "year": int(year_dir.name),
                            "tau_hour": int(tau),
                            "model": model,
                            "n_pairs": value["n_pairs"],
                            "ageostrophic_ratio_850": value["ageo_ratio_850"],
                            "ageostrophic_ratio_700": value["ageo_ratio_700"],
                            "hydrostatic_ratio": value["hydrostatic_ratio"],
                        }
                    )
    return rows


def collect_diurnal_rows(root: Path) -> list[dict[str, Any]]:
    rows = []
    for horizon_dir in sorted(root.glob("*h")):
        for year_dir in sorted(horizon_dir.glob("20??")):
            for path in sorted(year_dir.glob("diurnal_*.json")):
                payload = json.loads(path.read_text())
                for region, months in payload["monthly"].items():
                    for month, fields in months.items():
                        for field, value in fields.items():
                            rows.append(
                                {
                                    "horizon": horizon_dir.name,
                                    "year": int(year_dir.name),
                                    "model": payload["model_name"],
                                    "region": region,
                                    "month": month,
                                    "field": field,
                                    "amplitude_ratio": value["amp_ratio"],
                                    "amplitude_abs_error": abs(
                                        float(value["amp_ratio"]) - 1.0
                                    ),
                                    "peak_hour_error": value["peak_hour_error"],
                                }
                            )
    return rows


def collect_region_rows(root: Path) -> list[dict[str, Any]]:
    rows = []
    for horizon_dir in sorted(root.glob("*h")):
        for year_dir in sorted(horizon_dir.glob("20??")):
            region_dir = year_dir / "region_season"
            for path in sorted(region_dir.glob("*.json")):
                payload = json.loads(path.read_text())
                model_name = payload.get("name", path.stem)
                entries = payload.get("per_region_season_hour", {})
                for region, seasons in entries.items():
                    for season, hours in seasons.items():
                        for tau, methods in hours.items():
                            model = methods.get("model", {})
                            baseline = methods.get("bilinear", {})
                            for key, rmse in model.items():
                                if not key.startswith("rmse_"):
                                    continue
                                field = key[len("rmse_") :]
                                base_rmse = baseline.get(key, "")
                                rows.append(
                                    {
                                        "horizon": horizon_dir.name,
                                        "year": int(year_dir.name),
                                        "region": region,
                                        "season": season,
                                        "tau_hour": int(tau),
                                        "field": field,
                                        "model": model_name,
                                        "rmse_norm": rmse,
                                        "linear_rmse_norm": base_rmse,
                                        "relative_to_linear_pct": (
                                            100.0 * (float(rmse) / float(base_rmse) - 1.0)
                                            if base_rmse != "" and float(base_rmse) != 0.0
                                            else ""
                                        ),
                                        "n_windows": methods.get("n", ""),
                                    }
                                )
                aggregated = payload.get("per_region_season", {})
                for region, seasons in aggregated.items():
                    for season, values in seasons.items():
                        for field, rmse in values.get(
                            "rmse_per_channel", {}
                        ).items():
                            rows.append(
                                {
                                    "horizon": horizon_dir.name,
                                    "year": int(year_dir.name),
                                    "region": region,
                                    "season": season,
                                    "tau_hour": payload.get(
                                        "tau_aggregation", "aggregate"
                                    ),
                                    "field": field,
                                    "model": payload.get(
                                        "model_name", model_name
                                    ),
                                    "rmse_norm": rmse,
                                    "linear_rmse_norm": "",
                                    "relative_to_linear_pct": "",
                                    "n_windows": values.get(
                                        "n_samples", ""
                                    ),
                                }
                            )
    return rows


def _append_pair(
    rows: list[dict[str, Any]],
    *,
    horizon: str,
    year: int,
    baseline: str,
    metric: str,
    tau: int | str,
    payload: dict[str, Any],
    left_key: str,
    right_key: str,
) -> None:
    rows.append(
        {
            "horizon": horizon,
            "year": year,
            "baseline": baseline,
            "metric": metric,
            "tau_hour": tau,
            "candidate": payload[left_key],
            "baseline_value": payload[right_key],
            "delta_candidate_minus_baseline": payload[
                "delta_left_minus_right"
            ],
            "ci95_low": payload["delta_ci95"][0],
            "ci95_high": payload["delta_ci95"][2],
            "p_raw": payload["p_paired_block_permutation"],
            "p_holm": "",
        }
    )


def collect_paired_rows(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for horizon_dir in sorted(root.glob("*h")):
        for year_dir in sorted(horizon_dir.glob("20??")):
            full_year = year_dir / "full_year"
            for name, metric in (
                ("paired_rmse.json", "rmse_norm"),
                ("paired_acc.json", "acc"),
                ("paired_temporal_curvature.json", "temporal_curvature_rmse"),
            ):
                path = full_year / name
                if not path.is_file():
                    continue
                payload = json.loads(path.read_text())
                for baseline, comparison in payload["comparisons"].items():
                    if metric == "acc":
                        left_key, right_key = "left", "right"
                    else:
                        left_key, right_key = "left_rmse", "right_rmse"
                    _append_pair(
                        rows,
                        horizon=horizon_dir.name,
                        year=int(year_dir.name),
                        baseline=baseline,
                        metric=metric,
                        tau="all",
                        payload=comparison,
                        left_key=left_key,
                        right_key=right_key,
                    )
                    if metric == "temporal_curvature_rmse":
                        continue
                    for tau, per_tau in comparison.get("per_tau", {}).items():
                        _append_pair(
                            rows,
                            horizon=horizon_dir.name,
                            year=int(year_dir.name),
                            baseline=baseline,
                            metric=metric,
                            tau=int(tau),
                            payload=per_tau,
                            left_key=left_key,
                            right_key=right_key,
                        )

            physical_path = full_year / "paired_physical.json"
            if physical_path.is_file():
                payload = json.loads(physical_path.read_text())
                for baseline, comparison in payload["comparisons"].items():
                    for diagnostic, values in comparison["diagnostics"].items():
                        _append_pair(
                            rows,
                            horizon=horizon_dir.name,
                            year=int(year_dir.name),
                            baseline=baseline,
                            metric=f"physical_{diagnostic}",
                            tau="all",
                            payload=values,
                            left_key="left",
                            right_key="right",
                        )

            for path in sorted((year_dir / "spectra").glob("paired_tau*.json")):
                payload = json.loads(path.read_text())
                for baseline, comparison in payload["comparisons"].items():
                    for metric, values in comparison.items():
                        if metric not in {
                            "energy_log_error",
                            "shape_log_error",
                            "coherence",
                        }:
                            continue
                        _append_pair(
                            rows,
                            horizon=horizon_dir.name,
                            year=int(year_dir.name),
                            baseline=baseline,
                            metric=f"spectral_{metric}",
                            tau=int(comparison["tau"]),
                            payload=values,
                            left_key="left",
                            right_key="right",
                        )

    families: dict[tuple[str, int, str, str], list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        families[
            (row["horizon"], row["year"], row["baseline"], row["metric"])
        ].append(index)
    for indices in families.values():
        order = sorted(indices, key=lambda index: float(rows[index]["p_raw"]))
        adjusted = [0.0] * len(order)
        running = 0.0
        count = len(order)
        for rank, index in enumerate(order):
            running = max(
                running,
                min(1.0, (count - rank) * float(rows[index]["p_raw"])),
            )
            adjusted[rank] = running
        for rank, index in enumerate(order):
            rows[index]["p_holm"] = adjusted[rank]
    return rows


def _mean(rows: list[dict[str, Any]], key: str) -> float:
    values = [float(row[key]) for row in rows if row.get(key, "") != ""]
    return float(np.mean(values)) if values else float("nan")


def build_markdown(
    field_rows: list[dict[str, Any]],
    relative_rows: list[dict[str, Any]],
    spectral_rows: list[dict[str, Any]],
    physics_rows: list[dict[str, Any]],
    diurnal_rows: list[dict[str, Any]],
    paired_rows: list[dict[str, Any]],
) -> str:
    lines = [
        "# Detailed Weather Interpolation Benchmark",
        "",
        "Lower RMSE, spectral energy/shape error, and balance-ratio distance from one are better; higher ACC and coherence are better.",
        "",
        "## Aggregate RMSE and ACC",
        "",
        "| Horizon | Year | Model | RMSE norm | ACC | Field-hour cells |",
        "|---|---:|---|---:|---:|---:|",
    ]
    aggregate_groups: dict[tuple[str, int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in field_rows:
        aggregate_groups[(row["horizon"], row["year"], row["model"])].append(row)
    for key, rows in sorted(aggregate_groups.items()):
        lines.append(
            f"| {key[0]} | {key[1]} | {DISPLAY_NAMES.get(key[2], key[2])} | "
            f"{_mean(rows, 'rmse_norm'):.6f} | {_mean(rows, 'acc'):.6f} | {len(rows)} |"
        )

    lines.extend(
        [
            "",
            "## WeatherBridge detail-path ablation by hour",
            "",
            "| Horizon | Year | Baseline | Hour | Delta RMSE | Wins |",
            "|---|---:|---|---:|---:|---:|",
        ]
    )
    relative_groups: dict[tuple[str, int, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in relative_rows:
        relative_groups[
            (row["horizon"], row["year"], row["baseline"], row["tau_hour"])
        ].append(row)
    for key, rows in sorted(relative_groups.items()):
        wins = sum(bool(row["candidate_rmse_wins"]) for row in rows)
        lines.append(
            f"| {key[0]} | {key[1]} | {DISPLAY_NAMES.get(key[2], key[2])} | "
            f"{key[3]} | {_mean(rows, 'rmse_relative_delta_pct'):+.2f}% | "
            f"{wins}/{len(rows)} |"
        )

    exceptions = [row for row in relative_rows if not row["candidate_rmse_wins"]]
    lines.extend(["", "## RMSE exceptions", ""])
    if exceptions:
        ordered = sorted(
            exceptions,
            key=lambda value: value["rmse_relative_delta_pct"],
            reverse=True,
        )
        lines.append(
            f"Showing the 30 largest regressions out of {len(ordered)}; "
            "the CSV contains every field-hour cell."
        )
        lines.append("")
        for row in ordered[:30]:
            lines.append(
                f"- {row['horizon']} {row['year']} h={row['tau_hour']} "
                f"{row['field']} vs {DISPLAY_NAMES.get(row['baseline'], row['baseline'])}: "
                f"{row['rmse_relative_delta_pct']:+.3f}%"
            )
    else:
        lines.append("No field-hour RMSE exceptions.")

    lines.extend(
        [
            "",
            "## Spherical spectrum",
            "",
            "| Horizon | Year | Model | Energy ratio | Energy error | Shape error | Coherence |",
            "|---|---:|---|---:|---:|---:|---:|",
        ]
    )
    spectral_groups: dict[tuple[str, int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in spectral_rows:
        spectral_groups[(row["horizon"], row["year"], row["model"])].append(row)
    for key, rows in sorted(spectral_groups.items()):
        lines.append(
            f"| {key[0]} | {key[1]} | {DISPLAY_NAMES.get(key[2], key[2])} | "
            f"{_mean(rows, 'hf_energy_ratio'):.4f} | "
            f"{_mean(rows, 'hf_energy_log_error'):.4f} | "
            f"{_mean(rows, 'hf_log_shape_error'):.4f} | "
            f"{_mean(rows, 'hf_coherence'):.4f} |"
        )

    if physics_rows:
        lines.extend(
            [
                "",
                "## Diagnostic balance",
                "",
                "| Horizon | Year | Model | Ageo 850 | Ageo 700 | Hydrostatic |",
                "|---|---:|---|---:|---:|---:|",
            ]
        )
        groups: dict[tuple[str, int, str], list[dict[str, Any]]] = defaultdict(list)
        for row in physics_rows:
            groups[(row["horizon"], row["year"], row["model"])].append(row)
        for key, rows in sorted(groups.items()):
            lines.append(
                f"| {key[0]} | {key[1]} | {DISPLAY_NAMES.get(key[2], key[2])} | "
                f"{_mean(rows, 'ageostrophic_ratio_850'):.4f} | "
                f"{_mean(rows, 'ageostrophic_ratio_700'):.4f} | "
                f"{_mean(rows, 'hydrostatic_ratio'):.4f} |"
            )

    if diurnal_rows:
        lines.extend(
            [
                "",
                "## Regional diurnal cycle",
                "",
                "| Horizon | Year | Model | Amplitude error | Peak error (h) | Strata |",
                "|---|---:|---|---:|---:|---:|",
            ]
        )
        groups = defaultdict(list)
        for row in diurnal_rows:
            groups[(row["horizon"], row["year"], row["model"])].append(row)
        for key, rows in sorted(groups.items()):
            lines.append(
                f"| {key[0]} | {key[1]} | {DISPLAY_NAMES.get(key[2], key[2])} | "
                f"{_mean(rows, 'amplitude_abs_error'):.4f} | "
                f"{_mean(rows, 'peak_hour_error'):.3f} | {len(rows)} |"
            )
    if paired_rows:
        lines.extend(
            [
                "",
                "## Paired weekly-block tests",
                "",
                "Holm correction is applied within each horizon-year-baseline-metric family.",
                "",
                "| Horizon | Year | Baseline | Metric | Significant | Tests |",
                "|---|---:|---|---|---:|---:|",
            ]
        )
        groups = defaultdict(list)
        for row in paired_rows:
            groups[
                (row["horizon"], row["year"], row["baseline"], row["metric"])
            ].append(row)
        for key, rows in sorted(groups.items()):
            significant = sum(float(row["p_holm"]) < 0.05 for row in rows)
            lines.append(
                f"| {key[0]} | {key[1]} | {DISPLAY_NAMES.get(key[2], key[2])} | "
                f"{key[3]} | {significant} | {len(rows)} |"
            )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    field_rows = collect_field_rows(args.root)
    relative_rows = collect_relative_rows(field_rows)
    spectral_rows = collect_spectral_rows(args.root)
    spectral_relative_rows = collect_spectral_relative_rows(spectral_rows)
    physics_rows = collect_physics_rows(args.root)
    diurnal_rows = collect_diurnal_rows(args.root)
    region_rows = collect_region_rows(args.root)
    paired_rows = collect_paired_rows(args.root)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.out_dir / "field_hour_metrics.csv", field_rows)
    _write_csv(args.out_dir / "field_hour_relative.csv", relative_rows)
    _write_csv(args.out_dir / "spectral_field_hour.csv", spectral_rows)
    _write_csv(
        args.out_dir / "spectral_field_hour_relative.csv",
        spectral_relative_rows,
    )
    _write_csv(args.out_dir / "physics_hour.csv", physics_rows)
    _write_csv(args.out_dir / "diurnal_region_month_field.csv", diurnal_rows)
    _write_csv(args.out_dir / "region_season_field_hour.csv", region_rows)
    _write_csv(args.out_dir / "paired_tests.csv", paired_rows)
    (args.out_dir / "REPORT.md").write_text(
        build_markdown(
            field_rows,
            relative_rows,
            spectral_rows,
            physics_rows,
            diurnal_rows,
            paired_rows,
        )
    )
    print(
        json.dumps(
            {
                "field_hour_rows": len(field_rows),
                "relative_rows": len(relative_rows),
                "spectral_rows": len(spectral_rows),
                "spectral_relative_rows": len(spectral_relative_rows),
                "physics_rows": len(physics_rows),
                "diurnal_rows": len(diurnal_rows),
                "region_rows": len(region_rows),
                "paired_rows": len(paired_rows),
                "out_dir": str(args.out_dir),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
