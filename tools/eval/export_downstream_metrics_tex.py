#!/usr/bin/env python3
"""Validate journal downstream diagnostics and export two LaTeX tables."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

from tools.train.training_protocol import memmap_dataset_provenance

MODELS = (
    ("flow_spectral", "WeatherBridge"),
    ("dcae", "WeatherDCAE-14M"),
    ("linear", "Linear Interp."),
)
PHYSICAL_METRICS = (
    "ageo_ratio_850",
    "ageo_ratio_700",
    "hydrostatic_ratio",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _finite(value: Any, context: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{context}: expected a finite value")
    return result


def _validate_file_record(record: dict[str, Any], context: str) -> None:
    path_value = record.get("path")
    expected = record.get("sha256")
    if not path_value or not expected:
        raise ValueError(f"{context}: incomplete file provenance")
    path = Path(str(path_value))
    if not path.is_file() or _sha256(path) != expected:
        raise ValueError(f"{context}: stale provenance file {path}")


def _validate_dataset_provenance(record: Any, context: str) -> str:
    if not isinstance(record, dict):
        raise ValueError(f"{context}: missing evaluation dataset provenance")
    try:
        current = memmap_dataset_provenance(
            record["root"],
            [int(year) for year in record["years"]],
            sampled_bytes_per_file=int(record["sampled_bytes_per_file_limit"]),
        )
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise ValueError(f"{context}: invalid evaluation dataset provenance") from error
    if current != record:
        raise ValueError(f"{context}: stale evaluation dataset provenance")
    if current.get("years") != [2020]:
        raise ValueError(f"{context}: expected the 2020 evaluation dataset")
    return str(current["identity_sha256"])


def _internal_name(stem: str, horizon: int) -> str:
    if stem == "dcae":
        return "weatherdcae_14m_6yr" if horizon == 6 else "weatherdcae_14m"
    return stem


def _expected_weight_name(stem: str, horizon: int) -> str | None:
    names = {
        ("flow_spectral", 6): "weatherbridge_flow_spectral_6h_bare.pt",
        ("flow_spectral", 12): "weatherbridge_flow_spectral_12h_bare.pt",
        ("dcae", 6): "weatherdcae_14m_6yr_6h_bare.pt",
        ("dcae", 12): "weatherdcae_14m_12h_bare.pt",
    }
    return names.get((stem, horizon))


def _load_physics(
    path: Path,
    horizon: int,
    model_name: str,
    expected_weight: str | None,
) -> tuple[dict[str, float], dict[str, str]]:
    payload = json.loads(path.read_text())
    expected_taus = list(range(1, horizon))
    if (
        payload.get("model_name") != model_name
        or payload.get("year") != 2020
        or payload.get("delta_t_hours") != horizon
        or payload.get("taus") != expected_taus
        or int(payload.get("n_pairs_used", -1)) != 180
        or payload.get("sampling", {}).get("strategy") != "uniform_over_year"
        or payload.get("sampling", {}).get("candidate_stride_hours") != 48
        or payload.get("sampling", {}).get("max_pairs") != 180
    ):
        raise ValueError(f"{path}: invalid physical diagnostic protocol")
    start_hours = payload.get("sample_start_hours")
    if (
        not isinstance(start_hours, list)
        or len(start_hours) != 180
        or start_hours != sorted(set(start_hours))
        or any(not isinstance(hour, int) or hour < 0 or hour % 48 for hour in start_hours)
    ):
        raise ValueError(f"{path}: invalid physical sample index")
    index_sha = hashlib.sha256(
        "\n".join(map(str, start_hours)).encode("utf-8")
    ).hexdigest()
    if payload["sampling"].get("index_sha256") != index_sha:
        raise ValueError(f"{path}: physical sample-index hash mismatch")
    provenance = payload.get("model_provenance", {})
    if model_name == "linear":
        if provenance != {"kind": "linear"}:
            raise ValueError(f"{path}: invalid linear provenance")
    else:
        _validate_file_record(provenance, f"{path}: model")
        if Path(str(provenance["path"])).name != expected_weight:
            raise ValueError(f"{path}: unexpected model artifact")
    normalization_hashes = {}
    for key, record in payload.get("normalization", {}).items():
        if isinstance(record, dict) and "path" in record:
            _validate_file_record(record, f"{path}: normalization")
            normalization_hashes[key] = str(record["sha256"])
    if set(normalization_hashes) != {"stats", "surface_stats", "static_features"}:
        raise ValueError(f"{path}: incomplete normalization provenance")
    values = {name: [] for name in PHYSICAL_METRICS}
    for tau in expected_taus:
        row = payload.get("per_tau", {}).get(str(tau), {}).get(model_name)
        if not isinstance(row, dict) or int(row.get("n_pairs", -1)) != 180:
            raise ValueError(f"{path}: incomplete physical tau={tau}")
        for metric in PHYSICAL_METRICS:
            values[metric].append(_finite(row.get(metric), f"{path}: {metric}"))
    metrics = {
        metric: sum(metric_values) / len(metric_values)
        for metric, metric_values in values.items()
    }
    provenance_summary = {
        "index_sha256": index_sha,
        "model_sha256": str(provenance.get("sha256", "linear")),
        "evaluation_dataset_identity_sha256": _validate_dataset_provenance(
            payload.get("evaluation_dataset_provenance"),
            str(path),
        ),
        **{
            f"normalization_{key}_sha256": value
            for key, value in normalization_hashes.items()
        },
    }
    return metrics, provenance_summary


def _load_diurnal(
    path: Path,
    horizon: int,
    model_name: str,
    expected_weight: str | None,
) -> tuple[dict[str, float], dict[str, str]]:
    payload = json.loads(path.read_text())
    expected_taus = list(range(1, horizon))
    if (
        payload.get("model_name") != model_name
        or payload.get("year") != 2020
        or payload.get("delta_t_hours") != horizon
        or payload.get("taus") != expected_taus
        or payload.get("surface_channels") != ["t2m", "u10", "v10"]
        or set(payload.get("regions", {}))
        != {"sahel", "amazon", "congo", "se_aus"}
    ):
        raise ValueError(f"{path}: invalid diurnal diagnostic protocol")
    coverage = payload.get("coverage", {})
    expected_anchor_hours = [0, 6, 12, 18] if horizon == 6 else [0, 12]
    if (
        coverage.get("anchor_hours_utc") != expected_anchor_hours
        or coverage.get("months") != 12
        or coverage.get("last_anchor_interval_included") is not True
        or coverage.get("time_basis") != "approximate local solar time"
        or len(coverage.get("month_windows", {})) != 12
    ):
        raise ValueError(f"{path}: incomplete diurnal coverage")
    provenance = payload.get("model_provenance", {})
    if model_name == "linear":
        if provenance != {"kind": "linear"}:
            raise ValueError(f"{path}: invalid linear provenance")
    else:
        _validate_file_record(provenance, f"{path}: model")
        if Path(str(provenance["path"])).name != expected_weight:
            raise ValueError(f"{path}: unexpected model artifact")
    normalization_hashes = {}
    for key, record in payload.get("normalization", {}).items():
        if isinstance(record, dict) and "path" in record:
            _validate_file_record(record, f"{path}: normalization")
            normalization_hashes[key] = str(record["sha256"])
    if set(normalization_hashes) != {"stats", "surface_stats", "static_features"}:
        raise ValueError(f"{path}: incomplete normalization provenance")
    amplitude_errors: list[float] = []
    peak_errors: list[float] = []
    monthly = payload.get("monthly", {})
    expected_months = {f"2020-{month:02d}" for month in range(1, 13)}
    if set(coverage["month_windows"]) != expected_months:
        raise ValueError(f"{path}: invalid diurnal month windows")
    for month, record in coverage["month_windows"].items():
        n_hours = int(record.get("n_hours", -1))
        n_days = int(record.get("n_complete_days", -1))
        if n_days < 27 or n_hours != 24 * n_days:
            raise ValueError(f"{path}: invalid complete-day coverage for {month}")
    for region in ("sahel", "amazon", "congo", "se_aus"):
        months = monthly.get(region, {})
        if set(months) != expected_months:
            raise ValueError(f"{path}: incomplete months for {region}")
        for month, fields in months.items():
            if set(fields) != {"t2m", "u10", "v10"}:
                raise ValueError(f"{path}: incomplete fields for {region} {month}")
            for field, row in fields.items():
                ratio = _finite(
                    row.get("amp_ratio"),
                    f"{path}: {region} {month} {field} amplitude",
                )
                peak = _finite(
                    row.get("peak_hour_error"),
                    f"{path}: {region} {month} {field} peak",
                )
                if not 0.0 <= peak <= 12.0:
                    raise ValueError(f"{path}: invalid circular peak error")
                amplitude_errors.append(abs(ratio - 1.0))
                peak_errors.append(peak)
    if len(amplitude_errors) != 144:
        raise ValueError(f"{path}: expected 144 diurnal strata")
    metrics = {
        "amplitude_error": sum(amplitude_errors) / len(amplitude_errors),
        "peak_error": sum(peak_errors) / len(peak_errors),
    }
    provenance_summary = {
        "model_sha256": str(provenance.get("sha256", "linear")),
        "evaluation_dataset_identity_sha256": _validate_dataset_provenance(
            payload.get("evaluation_dataset_provenance"),
            str(path),
        ),
        **{
            f"normalization_{key}_sha256": value
            for key, value in normalization_hashes.items()
        },
    }
    return metrics, provenance_summary


def _format_best(
    value: float,
    *,
    values: list[float],
    target: float | None = None,
) -> str:
    if target is None:
        best = min(values)
    else:
        best = min(values, key=lambda current: abs(current - target))
    rendered = f"{value:.3f}"
    return f"\\textbf{{{rendered}}}" if math.isclose(value, best) else rendered


def _write_atomic(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text("\n".join(lines) + "\n")
    os.replace(temporary, path)


def export_tables(root: Path, physics_tex: Path, diurnal_tex: Path) -> dict[str, str]:
    physics: dict[int, dict[str, dict[str, float]]] = {}
    diurnal: dict[int, dict[str, dict[str, float]]] = {}
    for horizon in (6, 12):
        physics[horizon] = {}
        diurnal[horizon] = {}
        physics_provenance = None
        diurnal_provenance = None
        for stem, label in MODELS:
            internal_name = _internal_name(stem, horizon)
            expected_weight = _expected_weight_name(stem, horizon)
            physics_metrics, current_physics_provenance = _load_physics(
                root
                / f"{horizon}h"
                / "2020"
                / f"physics_{internal_name}.json",
                horizon,
                internal_name,
                expected_weight,
            )
            diurnal_metrics, current_diurnal_provenance = _load_diurnal(
                root
                / f"{horizon}h"
                / "2020"
                / f"diurnal_{internal_name}.json",
                horizon,
                internal_name,
                expected_weight,
            )
            physics[horizon][label] = physics_metrics
            diurnal[horizon][label] = diurnal_metrics
            current_physics_shared = {
                key: value
                for key, value in current_physics_provenance.items()
                if key != "model_sha256"
            }
            current_diurnal_shared = {
                key: value
                for key, value in current_diurnal_provenance.items()
                if key != "model_sha256"
            }
            if physics_provenance is None:
                physics_provenance = current_physics_shared
                diurnal_provenance = current_diurnal_shared
            elif (
                physics_provenance != current_physics_shared
                or diurnal_provenance != current_diurnal_shared
            ):
                raise ValueError(
                    f"{horizon}h downstream models use different indices or normalization"
                )
        physics_inputs = {
            key: value
            for key, value in physics_provenance.items()
            if key != "index_sha256"
        }
        if physics_inputs != diurnal_provenance:
            raise ValueError(
                f"{horizon}h physical and diurnal diagnostics use different inputs"
            )

    physics_lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Diagnostic-balance ratios relative to ERA5 over 180 uniformly spaced 2020 anchor windows and all interior hours. One is the reference value; bold marks the closest ratio within each horizon and diagnostic.}",
        "\\label{tab:physics_balance}",
        "\\small",
        "\\setlength{\\tabcolsep}{4pt}",
        "\\begin{tabular}{llrrr}",
        "\\toprule",
        "Horizon & Model & Ageo 850 & Ageo 700 & Hydrostatic \\\\",
        "\\midrule",
    ]
    for horizon in (6, 12):
        metric_values = {
            metric: [physics[horizon][label][metric] for _, label in MODELS]
            for metric in PHYSICAL_METRICS
        }
        for index, (_, label) in enumerate(MODELS):
            prefix = f"{horizon}\\,h" if index == 0 else ""
            rendered = [
                _format_best(
                    physics[horizon][label][metric],
                    values=metric_values[metric],
                    target=1.0,
                )
                for metric in PHYSICAL_METRICS
            ]
            physics_lines.append(
                f"{prefix} & {label} & " + " & ".join(rendered) + " \\\\"
            )
        if horizon == 6:
            physics_lines.append("\\midrule")
    physics_lines.extend(("\\bottomrule", "\\end{tabular}", "\\end{table}"))

    diurnal_lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Regional diurnal-cycle errors over four regions, 12 months and three surface fields (144 strata). Amplitude error is $|A_{\\mathrm{recon}}/A_{\\mathrm{ERA5}}-1|$; peak error is the circular local-time difference. Lower is better; bold marks the lowest value per horizon.}",
        "\\label{tab:diurnal}",
        "\\small",
        "\\setlength{\\tabcolsep}{5pt}",
        "\\begin{tabular}{llrr}",
        "\\toprule",
        "Horizon & Model & Amplitude error & Peak error (h) \\\\",
        "\\midrule",
    ]
    for horizon in (6, 12):
        amplitude_values = [
            diurnal[horizon][label]["amplitude_error"] for _, label in MODELS
        ]
        peak_values = [
            diurnal[horizon][label]["peak_error"] for _, label in MODELS
        ]
        for index, (_, label) in enumerate(MODELS):
            prefix = f"{horizon}\\,h" if index == 0 else ""
            diurnal_lines.append(
                f"{prefix} & {label} & "
                f"{_format_best(diurnal[horizon][label]['amplitude_error'], values=amplitude_values)} & "
                f"{_format_best(diurnal[horizon][label]['peak_error'], values=peak_values)} \\\\"
            )
        if horizon == 6:
            diurnal_lines.append("\\midrule")
    diurnal_lines.extend(("\\bottomrule", "\\end{tabular}", "\\end{table}"))
    _write_atomic(physics_tex, physics_lines)
    _write_atomic(diurnal_tex, diurnal_lines)
    return {str(physics_tex): _sha256(physics_tex), str(diurnal_tex): _sha256(diurnal_tex)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--physics-tex", type=Path, required=True)
    parser.add_argument("--diurnal-tex", type=Path, required=True)
    args = parser.parse_args()
    result = export_tables(args.root, args.physics_tex, args.diurnal_tex)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
