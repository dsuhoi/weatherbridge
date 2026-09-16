#!/usr/bin/env python3
"""Export every inferential test used by the journal selector to CSV."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any

FIELDS = (
    "candidate",
    "reference",
    "horizon_hours",
    "year",
    "family",
    "metric",
    "scope",
    "tau",
    "channel",
    "better",
    "left",
    "right",
    "delta_left_minus_right",
    "ci95_low",
    "ci95_median",
    "ci95_high",
    "n_windows",
    "n_blocks",
    "block_days",
    "draws",
    "p_raw_two_sided",
    "p_censored",
    "p_raw_left_better",
    "p_raw_left_worse",
    "p_holm_two_sided",
    "p_holm_left_better",
    "p_holm_left_worse",
    "family_alpha",
    "source_path",
    "source_sha256",
)

PAIRWISE_PREFIXES = {
    "paired_rmse_": "rmse",
    "paired_hard_window_": "hard_window_rmse",
    "paired_acc_": "acc",
    "paired_temporal_curvature_": "temporal_curvature",
    "paired_physical_": "physical",
}

SPECTRAL_METRICS = (
    "energy_log_error",
    "shape_log_error",
    "coherence",
    "signed_cospectrum",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_bound_path(
    path_name: str,
    expected: str,
    source_dir: Path | None,
) -> Path:
    """Resolve a hash-bound source locally or from the portable snapshot."""
    source = Path(path_name)
    if source.is_file():
        actual = _sha256(source)
        if actual != expected:
            raise ValueError(f"stale bound input {source}")
        return source
    if source_dir is not None:
        suffix = source.suffix or ".bin"
        snapshot = source_dir / f"{expected}{suffix}"
        if snapshot.is_file():
            actual = _sha256(snapshot)
            if actual != expected:
                raise ValueError(f"stale snapshot input {snapshot}")
            return snapshot
    raise FileNotFoundError(
        f"bound input is unavailable locally and has no matching snapshot: {source}"
    )


def _finite(value: Any, context: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{context}: expected a finite number")
    return result


def _optional(value: Any, context: str) -> float | str:
    if value is None:
        return ""
    return _finite(value, context)


def _holm(p_values: Iterable[float]) -> list[float]:
    values = list(p_values)
    if not values:
        return []
    if any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in values):
        raise ValueError("Holm adjustment received an invalid p-value")
    order = sorted(range(len(values)), key=values.__getitem__)
    adjusted = [1.0] * len(values)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (len(values) - rank) * values[index])
        adjusted[index] = min(1.0, running)
    return adjusted


def _context(path: Path) -> tuple[int, int]:
    text = path.as_posix()
    match = re.search(r"/(6|12)h/(2020|2021)/", text)
    if match is None:
        match = re.search(r"/(6|12)h_(2020|2021)/", text)
    if match is None:
        raise ValueError(f"cannot infer horizon/year from {path}")
    return int(match.group(1)), int(match.group(2))


def _ci(row: dict[str, Any], context: str) -> tuple[float | str, ...]:
    values = row.get("delta_ci95")
    if values is None:
        return "", "", ""
    if not isinstance(values, list) or len(values) != 3:
        raise ValueError(f"{context}: invalid 95% interval")
    return tuple(_finite(value, context) for value in values)


def _metric_values(row: dict[str, Any], context: str) -> tuple[Any, ...]:
    left = row.get("left", row.get("left_rmse"))
    right = row.get("right", row.get("right_rmse"))
    delta = row.get("delta_left_minus_right")
    return (
        _optional(left, f"{context} left"),
        _optional(right, f"{context} right"),
        _optional(delta, f"{context} delta"),
        *_ci(row, context),
    )


def _base_row(
    *,
    candidate: str,
    reference: str,
    horizon: int,
    year: int,
    family: str,
    metric: str,
    scope: str,
    tau: int | str,
    channel: str,
    better: str,
    values: tuple[Any, ...],
    metadata: dict[str, Any],
    source: Path,
    source_hash: str,
) -> dict[str, Any]:
    row = dict.fromkeys(FIELDS, "")
    row.update(
        {
            "candidate": candidate,
            "reference": reference,
            "horizon_hours": horizon,
            "year": year,
            "family": family,
            "metric": metric,
            "scope": scope,
            "tau": tau,
            "channel": channel,
            "better": better,
            "left": values[0],
            "right": values[1],
            "delta_left_minus_right": values[2],
            "ci95_low": values[3],
            "ci95_median": values[4],
            "ci95_high": values[5],
            "n_windows": metadata.get("n_windows", ""),
            "n_blocks": metadata.get("n_blocks", ""),
            "block_days": metadata.get("block_days", ""),
            "draws": metadata.get("bootstrap_draws", metadata.get("draws", "")),
            "source_path": str(source),
            "source_sha256": source_hash,
        }
    )
    return row


def _attach_p_values(target: dict[str, Any], source: dict[str, Any], context: str) -> None:
    mapping = {
        "p_paired_block_permutation": "p_raw_two_sided",
        "p_left_better_one_sided": "p_raw_left_better",
        "p_left_worse_one_sided": "p_raw_left_worse",
        "p_holm": "p_holm_two_sided",
        "p_left_better_holm": "p_holm_left_better",
        "p_left_worse_holm": "p_holm_left_worse",
    }
    for source_key, target_key in mapping.items():
        if source_key in source:
            value = _finite(source[source_key], f"{context} {source_key}")
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{context}: p-value outside [0, 1]")
            target[target_key] = value


def _annotate_p_censoring(row: dict[str, Any]) -> None:
    """Mark Monte Carlo p-values that equal the plus-one resolution floor."""
    raw = row.get("p_raw_two_sided", "")
    draws = row.get("draws", "")
    if raw == "" or draws == "":
        row["p_censored"] = ""
        return
    n_draws = int(draws)
    if n_draws <= 0:
        raise ValueError("Monte Carlo draw count must be positive")
    floor = 1.0 / (n_draws + 1.0)
    row["p_censored"] = str(
        math.isclose(float(raw), floor, rel_tol=1e-9, abs_tol=1e-15)
    ).lower()


def _pairwise_rows(
    path: Path,
    source_hash: str,
    family: str,
    *,
    context_path: Path | None = None,
) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text())
    candidate = str(payload.get("left", ""))
    comparisons = payload.get("comparisons", {})
    if not candidate or not isinstance(comparisons, dict) or not comparisons:
        raise ValueError(f"{path}: ambiguous pairwise comparison")
    rows: list[dict[str, Any]] = []
    for reference, comparison in sorted(comparisons.items()):
        if not reference or not isinstance(comparison, dict):
            raise ValueError(f"{path}: invalid pairwise comparison")
        rows.extend(
            _pairwise_comparison_rows(
                path,
                source_hash,
                family,
                candidate,
                str(reference),
                comparison,
                context_path=context_path,
            )
        )
    return rows


def _pairwise_comparison_rows(
    path: Path,
    source_hash: str,
    family: str,
    candidate: str,
    reference: str,
    comparison: dict[str, Any],
    *,
    context_path: Path | None = None,
) -> list[dict[str, Any]]:
    horizon, year = _context(context_path or path)
    row_source = context_path or path
    rows: list[dict[str, Any]] = []

    if family == "physical":
        diagnostics = comparison.get("diagnostics", {})
        if not isinstance(diagnostics, dict) or not diagnostics:
            raise ValueError(f"{path}: missing physical diagnostics")
        p_raw = [
            _finite(row["p_paired_block_permutation"], f"{path} {name}")
            for name, row in diagnostics.items()
        ]
        p_holm = _holm(p_raw)
        for (name, metric), adjusted in zip(diagnostics.items(), p_holm, strict=True):
            row = _base_row(
                candidate=candidate,
                reference=reference,
                horizon=horizon,
                year=year,
                family=family,
                metric=name,
                scope="diagnostic",
                tau="all",
                channel="",
                better=str(metric.get("better", "lower")),
                values=_metric_values(metric, f"{path} {name}"),
                metadata=comparison,
                source=row_source,
                source_hash=source_hash,
            )
            _attach_p_values(row, metric, f"{path} {name}")
            row["p_holm_two_sided"] = adjusted
            row["family_alpha"] = 0.05
            rows.append(row)
        return rows

    better = str(comparison.get("better", "lower" if family != "acc" else "higher"))
    metric_name = str(comparison.get("metric", family))
    aggregate = _base_row(
        candidate=candidate,
        reference=reference,
        horizon=horizon,
        year=year,
        family=family,
        metric=metric_name,
        scope="aggregate",
        tau="all",
        channel="",
        better=better,
        values=_metric_values(comparison, f"{path} aggregate"),
        metadata=comparison,
        source=row_source,
        source_hash=source_hash,
    )
    _attach_p_values(aggregate, comparison, f"{path} aggregate")
    rows.append(aggregate)

    for tau, metric in comparison.get("per_tau", {}).items():
        row = _base_row(
            candidate=candidate,
            reference=reference,
            horizon=horizon,
            year=year,
            family=family,
            metric=metric_name,
            scope="query_hour",
            tau=int(tau),
            channel="",
            better=better,
            values=_metric_values(metric, f"{path} tau={tau}"),
            metadata=comparison,
            source=row_source,
            source_hash=source_hash,
        )
        _attach_p_values(row, metric, f"{path} tau={tau}")
        rows.append(row)

    family_meta = comparison.get("cellwise_family", {})
    alpha = family_meta.get("alpha", 0.05)
    for tau, channels in comparison.get("per_channel_tau", {}).items():
        for channel, metric in channels.items():
            row = _base_row(
                candidate=candidate,
                reference=reference,
                horizon=horizon,
                year=year,
                family=family,
                metric=metric_name,
                scope="field_hour",
                tau=int(tau),
                channel=str(channel),
                better=better,
                values=_metric_values(metric, f"{path} tau={tau} {channel}"),
                metadata=comparison,
                source=row_source,
                source_hash=source_hash,
            )
            _attach_p_values(row, metric, f"{path} tau={tau} {channel}")
            row["family_alpha"] = _finite(alpha, f"{path} alpha")
            rows.append(row)
    return rows


def _spectral_rows(
    path: Path,
    source_hash: str,
    *,
    context_path: Path | None = None,
    source_dir: Path | None = None,
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    summary = json.loads(path.read_text())
    if summary.get("schema_version") != 2:
        raise ValueError(f"{path}: expected spectral summary schema 2")
    logical_path = context_path or path
    horizon, year = _context(logical_path)
    reference = str(summary.get("reference", ""))
    source_inputs = summary.get("input_sha256", {})
    if not reference or not isinstance(source_inputs, dict) or not source_inputs:
        raise ValueError(f"{path}: incomplete spectral provenance")
    kind_match = re.search(
        r"global_(scalar|vector)_([^/]+)_vs_", logical_path.name
    )
    if kind_match is None:
        raise ValueError(f"{path}: cannot infer spectral identity")
    kind, candidate = kind_match.groups()
    records: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = {
        metric: [] for metric in SPECTRAL_METRICS
    }
    consumed: dict[str, str] = {str(path): source_hash}
    for source_name, expected in source_inputs.items():
        source = _resolve_bound_path(str(source_name), str(expected), source_dir)
        actual = _sha256(source)
        consumed[str(source)] = actual
        payload = json.loads(source.read_text())
        comparison = payload.get("comparisons", {}).get(reference)
        if not isinstance(comparison, dict):
            raise TypeError(f"{source}: missing spectral reference {reference}")
        tau = int(comparison.get("tau", -1))
        metadata = comparison
        for channel, metrics in comparison.get("per_channel", {}).items():
            for metric_name in SPECTRAL_METRICS:
                metric = metrics.get(metric_name)
                if not isinstance(metric, dict):
                    raise TypeError(f"{source}: incomplete spectral cell")
                row = _base_row(
                    candidate=candidate,
                    reference=reference,
                    horizon=horizon,
                    year=year,
                    family=f"{kind}_spectral",
                    metric=metric_name,
                    scope="field_hour",
                    tau=tau,
                    channel=str(channel),
                    better=str(metric["better"]),
                    values=_metric_values(metric, f"{source} {channel} {metric_name}"),
                    metadata=metadata,
                    source=Path(source_name),
                    source_hash=actual,
                )
                _attach_p_values(row, metric, f"{source} {channel} {metric_name}")
                records[metric_name].append((row, metric))

    rows: list[dict[str, Any]] = []
    yearly = summary.get("yearly", {}).get(str(year), {})
    for metric_name, metric_records in records.items():
        adjusted = _holm(
            _finite(metric["p_paired_block_permutation"], metric_name)
            for _, metric in metric_records
        )
        family = yearly.get(metric_name, {}).get("multiplicity_family", {})
        expected_count = int(family.get("n_hypotheses", -1))
        if expected_count != len(metric_records):
            raise ValueError(f"{path}: spectral family size mismatch for {metric_name}")
        for (row, _), p_holm in zip(metric_records, adjusted, strict=True):
            row["p_holm_two_sided"] = p_holm
            row["family_alpha"] = _finite(family.get("alpha", 0.05), "spectral alpha")
            rows.append(row)
    return rows, consumed


def _snapshot_sources(
    consumed: dict[str, str], source_dir: Path
) -> tuple[dict[str, str], dict[str, str]]:
    source_dir.mkdir(parents=True, exist_ok=True)
    project_dir = Path.cwd().resolve()
    snapshots: dict[str, str] = {}
    originals: dict[str, str] = {}
    expected_targets = {
        source_dir / f"{expected}{Path(source_name).suffix or '.bin'}"
        for source_name, expected in consumed.items()
    }
    for stale in source_dir.iterdir():
        if stale.is_file() and stale not in expected_targets:
            stale.unlink()
    for source_name, expected in sorted(consumed.items()):
        source = Path(source_name)
        suffix = source.suffix or ".bin"
        target = source_dir / f"{expected}{suffix}"
        if not target.is_file() or _sha256(target) != expected:
            temporary = target.with_suffix(target.suffix + ".tmp")
            temporary.write_bytes(source.read_bytes())
            if _sha256(temporary) != expected:
                temporary.unlink(missing_ok=True)
                raise ValueError(f"source changed while snapshotting: {source}")
            os.replace(temporary, target)
        resolved = target.resolve()
        try:
            manifest_path = resolved.relative_to(project_dir).as_posix()
        except ValueError:
            manifest_path = str(resolved)
        snapshots[manifest_path] = expected
        originals[manifest_path] = source_name
    return snapshots, originals


def export_statistics(
    champion_path: Path,
    out_csv: Path,
    out_manifest: Path,
    source_dir: Path | None = None,
) -> dict[str, Any]:
    champion = json.loads(champion_path.read_text())
    if champion.get("schema_version") != 1:
        raise ValueError("champion JSON has an unsupported schema")
    inputs = champion.get("input_sha256", {})
    if not isinstance(inputs, dict) or not inputs:
        raise ValueError("champion JSON has no bound inputs")

    rows: list[dict[str, Any]] = []
    consumed: dict[str, str] = {}
    for path_name, expected in inputs.items():
        logical_path = Path(path_name)
        family = next(
            (
                value
                for prefix, value in PAIRWISE_PREFIXES.items()
                if logical_path.name.startswith(prefix)
            ),
            None,
        )
        is_spectral = logical_path.name.startswith(
            ("global_scalar_", "global_vector_")
        )
        if family is None and not is_spectral:
            continue
        try:
            path = _resolve_bound_path(str(path_name), str(expected), source_dir)
        except ValueError as error:
            raise ValueError(f"stale champion input {logical_path}") from error
        actual = _sha256(path)
        if family is not None:
            consumed[str(path)] = actual
            rows.extend(
                _pairwise_rows(
                    path,
                    actual,
                    family,
                    context_path=logical_path,
                )
            )
        elif is_spectral:
            spectral, spectral_sources = _spectral_rows(
                path,
                actual,
                context_path=logical_path,
                source_dir=source_dir,
            )
            rows.extend(spectral)
            consumed.update(spectral_sources)

    if not rows:
        raise ValueError("champion inputs contain no inferential statistics")
    missing_p = [
        f"{row['family']}:{row['metric']}:{row['scope']}:"
        f"tau={row['tau']}:channel={row['channel']}"
        for row in rows
        if row["p_raw_two_sided"] == ""
    ]
    if missing_p:
        raise ValueError(
            "inferential rows without a two-sided raw p-value: "
            + ", ".join(missing_p[:5])
        )
    for row in rows:
        _annotate_p_censoring(row)
    rows.sort(
        key=lambda row: (
            str(row["candidate"]),
            int(row["horizon_hours"]),
            int(row["year"]),
            str(row["family"]),
            str(row["metric"]),
            str(row["scope"]),
            str(row["tau"]),
            str(row["channel"]),
        )
    )
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    temporary_csv = out_csv.with_suffix(out_csv.suffix + ".tmp")
    with temporary_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary_csv, out_csv)

    manifest_sources = consumed
    source_original_paths: dict[str, str] = {}
    if source_dir is not None:
        manifest_sources, source_original_paths = _snapshot_sources(
            consumed, source_dir
        )

    manifest = {
        "schema_version": 1,
        "status": "complete",
        "winner": champion.get("winner"),
        "champion_sha256": _sha256(champion_path),
        "exporter_sha256": _sha256(Path(__file__).resolve()),
        "csv_path": str(out_csv),
        "csv_sha256": _sha256(out_csv),
        "row_count": len(rows),
        "source_sha256": manifest_sources,
        "source_original_paths": source_original_paths,
    }
    temporary_manifest = out_manifest.with_suffix(out_manifest.suffix + ".tmp")
    temporary_manifest.write_text(json.dumps(manifest, indent=2) + "\n")
    os.replace(temporary_manifest, out_manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--champion-json", type=Path, required=True)
    parser.add_argument("--out-csv", type=Path, required=True)
    parser.add_argument("--out-manifest", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path)
    args = parser.parse_args()
    result = export_statistics(
        args.champion_json,
        args.out_csv,
        args.out_manifest,
        args.source_dir,
    )
    print(json.dumps({"status": result["status"], "rows": result["row_count"]}))


if __name__ == "__main__":
    main()
