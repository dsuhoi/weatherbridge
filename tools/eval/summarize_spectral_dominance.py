#!/usr/bin/env python3
"""Summarize field-hour spectral dominance from cellwise paired artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


METRICS = {
    "energy_log_error": "lower",
    "shape_log_error": "lower",
    "coherence": "higher",
    "signed_cospectrum": "higher",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def holm_adjust(p_values: list[float]) -> list[float]:
    """Holm-Bonferroni adjustment for one predeclared hypothesis family."""
    order = np.argsort(np.asarray(p_values, dtype=np.float64))
    adjusted = np.empty(len(p_values), dtype=np.float64)
    running = 0.0
    count = len(p_values)
    for rank, index in enumerate(order):
        running = max(running, (count - rank) * p_values[int(index)])
        adjusted[int(index)] = min(1.0, running)
    return adjusted.tolist()


def summarize(
    root: Path,
    *,
    years: list[int],
    taus: list[int],
    pattern: str,
    reference: str,
) -> dict[str, Any]:
    yearly: dict[str, Any] = {}
    input_sha256: dict[str, str] = {}
    for year in years:
        metrics = {
            name: {
                "wins": 0,
                "significant_wins_holm_global": 0,
                "failures": [],
            }
            for name in METRICS
        }
        records: dict[str, list[dict[str, Any]]] = {
            name: [] for name in METRICS
        }
        channel_names: set[str] = set()
        expected_channels: int | None = None
        for tau in taus:
            path = root / pattern.format(year=year, tau=tau)
            payload = json.loads(path.read_text())
            input_sha256[str(path)] = _sha256(path)
            left_path = Path(payload.get("left_path", ""))
            if not left_path.is_file() or payload.get("left_sha256") != _sha256(
                left_path
            ):
                raise ValueError(f"{path}: stale spectral left input")
            right_hash = payload.get("right_sha256", {}).get(reference)
            right_path = Path(
                payload.get("right_paths", {}).get(reference, "")
            )
            if (
                not right_path.is_file()
                or not isinstance(right_hash, str)
                or right_hash != _sha256(right_path)
            ):
                raise ValueError(f"{path}: stale spectral right input")
            try:
                comparison = payload["comparisons"][reference]
                per_channel = comparison["per_channel"]
            except (KeyError, TypeError) as error:
                raise ValueError(f"{path}: missing cellwise comparison") from error
            if expected_channels is None:
                expected_channels = len(per_channel)
                if expected_channels < 1:
                    raise ValueError(f"{path}: empty channel family")
            elif len(per_channel) != expected_channels:
                raise ValueError(f"{path}: inconsistent channel count")
            channel_names.update(per_channel)
            for channel, channel_metrics in per_channel.items():
                for metric_name, direction in METRICS.items():
                    metric = channel_metrics[metric_name]
                    delta = float(metric["delta_left_minus_right"])
                    better = delta < 0.0 if direction == "lower" else delta > 0.0
                    raw_p = float(metric["p_paired_block_permutation"])
                    records[metric_name].append(
                        {
                            "tau": tau,
                            "channel": channel,
                            "delta_left_minus_right": delta,
                            "better": better,
                            "p_raw": raw_p,
                            "p_holm_within_tau": float(metric["p_holm"]),
                        }
                    )
                    if better:
                        metrics[metric_name]["wins"] += 1
        assert expected_channels is not None
        expected = expected_channels * len(taus)
        if len(channel_names) != expected_channels:
            raise ValueError(f"year {year}: inconsistent channel set")
        for metric_name, metric in metrics.items():
            metric_records = records[metric_name]
            adjusted = holm_adjust([record["p_raw"] for record in metric_records])
            for record, p_global in zip(metric_records, adjusted):
                record["p_holm_global_tau_channel"] = p_global
                if record["better"] and p_global < 0.05:
                    metric["significant_wins_holm_global"] += 1
                if not record["better"]:
                    metric["failures"].append(
                        {
                            key: value
                            for key, value in record.items()
                            if key != "better"
                        }
                    )
            metric["cell_count"] = expected
            metric["all_pointwise_left_better"] = metric["wins"] == expected
            metric["all_left_better_holm_global"] = (
                metric["significant_wins_holm_global"] == expected
            )
            metric["multiplicity_family"] = {
                "method": "Holm-Bonferroni",
                "dimensions": "tau_x_channel",
                "n_hypotheses": expected,
                "alpha": 0.05,
            }
        yearly[str(year)] = metrics

    strict = all(
        yearly[str(year)][metric]["all_pointwise_left_better"]
        for year in years
        for metric in METRICS
    )
    return {
        "schema_version": 2,
        "reference": reference,
        "years": years,
        "taus": taus,
        "metrics": METRICS,
        "yearly": yearly,
        "strict_pointwise_spectral_dominance": strict,
        "input_sha256": input_sha256,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--years", default="2020,2021")
    parser.add_argument("--taus", default="1,2,3,4,5")
    parser.add_argument(
        "--pattern",
        default="{year}/spectra/paired_refine_vs_references_tau{tau}.json",
    )
    parser.add_argument("--reference", required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(
        args.root,
        years=[int(value) for value in args.years.split(",")],
        taus=[int(value) for value in args.taus.split(",")],
        pattern=args.pattern,
        reference=args.reference,
    )
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"strict": result["strict_pointwise_spectral_dominance"]}))


if __name__ == "__main__":
    main()
