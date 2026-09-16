#!/usr/bin/env python3
"""Select a 2020 candidate under strict field-hour DCAE dominance gates."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def parse_named_path(value: str) -> tuple[str, Path]:
    name, separator, raw_path = value.partition(":")
    if not separator or not name or not raw_path:
        raise argparse.ArgumentTypeError("expected NAME:PATH")
    return name, Path(raw_path)


def _load_metrics(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if [int(year) for year in payload.get("years", [])] != [2020]:
        raise ValueError(f"{path}: selection metrics must contain only 2020")
    channels = [str(name) for name in payload.get("channel_names", [])]
    if len(channels) != 24:
        raise ValueError(f"{path}: expected 24 canonical channels")

    rmse: dict[str, float] = {}
    acc: dict[str, float] = {}
    for hour in range(1, 6):
        try:
            metrics = payload["per_tau"][str(hour)]["model"]
        except (KeyError, TypeError) as error:
            raise ValueError(f"{path}: missing model metrics at hour {hour}") from error
        for channel in channels:
            cell = f"h{hour}/{channel}"
            rmse_value = float(metrics[f"rmse_norm_{channel}"])
            acc_value = float(metrics[f"acc_{channel}"])
            if not math.isfinite(rmse_value) or rmse_value <= 0.0:
                raise ValueError(f"{path}: invalid RMSE at {cell}")
            if not math.isfinite(acc_value):
                raise ValueError(f"{path}: invalid ACC at {cell}")
            rmse[cell] = rmse_value
            acc[cell] = acc_value

    return {
        "path": str(path.resolve()),
        "checkpoint": payload.get("checkpoint"),
        "evaluation_index_sha256": payload.get("evaluation_protocol", {}).get(
            "index_sha256"
        ),
        "rmse": rmse,
        "acc": acc,
    }


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def select_champion(
    candidates: list[tuple[str, Path]],
    reference: tuple[str, Path],
) -> dict[str, Any]:
    if not candidates:
        raise ValueError("at least one candidate is required")
    names = [name for name, _ in candidates]
    if len(names) != len(set(names)):
        raise ValueError("candidate names must be unique")

    reference_name, reference_path = reference
    baseline = _load_metrics(reference_path)
    rows: dict[str, dict[str, Any]] = {}
    hashes = {baseline["evaluation_index_sha256"]}

    for name, path in candidates:
        candidate = _load_metrics(path)
        hashes.add(candidate["evaluation_index_sha256"])
        rmse_relative = {
            cell: candidate["rmse"][cell] / baseline["rmse"][cell] - 1.0
            for cell in baseline["rmse"]
        }
        acc_delta = {
            cell: candidate["acc"][cell] - baseline["acc"][cell]
            for cell in baseline["acc"]
        }
        rmse_regressions = {
            cell: value for cell, value in rmse_relative.items() if value >= 0.0
        }
        acc_regressions = {
            cell: value for cell, value in acc_delta.items() if value <= 0.0
        }
        held_cells = [
            cell for cell in rmse_relative if cell.startswith(("h2/", "h4/"))
        ]
        strict = not rmse_regressions and not acc_regressions
        rows[name] = {
            "path": candidate["path"],
            "checkpoint": candidate["checkpoint"],
            "rmse_wins": 120 - len(rmse_regressions),
            "acc_wins": 120 - len(acc_regressions),
            "cell_count": 120,
            "rmse_regressions": rmse_regressions,
            "acc_regressions": acc_regressions,
            "worst_rmse_relative": max(rmse_relative.values()),
            "worst_acc_delta": min(acc_delta.values()),
            "mean_rmse_relative": _mean(list(rmse_relative.values())),
            "held_rmse_relative": _mean(
                [rmse_relative[cell] for cell in held_cells]
            ),
            "per_cell_rmse_relative": rmse_relative,
            "per_cell_acc_delta": acc_delta,
            "strict_pointwise_dominance": strict,
        }

    if None in hashes or len(hashes) != 1:
        raise ValueError("candidates and reference use different evaluation indices")

    def diagnostic_rank(name: str) -> tuple[float, ...]:
        row = rows[name]
        return (
            120 - int(row["rmse_wins"]),
            max(0.0, float(row["worst_rmse_relative"])),
            120 - int(row["acc_wins"]),
            max(0.0, -float(row["worst_acc_delta"])),
            float(row["held_rmse_relative"]),
            float(row["mean_rmse_relative"]),
        )

    strict_names = [
        name for name in names if rows[name]["strict_pointwise_dominance"]
    ]
    strict_champion = (
        min(strict_names, key=lambda name: rows[name]["mean_rmse_relative"])
        if strict_names
        else None
    )
    diagnostic_winner = min(names, key=diagnostic_rank)
    return {
        "schema_version": 1,
        "selection_rule": {
            "selection_year": 2020,
            "ood_year_excluded_from_selection": 2021,
            "reference": reference_name,
            "required_rmse_wins": 120,
            "required_acc_wins": 120,
            "ties_count_as_regressions": True,
            "diagnostic_fallback": (
                "fewest RMSE regressions, smallest worst RMSE regression, "
                "fewest ACC regressions, smallest worst ACC deficit"
            ),
        },
        "reference": baseline,
        "candidates": rows,
        "strict_champion": strict_champion,
        "diagnostic_winner": diagnostic_winner,
        "winner": strict_champion or diagnostic_winner,
        "eligible_for_ood_confirmation": strict_champion is not None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", action="append", type=parse_named_path, required=True)
    parser.add_argument("--reference", type=parse_named_path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    args = parser.parse_args()
    result = select_champion(args.candidate, args.reference)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                "strict_champion": result["strict_champion"],
                "diagnostic_winner": result["diagnostic_winner"],
                "eligible_for_ood_confirmation": result[
                    "eligible_for_ood_confirmation"
                ],
            }
        )
    )


if __name__ == "__main__":
    main()
