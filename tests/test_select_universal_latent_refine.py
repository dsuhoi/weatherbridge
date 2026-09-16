from __future__ import annotations

import csv
from pathlib import Path

from tools.eval.select_universal_latent_refine import select_candidates


FIELDS = [
    "epoch",
    "step",
    "val/rmse_h1",
    "val/rmse_h2",
    "val/rmse_h3",
    "val/rmse_h4",
    "val/rmse_h5",
    "val/rmse_mean",
]


def _write_metrics(path: Path, values: list[float]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerow(
            {
                "epoch": 7,
                "step": 13135,
                **{
                    f"val/rmse_h{hour}": values[hour - 1]
                    for hour in range(1, 6)
                },
                "val/rmse_mean": sum(values) / 5.0,
            }
        )


def test_selector_prioritizes_held_hours_without_2021_input(tmp_path: Path) -> None:
    reference = tmp_path / "reference.csv"
    standard = tmp_path / "standard.csv"
    swap = tmp_path / "swap.csv"
    _write_metrics(reference, [0.08, 0.10, 0.12, 0.10, 0.08])
    _write_metrics(standard, [0.079, 0.099, 0.119, 0.099, 0.079])
    _write_metrics(swap, [0.081, 0.096, 0.121, 0.096, 0.081])

    result = select_candidates(
        [("standard", standard), ("swap", swap)],
        ("detail", reference),
        held_weight=0.7,
        overall_regression_limit=0.005,
    )

    assert result["winner"] == "swap"
    assert result["winner_screen_pass"] is True
    assert result["selection_rule"]["ood_year_excluded_from_selection"] == 2021
