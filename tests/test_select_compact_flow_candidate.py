import csv
from pathlib import Path

from tools.eval.select_compact_flow_candidate import select_candidate


def _write_curve(path: Path, epochs: list[list[float]]) -> None:
    fieldnames = [
        "epoch",
        "step",
        "val/recon_l1",
        *(f"val/rmse_h{hour}" for hour in range(1, 6)),
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for epoch, values in enumerate(epochs):
            writer.writerow(
                {
                    "epoch": epoch,
                    "step": (epoch + 1) * 1642 - 1,
                    "val/recon_l1": 0.1,
                    **{
                        f"val/rmse_h{hour}": value
                        for hour, value in enumerate(values, start=1)
                    },
                }
            )


def test_selects_best_eligible_short_budget_candidate(
    tmp_path: Path,
) -> None:
    reference = tmp_path / "reference.csv"
    small = tmp_path / "small.csv"
    medium = tmp_path / "medium.csv"
    _write_curve(reference, [[1.0] * 5, [1.0] * 5])
    _write_curve(small, [[1.0] * 5, [0.99, 0.98, 0.99, 0.98, 0.99]])
    _write_curve(
        medium,
        [[1.0] * 5, [0.97, 0.95, 0.97, 0.95, 0.97]],
    )

    report = select_candidate(
        {"small": small, "medium": medium},
        reference,
        reference_name="flow",
        epoch=1,
        held_relative_limit=0.03,
        all_hour_relative_limit=0.03,
        per_hour_relative_limit=0.08,
    )

    assert report["ranking"] == ["medium", "small"]
    assert report["eligible"] == ["small", "medium"]
    assert report["selected"] == "medium"
    assert report["candidates"]["medium"]["held_relative_delta"] < 0


def test_excludes_candidate_that_fails_per_hour_guard(
    tmp_path: Path,
) -> None:
    reference = tmp_path / "reference.csv"
    uneven = tmp_path / "uneven.csv"
    _write_curve(reference, [[1.0] * 5, [1.0] * 5])
    _write_curve(
        uneven,
        [[1.0] * 5, [1.09, 0.90, 0.90, 0.90, 0.90]],
    )

    report = select_candidate(
        {"uneven": uneven},
        reference,
        reference_name="flow",
        epoch=1,
        held_relative_limit=0.03,
        all_hour_relative_limit=0.03,
        per_hour_relative_limit=0.08,
    )

    assert report["selected"] is None
    assert not report["candidates"]["uneven"]["checks"][
        "per_hour_relative_delta_max"
    ]
