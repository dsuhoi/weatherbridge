import csv
from pathlib import Path

import pytest

from tools.eval.summarize_matched_training_curves import build_report


def _write_curve(
    path: Path,
    held_by_epoch: tuple[float, ...],
    *,
    start_epoch: int = 0,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "epoch",
        "step",
        "val/recon_l1",
        *(f"val/rmse_h{hour}" for hour in range(1, 6)),
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for epoch, held in enumerate(held_by_epoch, start=start_epoch):
            writer.writerow(
                {
                    "epoch": epoch,
                    "step": (epoch + 1) * 10 - 1,
                    "val/recon_l1": held,
                    **{
                        f"val/rmse_h{hour}": (
                            held if hour in (2, 4) else held * 0.8
                        )
                        for hour in range(1, 6)
                    },
                }
            )


def test_reports_matched_epochs_and_final_reference_crossing(
    tmp_path: Path,
) -> None:
    left = tmp_path / "left.csv"
    right = tmp_path / "right.csv"
    _write_curve(left, (0.95, 0.79))
    _write_curve(right, (1.0, 0.9, 0.8))

    report = build_report(
        left,
        right,
        left_name="upr",
        right_name="pp3",
        steps_per_epoch=10,
    )

    assert list(report["matched_epochs"]) == ["0", "1"]
    assert report["matched_epochs"]["1"]["held_relative_delta"] == pytest.approx(
        0.79 / 0.9 - 1.0
    )
    assert report["left_first_epoch_better_than_reference_final"] == 1
    assert report["diagnostic_only"] is True


def test_auto_merges_resumed_lightning_logger_versions(
    tmp_path: Path,
) -> None:
    logger = tmp_path / "left" / "lightning_logs"
    left_v0 = logger / "version_0" / "metrics.csv"
    left_v1 = logger / "version_1" / "metrics.csv"
    right = tmp_path / "right.csv"
    _write_curve(left_v0, (0.95, 0.9))
    _write_curve(left_v1, (0.79,), start_epoch=2)
    _write_curve(right, (1.0, 0.9, 0.8))

    report = build_report(
        left_v0,
        right,
        left_name="upr",
        right_name="pp3",
        steps_per_epoch=10,
    )

    assert report["schema_version"] == 2
    assert report["completed_epochs"]["upr"] == [0, 1, 2]
    assert list(report["matched_epochs"]) == ["0", "1", "2"]
    assert len(report["input_sha256"]["upr"]) == 2
    assert report["input_sha256"]["upr"][1]["path"] == str(
        left_v1.resolve()
    )


def test_rejects_unmatched_epoch_step(tmp_path: Path) -> None:
    left = tmp_path / "left.csv"
    right = tmp_path / "right.csv"
    _write_curve(left, (0.9,))
    _write_curve(right, (1.0,))
    rows = left.read_text().replace(",9,", ",8,")
    left.write_text(rows)

    with pytest.raises(ValueError, match="step 8"):
        build_report(
            left,
            right,
            left_name="upr",
            right_name="pp3",
            steps_per_epoch=10,
        )
