import csv
from pathlib import Path

import pytest

from tools.train.select_validation_metrics_csv import (
    select_validation_metrics_csv,
)


def _write_metrics(
    run_dir: Path,
    version: int,
    rows: list[dict[str, object]],
) -> Path:
    path = (
        run_dir
        / "lightning_logs"
        / f"version_{version}"
        / "metrics.csv"
    )
    path.parent.mkdir(parents=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("epoch", "step", "val/rmse_mean"),
        )
        writer.writeheader()
        writer.writerows(rows)
    return path


def test_selects_resumed_version_with_required_validation(tmp_path) -> None:
    run_dir = tmp_path / "experiment"
    _write_metrics(
        run_dir,
        0,
        [{"epoch": 0, "step": 1641, "val/rmse_mean": 0.09}],
    )
    expected = _write_metrics(
        run_dir,
        1,
        [{"epoch": 1, "step": 3283, "val/rmse_mean": 0.08}],
    )
    _write_metrics(
        run_dir,
        2,
        [{"epoch": 2, "step": 3300, "val/rmse_mean": ""}],
    )

    selected = select_validation_metrics_csv(
        run_dir,
        required_epoch=1,
        required_step=3283,
    )

    assert selected == expected


def test_prefers_latest_version_containing_same_epoch(tmp_path) -> None:
    run_dir = tmp_path / "experiment"
    _write_metrics(
        run_dir,
        0,
        [{"epoch": 1, "step": 3283, "val/rmse_mean": 0.08}],
    )
    expected = _write_metrics(
        run_dir,
        3,
        [{"epoch": 1, "step": 3283, "val/rmse_mean": 0.079}],
    )

    assert (
        select_validation_metrics_csv(
            run_dir,
            required_epoch=1,
            required_step=3283,
        )
        == expected
    )


def test_rejects_missing_required_validation(tmp_path) -> None:
    run_dir = tmp_path / "experiment"
    _write_metrics(
        run_dir,
        0,
        [{"epoch": 1, "step": 3283, "val/rmse_mean": ""}],
    )

    with pytest.raises(FileNotFoundError, match="validation epoch 1"):
        select_validation_metrics_csv(
            run_dir,
            required_epoch=1,
            required_step=3283,
        )
