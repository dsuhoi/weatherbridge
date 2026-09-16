import csv
from pathlib import Path

from tools.eval.select_upr_lite_two_epoch_screen import build_screen_report


def _write_metrics(root: Path, h2: float, h4: float) -> None:
    path = root / "lightning_logs" / "version_0" / "metrics.csv"
    path.parent.mkdir(parents=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "epoch",
                "step",
                "train/recon_l1",
                "val/rmse_h2",
                "val/rmse_h4",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "epoch": 0,
                "step": 10,
                "train/recon_l1": 0.2,
            }
        )
        writer.writerow(
            {
                "epoch": 1,
                "step": 20,
                "val/rmse_h2": h2,
                "val/rmse_h4": h4,
            }
        )


def test_two_epoch_screen_uses_only_held_hours_and_keeps_top_three(
    tmp_path: Path,
) -> None:
    experiments = {
        "a": ("exp_a", 2.0),
        "b": ("exp_b", 1.0),
        "c": ("exp_c", 1.5),
        "d": ("exp_d", 0.5),
    }
    _write_metrics(tmp_path / "exp_a", 0.10, 0.20)
    _write_metrics(tmp_path / "exp_b", 0.15, 0.15)
    _write_metrics(tmp_path / "exp_c", 0.20, 0.20)
    _write_metrics(tmp_path / "exp_d", 0.30, 0.30)

    report = build_screen_report(tmp_path, experiments)

    assert report["complete"]
    assert report["ranking"] == ["b", "a", "c", "d"]
    assert report["selected"] == ["b", "a", "c"]
    assert report["screen"]["temporal_ood_2021_used"] is False


def test_two_epoch_screen_stays_incomplete_without_all_arms(
    tmp_path: Path,
) -> None:
    experiments = {
        "ready": ("exp_ready", 1.0),
        "waiting": ("exp_waiting", 1.0),
    }
    _write_metrics(tmp_path / "exp_ready", 0.1, 0.1)

    report = build_screen_report(tmp_path, experiments)

    assert not report["complete"]
    assert report["missing"] == ["waiting"]
    assert report["selected"] == []
