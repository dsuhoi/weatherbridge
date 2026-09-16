import csv
from pathlib import Path

from tools.eval.assess_two_epoch_candidate import assess


def _write_curve(path: Path, epoch_one_hours: tuple[float, ...]) -> None:
    fields = [
        "epoch",
        "step",
        "val/recon_l1",
        *(f"val/rmse_h{hour}" for hour in range(1, 6)),
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for epoch, values in (
            (0, tuple(value * 1.1 for value in epoch_one_hours)),
            (1, epoch_one_hours),
        ):
            writer.writerow(
                {
                    "epoch": epoch,
                    "step": (epoch + 1) * 1642 - 1,
                    "val/recon_l1": sum(values) / len(values),
                    **{
                        f"val/rmse_h{hour}": values[hour - 1]
                        for hour in range(1, 6)
                    },
                }
            )


def _assess(candidate: Path, reference: Path) -> dict:
    return assess(
        candidate,
        reference,
        candidate_name="candidate",
        reference_name="reference",
        held_relative_limit=0.05,
        all_hour_relative_limit=0.05,
        per_hour_relative_limit=0.10,
    )


def test_generic_gate_promotes_candidate_within_limits(
    tmp_path: Path,
) -> None:
    candidate = tmp_path / "candidate.csv"
    reference = tmp_path / "reference.csv"
    _write_curve(reference, (0.06, 0.09, 0.10, 0.09, 0.06))
    _write_curve(candidate, (0.062, 0.093, 0.104, 0.093, 0.062))

    report = _assess(candidate, reference)

    assert report["promoted"] is True
    assert report["candidate"] == "candidate"
    assert all(report["checks"].values())


def test_generic_gate_rejects_worst_hour_regression(
    tmp_path: Path,
) -> None:
    candidate = tmp_path / "candidate.csv"
    reference = tmp_path / "reference.csv"
    _write_curve(reference, (0.06, 0.09, 0.10, 0.09, 0.06))
    _write_curve(candidate, (0.067, 0.09, 0.10, 0.09, 0.06))

    report = _assess(candidate, reference)

    assert report["promoted"] is False
    assert report["checks"]["per_hour_relative_delta_max"] is False


def test_gate_ignores_mutating_resume_sibling(tmp_path: Path) -> None:
    candidate_dir = tmp_path / "candidate" / "lightning_logs"
    reference_dir = tmp_path / "reference" / "lightning_logs"
    candidate = candidate_dir / "version_0" / "metrics.csv"
    reference = reference_dir / "version_0" / "metrics.csv"
    candidate.parent.mkdir(parents=True)
    reference.parent.mkdir(parents=True)
    _write_curve(reference, (0.06, 0.09, 0.10, 0.09, 0.06))
    _write_curve(candidate, (0.061, 0.091, 0.101, 0.091, 0.061))
    active_resume = candidate_dir / "version_1" / "metrics.csv"
    active_resume.parent.mkdir(parents=True)
    active_resume.write_text("epoch,step,train/loss\n2,3300,0.1\n")

    report = _assess(candidate, reference)
    active_resume.write_text("epoch,step,train/loss\n2,3320,0.09\n")
    repeated = _assess(candidate, reference)

    assert repeated == report
    assert len(report["comparison"]["input_sha256"]["candidate"]) == 1
