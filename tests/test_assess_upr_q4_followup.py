import csv
from pathlib import Path

import pytest

from tools.eval.assess_upr_q4_followup import assess


def _write_metrics(root: Path, h2: float, h4: float) -> None:
    path = root / "lightning_logs" / "version_0" / "metrics.csv"
    path.parent.mkdir(parents=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "epoch",
                "step",
                "val/rmse_h2",
                "val/rmse_h4",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "epoch": 1,
                "step": 3283,
                "val/rmse_h2": h2,
                "val/rmse_h4": h4,
            }
        )


def test_q4_followup_promotes_only_within_prespecified_quality_margin(
    tmp_path: Path,
) -> None:
    _write_metrics(tmp_path / "reference", 0.10, 0.10)
    _write_metrics(tmp_path / "candidate", 0.101, 0.102)

    promoted = assess(
        tmp_path,
        reference_experiment="reference",
        candidate_experiment="candidate",
        maximum_relative_rmse=1.02,
    )
    rejected = assess(
        tmp_path,
        reference_experiment="reference",
        candidate_experiment="candidate",
        maximum_relative_rmse=1.01,
    )

    assert promoted["promoted"]
    assert not rejected["promoted"]
    assert promoted["quality_ratio"] == pytest.approx(1.015)
