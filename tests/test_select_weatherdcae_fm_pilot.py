from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


SCRIPT = Path("tools/eval/select_weatherdcae_fm_pilot.py")


def _report(value: float) -> dict:
    return {
        "per_tau": {
            str(hour): {
                "model": {
                    "rmse_norm_T1000": value,
                    "rmse_norm_mslp": value,
                }
            }
            for hour in range(1, 6)
        }
    }


def test_selector_passes_only_a_broad_matched_improvement(tmp_path: Path) -> None:
    candidate = tmp_path / "candidate.json"
    control = tmp_path / "control.json"
    output = tmp_path / "selection.json"
    candidate.write_text(json.dumps(_report(0.99)))
    control.write_text(json.dumps(_report(1.0)))

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--candidate",
            str(candidate),
            "--control",
            str(control),
            "--reference",
            f"flow:{control}",
            "--output",
            str(output),
        ],
        check=False,
    )

    assert result.returncode == 0
    selection = json.loads(output.read_text())
    assert selection["continue_seed_ensemble"] is True
    assert selection["candidate_vs_matched_control"]["wins"] == 10
    assert selection["candidate_vs_flow"]["wins"] == 10


def test_selector_fails_a_mean_regression(tmp_path: Path) -> None:
    candidate = tmp_path / "candidate.json"
    control = tmp_path / "control.json"
    output = tmp_path / "selection.json"
    candidate.write_text(json.dumps(_report(1.01)))
    control.write_text(json.dumps(_report(1.0)))

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--candidate",
            str(candidate),
            "--control",
            str(control),
            "--output",
            str(output),
        ],
        check=False,
    )

    assert result.returncode == 2
    assert json.loads(output.read_text())["continue_seed_ensemble"] is False
