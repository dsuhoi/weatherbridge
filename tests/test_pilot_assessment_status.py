import csv
import json
from pathlib import Path

from tools.eval.assess_flow_spherical_ep_pilot import (
    assess as assess_spherical,
)
from tools.eval.assess_two_epoch_candidate import assess as assess_generic
from tools.eval.pilot_assessment_status import validate_pilot_assessment


def _write_curve(path: Path, values: tuple[float, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "epoch",
        "step",
        "val/recon_l1",
        *(f"val/rmse_h{hour}" for hour in range(1, 6)),
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for epoch, scale in ((0, 1.1), (1, 1.0)):
            writer.writerow(
                {
                    "epoch": epoch,
                    "step": (epoch + 1) * 1642 - 1,
                    "val/recon_l1": sum(values) * scale / len(values),
                    **{
                        f"val/rmse_h{hour}": value * scale
                        for hour, value in enumerate(values, start=1)
                    },
                }
            )


def _generic_artifact(tmp_path: Path) -> tuple[Path, Path, Path]:
    candidate = tmp_path / "candidate.csv"
    reference = tmp_path / "reference.csv"
    artifact = tmp_path / "assessment.json"
    _write_curve(reference, (0.06, 0.09, 0.10, 0.09, 0.06))
    _write_curve(candidate, (0.061, 0.091, 0.101, 0.091, 0.061))
    artifact.write_text(
        json.dumps(
            assess_generic(
                candidate,
                reference,
                candidate_name="candidate",
                reference_name="reference",
                held_relative_limit=0.02,
                all_hour_relative_limit=0.02,
                per_hour_relative_limit=0.05,
            )
        )
    )
    return artifact, candidate, reference


def test_recomputes_valid_generic_decision(tmp_path: Path) -> None:
    artifact, _, _ = _generic_artifact(tmp_path)

    result = validate_pilot_assessment(
        artifact,
        expected_candidate="candidate",
        expected_reference="reference",
    )

    assert result == {"valid": True, "promoted": True, "errors": []}


def test_rejects_tampered_decision(tmp_path: Path) -> None:
    artifact, _, _ = _generic_artifact(tmp_path)
    payload = json.loads(artifact.read_text())
    payload["promoted"] = False
    artifact.write_text(json.dumps(payload))

    result = validate_pilot_assessment(
        artifact,
        expected_candidate="candidate",
        expected_reference="reference",
    )

    assert result["valid"] is False
    assert any("promoted" in error for error in result["errors"])


def test_rejects_changed_metrics_input(tmp_path: Path) -> None:
    artifact, candidate, _ = _generic_artifact(tmp_path)
    candidate.write_text(candidate.read_text().replace("0.061", "0.071"))

    result = validate_pilot_assessment(
        artifact,
        expected_candidate="candidate",
        expected_reference="reference",
    )

    assert result["valid"] is False
    assert any("changed" in error for error in result["errors"])


def test_ignores_unbound_resume_metrics(tmp_path: Path) -> None:
    candidate = (
        tmp_path
        / "candidate"
        / "lightning_logs"
        / "version_0"
        / "metrics.csv"
    )
    reference = (
        tmp_path
        / "reference"
        / "lightning_logs"
        / "version_0"
        / "metrics.csv"
    )
    _write_curve(reference, (0.06, 0.09, 0.10, 0.09, 0.06))
    _write_curve(candidate, (0.061, 0.091, 0.101, 0.091, 0.061))
    resume = candidate.parents[1] / "version_1" / "metrics.csv"
    resume.parent.mkdir(parents=True)
    resume.write_text("epoch,step,train/loss\n2,3300,0.1\n")
    artifact = tmp_path / "assessment.json"
    artifact.write_text(
        json.dumps(
            assess_generic(
                candidate,
                reference,
                candidate_name="candidate",
                reference_name="reference",
                held_relative_limit=0.02,
                all_hour_relative_limit=0.02,
                per_hour_relative_limit=0.05,
            )
        )
    )
    resume.write_text("epoch,step,train/loss\n2,3400,0.09\n")

    result = validate_pilot_assessment(
        artifact,
        expected_candidate="candidate",
        expected_reference="reference",
    )

    assert result["valid"] is True


def test_non_spherical_candidate_cannot_use_geometry_rescue(
    tmp_path: Path,
) -> None:
    candidate = tmp_path / "candidate.csv"
    reference = tmp_path / "reference.csv"
    artifact = tmp_path / "assessment.json"
    _write_curve(reference, (0.06, 0.09, 0.10, 0.09, 0.06))
    _write_curve(candidate, (0.061, 0.091, 0.101, 0.091, 0.061))
    payload = assess_spherical(
        candidate,
        reference,
        pilot_name="flow_pp3_hf",
        reference_name="upr_implicit_global_14m",
    )
    payload["geometry_rescue_inputs"] = {"forged": True}
    artifact.write_text(json.dumps(payload))

    result = validate_pilot_assessment(
        artifact,
        expected_candidate="flow_pp3_hf",
        expected_reference="upr_implicit_global_14m",
    )

    assert result["valid"] is False
    assert any("forbidden" in error for error in result["errors"])


def test_global_gate_does_not_depend_on_polar_diagnostic_freshness(
    tmp_path: Path,
) -> None:
    candidate = tmp_path / "candidate.csv"
    reference = tmp_path / "reference.csv"
    artifact = tmp_path / "assessment.json"
    _write_curve(reference, (0.06, 0.09, 0.10, 0.09, 0.06))
    _write_curve(candidate, (0.061, 0.091, 0.101, 0.091, 0.061))
    payload = assess_spherical(
        candidate,
        reference,
        pilot_name="spherical",
        reference_name="reference",
    )
    payload["geometry_rescue_inputs"] = {
        "pilot_metrics": {"path": str(tmp_path / "stale.json")},
    }
    payload["observed"]["pilot_polar_rmse"] = 0.1
    artifact.write_text(json.dumps(payload))

    result = validate_pilot_assessment(
        artifact,
        expected_candidate="spherical",
        expected_reference="reference",
        allow_geometry_rescue=True,
    )

    assert result == {"valid": True, "promoted": True, "errors": []}


def test_geometry_rescue_still_requires_bound_polar_inputs(
    tmp_path: Path,
) -> None:
    candidate = tmp_path / "candidate.csv"
    reference = tmp_path / "reference.csv"
    artifact = tmp_path / "assessment.json"
    _write_curve(reference, (0.06, 0.09, 0.10, 0.09, 0.06))
    _write_curve(candidate, (0.061, 0.095, 0.101, 0.095, 0.061))
    payload = assess_spherical(
        candidate,
        reference,
        pilot_name="spherical",
        reference_name="reference",
    )
    payload["promotion_path"] = "polar_geometry_rescue"
    payload["promoted"] = True
    payload["geometry_rescue_inputs"] = {
        "pilot_metrics": {"path": str(tmp_path / "missing.json")},
    }
    artifact.write_text(json.dumps(payload))

    result = validate_pilot_assessment(
        artifact,
        expected_candidate="spherical",
        expected_reference="reference",
        allow_geometry_rescue=True,
    )

    assert result["valid"] is False
    assert any("geometry_rescue_inputs" in error for error in result["errors"])
