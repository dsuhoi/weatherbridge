import hashlib
import json
from pathlib import Path

import pytest

from tools.eval.check_validation_gate import check_gate


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _artifacts(tmp_path: Path) -> tuple[Path, Path]:
    selection_path = tmp_path / "selection.json"
    selection_path.write_text(
        json.dumps({"schema_version": 12, "winner": "candidate"}) + "\n"
    )
    validation_path = tmp_path / "validation.json"
    validation_path.write_text(
        json.dumps(
            {
                "schema_version": 11,
                "selection_report_sha256": _sha256(selection_path),
                "winner": "candidate",
                "summary": {
                    "selection_confirmed": True,
                    "cross_metric_generalization_confirmed": True,
                    "absolute_ood_robust_skill_passed": True,
                },
            }
        )
        + "\n"
    )
    return validation_path, selection_path


def test_gate_requires_cross_metric_confirmation(tmp_path: Path) -> None:
    validation_path, selection_path = _artifacts(tmp_path)

    result = check_gate(validation_path, selection_path)

    assert result["winner"] == "candidate"
    payload = json.loads(validation_path.read_text())
    payload["summary"]["cross_metric_generalization_confirmed"] = False
    validation_path.write_text(json.dumps(payload))
    with pytest.raises(
        ValueError,
        match="cross_metric_generalization_confirmed",
    ):
        check_gate(validation_path, selection_path)


def test_gate_rejects_selection_drift(tmp_path: Path) -> None:
    validation_path, selection_path = _artifacts(tmp_path)
    selection_path.write_text(
        json.dumps({"schema_version": 12, "winner": "other"}) + "\n"
    )

    with pytest.raises(ValueError, match="does not bind"):
        check_gate(validation_path, selection_path)


def test_gate_rejects_pre_extreme_regression_schema(
    tmp_path: Path,
) -> None:
    validation_path, selection_path = _artifacts(tmp_path)
    payload = json.loads(validation_path.read_text())
    payload["schema_version"] = 10
    validation_path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="unsupported validation schema"):
        check_gate(validation_path, selection_path)


def test_gate_rejects_pre_robust_selection_schema(
    tmp_path: Path,
) -> None:
    validation_path, selection_path = _artifacts(tmp_path)
    payload = json.loads(selection_path.read_text())
    payload["schema_version"] = 11
    selection_path.write_text(json.dumps(payload))

    with pytest.raises(
        ValueError,
        match="unsupported frozen selection schema",
    ):
        check_gate(validation_path, selection_path)


def test_gate_requires_absolute_ood_robust_skill(
    tmp_path: Path,
) -> None:
    validation_path, selection_path = _artifacts(tmp_path)
    payload = json.loads(validation_path.read_text())
    payload["summary"]["absolute_ood_robust_skill_passed"] = False
    validation_path.write_text(json.dumps(payload))

    with pytest.raises(
        ValueError,
        match="absolute_ood_robust_skill_passed",
    ):
        check_gate(validation_path, selection_path)
