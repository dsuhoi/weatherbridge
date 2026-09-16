"""Fail closed unless a validation report confirms the frozen selection."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


REQUIRED_SELECTION_SCHEMA_VERSION = 12
REQUIRED_VALIDATION_SCHEMA_VERSION = 11


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_gate(
    validation_path: Path,
    selection_path: Path,
) -> dict[str, Any]:
    if not validation_path.is_file():
        raise FileNotFoundError(validation_path)
    if not selection_path.is_file():
        raise FileNotFoundError(selection_path)
    validation = json.loads(validation_path.read_text())
    selection = json.loads(selection_path.read_text())
    validation_schema = validation.get("schema_version")
    selection_schema = selection.get("schema_version")
    if (
        isinstance(validation_schema, bool)
        or not isinstance(validation_schema, int)
        or validation_schema < REQUIRED_VALIDATION_SCHEMA_VERSION
    ):
        raise ValueError("unsupported validation schema")
    if (
        isinstance(selection_schema, bool)
        or not isinstance(selection_schema, int)
        or selection_schema < REQUIRED_SELECTION_SCHEMA_VERSION
    ):
        raise ValueError("unsupported frozen selection schema")
    if validation.get("selection_report_sha256") != _sha256(selection_path):
        raise ValueError("validation does not bind the current selection")
    if validation.get("winner") != selection.get("winner"):
        raise ValueError("validation winner differs from frozen selection")
    summary = validation.get("summary")
    if not isinstance(summary, dict):
        raise TypeError("validation report lacks a summary")
    for key in (
        "selection_confirmed",
        "cross_metric_generalization_confirmed",
        "absolute_ood_robust_skill_passed",
    ):
        if summary.get(key) is not True:
            raise ValueError(f"validation gate failed: {key}")
    return {
        "winner": validation["winner"],
        "selection_report_sha256": validation[
            "selection_report_sha256"
        ],
        "selection_confirmed": True,
        "cross_metric_generalization_confirmed": True,
        "absolute_ood_robust_skill_passed": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = check_gate(args.validation, args.selection)
    except (FileNotFoundError, TypeError, ValueError) as exc:
        parser.exit(1, f"validation gate not ready: {exc}\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
