#!/usr/bin/env python3
"""Freeze a validation-only distillation decision before final diagnostics."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_model(value: str) -> tuple[str, Path]:
    name, separator, raw_path = value.partition(":")
    if not separator or not name or not raw_path:
        raise argparse.ArgumentTypeError("expected NAME:CHECKPOINT")
    return name, Path(raw_path)


def freeze_distillation_selection(
    selection_path: Path,
    models: list[tuple[str, Path]],
) -> dict[str, Any]:
    selection = json.loads(selection_path.read_text())
    selected = selection.get("selected")
    candidates = selection.get("candidates", {})
    if (
        selection.get("status") != "pass"
        or selection.get("selection_role") != "era5_2020_validation_only"
        or not isinstance(selected, str)
        or selected not in candidates
        or candidates[selected].get("pass") is not True
    ):
        raise ValueError("distillation selection did not pass the 2020 gate")
    names = [name for name, _ in models]
    if "distilled_student" not in names:
        raise ValueError("frozen models must include distilled_student")
    if len(names) != len(set(names)):
        raise ValueError("frozen model names must be unique")
    frozen_models: dict[str, dict[str, Any]] = {}
    for name, checkpoint in models:
        if not checkpoint.is_file() or checkpoint.stat().st_size <= 0:
            raise ValueError(f"missing or empty checkpoint: {checkpoint}")
        frozen_models[name] = {
            "checkpoint_path": str(checkpoint.resolve()),
            "checkpoint_sha256": _sha256(checkpoint),
            "checkpoint_size_bytes": checkpoint.stat().st_size,
        }
    return {
        "schema_version": 12,
        "winner": "distilled_student",
        "winner_variant_in_source_selection": selected,
        "ood_attached_at_selection_time": False,
        "selection_year": 2020,
        "ood_year": 2021,
        "selection_rule": {
            "selection_year": 2020,
            "ood_year_excluded_from_selection": 2021,
            "role": "era5_2020_validation_only",
        },
        "selection_source": {
            "path": str(selection_path.resolve()),
            "sha256": _sha256(selection_path),
        },
        "models": frozen_models,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--model", action="append", type=parse_model, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = freeze_distillation_selection(args.selection, args.model)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(args.output)
    print(json.dumps({
        "winner": payload["winner"],
        "variant": payload["winner_variant_in_source_selection"],
        "models": list(payload["models"]),
    }))


if __name__ == "__main__":
    main()
