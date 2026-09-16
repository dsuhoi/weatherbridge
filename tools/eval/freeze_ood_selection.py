#!/usr/bin/env python3
"""Freeze a 2020-only model decision before any OOD evaluation."""

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


def freeze_selection(
    selection_path: Path,
    *,
    winner: str,
    models: list[tuple[str, Path]],
) -> dict[str, Any]:
    selection = json.loads(selection_path.read_text())
    if not isinstance(selection, dict):
        raise ValueError("selection must be a JSON object")
    rule = selection.get("selection_rule", {})
    if (
        rule.get("selection_year") != 2020
        or rule.get("ood_year_excluded_from_selection") != 2021
    ):
        raise ValueError("selection must be frozen on 2020 with 2021 excluded")
    names = [name for name, _ in models]
    if not names or len(names) != len(set(names)):
        raise ValueError("model names must be non-empty and unique")
    if winner not in names:
        raise ValueError("winner must be present in the frozen model set")

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
        "winner": winner,
        "winner_variant_in_source_selection": selection.get("winner"),
        "ood_attached_at_selection_time": False,
        "selection_year": 2020,
        "ood_year": 2021,
        "selection_source": {
            "path": str(selection_path.resolve()),
            "sha256": _sha256(selection_path),
            "schema_version": selection.get("schema_version"),
        },
        "models": frozen_models,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--winner", required=True)
    parser.add_argument("--model", action="append", type=parse_model, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = freeze_selection(
        args.selection,
        winner=args.winner,
        models=args.model,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(args.output)
    print(json.dumps({"winner": payload["winner"], "models": list(payload["models"])}))


if __name__ == "__main__":
    main()
