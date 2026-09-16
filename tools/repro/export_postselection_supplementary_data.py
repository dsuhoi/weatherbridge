#!/usr/bin/env python3
"""Export the frozen 2022 WeatherBridge assessment as Supplementary Data."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def export(
    assessment_path: Path,
    verification_path: Path,
    champion_path: Path,
    holdout_manifest_path: Path,
    out_json: Path,
    out_manifest: Path,
) -> dict[str, Any]:
    assessment = json.loads(assessment_path.read_text())
    if assessment.get("schema_version") != 1:
        raise ValueError("unsupported post-selection assessment schema")
    if assessment.get("status") != "confirmed_aggregate":
        raise ValueError("post-selection assessment is not aggregate-confirmed")
    if assessment.get("evaluated_candidate") != "flow_spectral":
        raise ValueError("assessment does not evaluate canonical WeatherBridge")
    if assessment.get("reference") != "weatherdcae_14m":
        raise ValueError("assessment reference is not WeatherDCAE-14M")

    bound_inputs = {
        "assessment": (assessment_path, None),
        "verification": (
            verification_path,
            assessment.get("verification_sha256"),
        ),
        "champion": (champion_path, assessment.get("champion_sha256")),
        "holdout_manifest": (
            holdout_manifest_path,
            assessment.get("manifest_sha256"),
        ),
    }
    source_sha256: dict[str, str] = {}
    for label, (path, expected) in bound_inputs.items():
        actual = _sha256(path)
        if expected is not None and actual != expected:
            raise ValueError(f"{label} hash does not match the assessment")
        source_sha256[str(path)] = actual

    payload = {
        "schema_version": 1,
        "status": "complete",
        "title": "Supplementary Data 2: frozen 2022 post-selection assessment",
        "candidate": {
            "public_name": "WeatherBridge",
            "internal_id": assessment["evaluated_candidate"],
        },
        "reference": {
            "public_name": "WeatherDCAE-14M",
            "internal_id": assessment["reference"],
        },
        "frozen_selector": {
            "winner": assessment["frozen_winner"],
            "selector_confirmation": assessment["selector_confirmation"],
            "selection_was_frozen_before_holdout": assessment[
                "selection_was_frozen_before_holdout"
            ],
        },
        "aggregate_confirmation_pass": assessment[
            "aggregate_confirmation_pass"
        ],
        "universal_dominance_claim_allowed": assessment[
            "universal_dominance_claim_allowed"
        ],
        "publication_claim": assessment["publication_claim"],
        "horizons": assessment["horizons"],
        "source_sha256": source_sha256,
    }
    _write_json(out_json, payload)

    manifest = {
        "schema_version": 1,
        "status": "complete",
        "data_sha256": _sha256(out_json),
        "exporter_sha256": _sha256(Path(__file__).resolve()),
        "source_sha256": source_sha256,
    }
    _write_json(out_manifest, manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assessment", type=Path, required=True)
    parser.add_argument("--verification", type=Path, required=True)
    parser.add_argument("--champion", type=Path, required=True)
    parser.add_argument("--holdout-manifest", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-manifest", type=Path, required=True)
    args = parser.parse_args()
    result = export(
        args.assessment,
        args.verification,
        args.champion,
        args.holdout_manifest,
        args.out_json,
        args.out_manifest,
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
