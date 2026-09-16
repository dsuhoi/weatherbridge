from __future__ import annotations

import hashlib
import json
from pathlib import Path

from tools.repro.export_postselection_supplementary_data import export


def _write_json(path: Path, payload: dict[str, object]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload) + "\n")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_export_binds_public_name_and_frozen_inputs(tmp_path: Path) -> None:
    verification = tmp_path / "verification.json"
    champion = tmp_path / "champion.json"
    holdout = tmp_path / "holdout.json"
    verification_hash = _write_json(verification, {"verified": True})
    champion_hash = _write_json(champion, {"winner": "weatherdcae_14m"})
    holdout_hash = _write_json(holdout, {"year": 2022})
    assessment = tmp_path / "assessment.json"
    _write_json(
        assessment,
        {
            "schema_version": 1,
            "status": "confirmed_aggregate",
            "evaluated_candidate": "flow_spectral",
            "reference": "weatherdcae_14m",
            "frozen_winner": "weatherdcae_14m",
            "selector_confirmation": False,
            "selection_was_frozen_before_holdout": True,
            "aggregate_confirmation_pass": True,
            "universal_dominance_claim_allowed": False,
            "publication_claim": "aggregate only",
            "horizons": {"6h": {"relative_delta_pct": -12.6}},
            "verification_sha256": verification_hash,
            "champion_sha256": champion_hash,
            "manifest_sha256": holdout_hash,
        },
    )
    output = tmp_path / "data.json"
    manifest = tmp_path / "data.manifest.json"

    result = export(
        assessment,
        verification,
        champion,
        holdout,
        output,
        manifest,
    )

    payload = json.loads(output.read_text())
    assert payload["candidate"] == {
        "public_name": "WeatherBridge",
        "internal_id": "flow_spectral",
    }
    assert payload["universal_dominance_claim_allowed"] is False
    assert result["data_sha256"] == hashlib.sha256(output.read_bytes()).hexdigest()
