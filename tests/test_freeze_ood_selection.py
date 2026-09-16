from __future__ import annotations

import hashlib
import json

import pytest

from tools.eval.freeze_ood_selection import freeze_selection


def test_freeze_selection_hashes_all_checkpoints(tmp_path) -> None:
    selection = tmp_path / "selection.json"
    selection.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "winner": "internal_variant",
                "selection_rule": {
                    "selection_year": 2020,
                    "ood_year_excluded_from_selection": 2021,
                },
            }
        )
    )
    candidate = tmp_path / "candidate.ckpt"
    reference = tmp_path / "reference.ckpt"
    candidate.write_bytes(b"candidate")
    reference.write_bytes(b"reference")

    result = freeze_selection(
        selection,
        winner="candidate",
        models=[("candidate", candidate), ("reference", reference)],
    )

    assert result["schema_version"] == 12
    assert result["winner"] == "candidate"
    assert result["winner_variant_in_source_selection"] == "internal_variant"
    assert result["ood_attached_at_selection_time"] is False
    assert result["models"]["candidate"]["checkpoint_sha256"] == hashlib.sha256(
        b"candidate"
    ).hexdigest()


def test_freeze_selection_rejects_ood_leakage_and_missing_winner(tmp_path) -> None:
    selection = tmp_path / "selection.json"
    selection.write_text(
        json.dumps(
            {
                "selection_rule": {
                    "selection_year": 2021,
                    "ood_year_excluded_from_selection": 2020,
                }
            }
        )
    )
    checkpoint = tmp_path / "model.ckpt"
    checkpoint.write_bytes(b"model")

    with pytest.raises(ValueError, match="2020"):
        freeze_selection(
            selection,
            winner="model",
            models=[("model", checkpoint)],
        )

    selection.write_text(
        json.dumps(
            {
                "selection_rule": {
                    "selection_year": 2020,
                    "ood_year_excluded_from_selection": 2021,
                }
            }
        )
    )
    with pytest.raises(ValueError, match="winner"):
        freeze_selection(
            selection,
            winner="missing",
            models=[("model", checkpoint)],
        )
