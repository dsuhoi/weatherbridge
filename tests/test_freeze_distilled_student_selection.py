from __future__ import annotations

import json

import pytest

from tools.eval.freeze_distilled_student_selection import (
    freeze_distillation_selection,
)


def test_freezes_passing_validation_only_student(tmp_path) -> None:
    selection = tmp_path / "selection.json"
    selection.write_text(json.dumps({
        "status": "pass",
        "selected": "q_latent",
        "selection_role": "era5_2020_validation_only",
        "candidates": {"q_latent": {"pass": True}},
    }))
    student = tmp_path / "student.ckpt"
    control = tmp_path / "control.ckpt"
    student.write_bytes(b"student")
    control.write_bytes(b"control")

    frozen = freeze_distillation_selection(
        selection,
        [("distilled_student", student), ("control", control)],
    )

    assert frozen["schema_version"] == 12
    assert frozen["winner"] == "distilled_student"
    assert frozen["winner_variant_in_source_selection"] == "q_latent"
    assert frozen["ood_attached_at_selection_time"] is False
    assert len(
        frozen["models"]["distilled_student"]["checkpoint_sha256"]
    ) == 64


def test_rejects_failed_or_nonvalidation_selection(tmp_path) -> None:
    selection = tmp_path / "selection.json"
    selection.write_text(json.dumps({
        "status": "fail",
        "selected": None,
        "selection_role": "era5_2020_validation_only",
        "candidates": {},
    }))
    checkpoint = tmp_path / "student.ckpt"
    checkpoint.write_bytes(b"student")

    with pytest.raises(ValueError, match="did not pass"):
        freeze_distillation_selection(
            selection,
            [("distilled_student", checkpoint)],
        )
