from __future__ import annotations

from pathlib import Path


SCRIPT = Path("tools/train/run_direct_response_distillation_6h_cloudru.sh")


def test_direct_response_pair_is_matched_and_fail_closed() -> None:
    source = SCRIPT.read_text()

    assert "--minimum-improvement-pct 0.5" in source
    assert "--routing-granularity field" in source
    assert "--distill_direct_blend 0.5" in source
    assert "--distill_schedule cosine_decay" in source
    assert "--distill_decay_end_fraction 0.75" in source
    assert "--init_weights_path \"$BASE\"" in source
    assert "--max_epochs 1 --lr 1e-5" in source
    assert "--train_batches_per_epoch 1640" in source
    assert "--seed 202711" in source
    assert "--trainable_scope all" in source
    assert "train_one control 0" in source
    assert "train_one student 1" in source
    assert "--matched-screen" in source
    assert 'touch "$STATE/strict.pass"' in source
    assert 'touch "$STATE/strict.failed"' in source


def test_direct_response_checkpoint_verifies_lineage_and_no_teacher_state() -> None:
    source = SCRIPT.read_text()

    assert 'lineage["checkpoint_sha256"]' in source
    assert 'direct["route"]["sha256"]' in source
    assert 'direct["teacher_lineage"]["checkpoint_sha256"]' in source
    assert 'not any("teacher" in key or "distill" in key for key in state)' in source
    assert 'assert route["active_routes"] == 25' in source
    assert 'assert active == [16, 17, 18, 19, 23]' in source
