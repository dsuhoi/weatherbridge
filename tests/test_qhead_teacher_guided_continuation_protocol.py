from __future__ import annotations

from pathlib import Path


SCRIPT = Path("tools/train/run_qhead_teacher_guided_continuation_6h_cloudru.sh")


def test_guided_continuation_has_matched_control_and_strict_gate() -> None:
    source = SCRIPT.read_text()

    assert "--init_weights_path \"$BASE\"" in source
    assert "--trainable_scope q_output_head" in source
    assert "--train_batches_per_epoch 3284" in source
    assert "--max_epochs 1 --lr 5e-6" in source
    assert "--seed 202708" in source
    assert "train_one control 0" in source
    assert "train_one guided 1" in source
    assert "--distill_high_gate teacher_better_truth" in source
    assert "--distill_high_weight 0.04" in source
    assert "--matched-screen" in source


def test_guided_continuation_checks_frozen_and_non_q_rows() -> None:
    source = SCRIPT.read_text()

    assert 'assert torch.equal(value, reference[key]), key' in source
    assert 'assert torch.equal(state[key][:12], reference[key][:12]), key' in source
    assert 'assert torch.equal(state[key][16:], reference[key][16:]), key' in source
    assert 'assert not torch.equal(state[key][12:16], reference[key][12:16]), key' in source
