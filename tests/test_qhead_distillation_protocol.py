from __future__ import annotations

from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "tools"
    / "train"
    / "run_qhead_distillation_6h_cloudru.sh"
)


def test_qhead_queue_uses_matched_control_and_flow_arms() -> None:
    source = SCRIPT.read_text()

    assert "ARMS=(qhead_gt qhead_flow)" in source
    assert '--init_weights_path "$CONTROL"' in source
    assert "--trainable_scope q_output_head" in source
    assert "--distill_high_weight 0.08" in source
    assert "--distill_high_weight 0" in source
    assert "--max_epochs 2 --lr 2e-5 --warmup_steps 100" in source
    assert "--train_batches_per_epoch 6568" in source


def test_qhead_queue_verifies_non_q_state_identity() -> None:
    source = SCRIPT.read_text()

    assert 'protocol.get("optimizer_weight_decay") == 0.0' in source
    assert 'torch.equal(state[key][:12], reference[key][:12])' in source
    assert 'torch.equal(state[key][16:], reference[key][16:])' in source
    assert 'not torch.equal(state[key][12:16], reference[key][12:16])' in source


def test_qhead_queue_runs_grouped_bootstraps_and_strict_gate() -> None:
    source = SCRIPT.read_text()

    assert "paired_block_bootstrap.py" in source
    assert "--channels Q1000,Q925,Q850,Q700" in source
    assert "select_specialist_distillation.py" in source
    assert "select_distilled_champion.py" in source
    assert 'touch "$STATE/strict.pass"' in source
    assert 'touch "$STATE/strict.failed"' in source
