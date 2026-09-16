from __future__ import annotations

from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "tools"
    / "train"
    / "run_qhead_gated_followup_6h_cloudru.sh"
)


def test_gated_followup_is_matched_and_truth_filtered() -> None:
    source = SCRIPT.read_text()

    assert "--trainable_scope q_output_head" in source
    assert "--distill_high_gate teacher_better" in source
    assert "--distill_high_weight 0.08" in source
    assert "--train_tau_subset 1 3 5" in source
    assert "--eval_tau 1 2 3 4 5" in source
    assert "--train_batches_per_epoch 6568" in source
    assert "--max_epochs 2" in source
    assert "--seed 202707" in source


def test_gated_checkpoint_preserves_non_q_network() -> None:
    source = SCRIPT.read_text()

    assert 'distill.get("high_gate") == "teacher_better"' in source
    assert 'protocol.get("optimizer_weight_decay") == 0.0' in source
    assert 'torch.equal(state[key][:12], reference[key][:12])' in source
    assert 'torch.equal(state[key][16:], reference[key][16:])' in source
    assert 'not any("teacher" in key or "distill" in key' in source


def test_gated_followup_runs_the_matched_2020_screen() -> None:
    source = SCRIPT.read_text()

    assert "batch_eval_12h_memmap.py" in source
    assert '--models "qhead_flow_gated:$CHECKPOINT"' in source
    assert "--eval-days-per-month 8" in source
    assert "--samples-per-date 2" in source
    assert "--proper-rmse --save-window-metrics" in source
    assert 'touch "$STATE/all.complete"' in source
