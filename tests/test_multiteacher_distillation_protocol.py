from __future__ import annotations

from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "tools"
    / "train"
    / "run_multiteacher_distillation_6h_cloudru.sh"
)


def test_queue_uses_a_matched_truth_only_continuation_control() -> None:
    source = SCRIPT.read_text()

    assert "--arch dcae_14m" in source
    assert '--init_weights_path "$DCAE_TEACHER"' in source
    assert "--max_epochs 2" in source
    assert "--lr 2e-5" in source
    assert "--train_tau_subset 1 3 5" in source
    assert "--eval_tau 1 2 3 4 5" in source
    assert "--train_batches_per_epoch 6568" in source
    assert "--distill_low_weight 0" in source
    assert "--distill_high_weight 0" in source


def test_queue_is_oracle_gated_and_uses_shared_gpu_locks() -> None:
    source = SCRIPT.read_text()

    assert source.index("run_oracle") < source.index(
        'run_arm 0 "$CONTROL_EXP" control'
    )
    assert '"$LOG_ROOT/.upr_lite_gpu${gpu}.lock"' in source
    assert 'run_arm 0 "$CONTROL_EXP" control true &' in source
    assert source.index('wait "$CONTROL_PID"') < source.index("flock -u 9")
    assert "aggregate_tolerance 0.001" in source
    assert "held_tolerance 0.002" in source
    assert 'payload["status"] == "pass"' in source


def test_distilled_checkpoint_excludes_teacher_parameters() -> None:
    source = SCRIPT.read_text()

    assert "parameters == 14365049" in source
    assert 'not any("teacher" in name or "distill" in name' in source
    assert 'distillation.get("truth_primary") is True' in source
    assert 'distillation.get("teacher_parameters_saved") is False' in source
