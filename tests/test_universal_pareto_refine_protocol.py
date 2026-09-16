from pathlib import Path


SCRIPT = Path("tools/train/run_universal_pareto_refine_pair_cloudru.sh")


def test_universal_pareto_pair_keeps_honest_sparse_and_all_tau_arms() -> None:
    source = SCRIPT.read_text()

    assert 'ARCH="flow_universal_pareto_refine"' in source
    assert 'PARAMETERS=13659644' in source
    assert 'train_taus=(1 3 5)' in source
    assert 'train_taus=(1 2 3 4 5)' in source
    assert "--eval_tau 1 2 3 4 5" in source
    assert "--loss_profile relative_group_pareto" in source
    assert "--group_balance_ema_decay 0.99" in source
    assert "--group_balance_cvar_fields 6" in source
    assert "distillation.get(\"enabled\") is False" in source


def test_universal_pareto_pair_uses_shared_gpu_locks() -> None:
    source = SCRIPT.read_text()

    assert '.upr_lite_gpu${gpu}.lock' in source
    assert "flock 7" in source
    assert "wait_for_gpu \"$gpu\"" in source
    assert "run_arm 0 sparse &" in source
    assert "run_arm 1 alltau &" in source
