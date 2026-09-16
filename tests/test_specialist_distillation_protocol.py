from __future__ import annotations

from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "tools"
    / "train"
    / "run_specialist_distillation_6h_cloudru.sh"
)


def test_specialist_arms_share_the_matched_control_recipe() -> None:
    source = SCRIPT.read_text()

    assert '--init_weights_path "$DCAE_INIT"' in source
    assert "--max_epochs 2 --lr 2e-5 --warmup_steps 100" in source
    assert "--train_tau_subset 1 3 5 --eval_tau 1 2 3 4 5" in source
    assert "--train_batches_per_epoch 6568" in source
    assert "--distill_high_mask_profile moisture" in source
    assert "--distill_high_weight 0.08" in source
    assert "--distill_schedule cosine_decay" in source
    assert "--distill_decay_end_fraction 0.75" in source


def test_selection_is_validation_only_before_ood_and_hres() -> None:
    source = SCRIPT.read_text()

    selection = source.index("select_specialist_distillation.py")
    ood = source.index('run_eval 0 2021 "$EVAL_2021"')
    hres = source.index("batch_eval_forecast_anchor.py")
    assert selection < ood < hres
    assert "--test-year 2020" not in source
    assert "selection_role" not in source


def test_queue_records_statistical_rmse_and_spectral_checks() -> None:
    source = SCRIPT.read_text()

    assert "paired_block_bootstrap.py" in source
    assert source.count("spectral_block_bootstrap.py") == 2
    assert "--channels Q1000,Q925,Q850,Q700" in source
    assert "assert not any(\"teacher\" in key" in source
