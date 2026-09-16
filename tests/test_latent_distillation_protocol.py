from __future__ import annotations

from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "tools"
    / "train"
    / "run_latent_distillation_6h_cloudru.sh"
)


def test_latent_arm_preserves_matched_truth_primary_recipe() -> None:
    source = SCRIPT.read_text()

    assert '--init_weights_path "$DCAE_INIT"' in source
    assert '--distill_latent_teacher_checkpoint "$CONTROL"' in source
    assert "--max_epochs 2 --lr 2e-5 --warmup_steps 100" in source
    assert "--train_tau_subset 1 3 5 --eval_tau 1 2 3 4 5" in source
    assert "--train_batches_per_epoch 6568" in source
    assert "--distill_high_mask_profile moisture" in source
    assert "--distill_high_weight 0.08" in source


def test_latent_objective_decays_in_time_and_across_blocks() -> None:
    source = SCRIPT.read_text()

    assert "--distill_latent_weight 0.0005" in source
    assert "--distill_latent_every_n_steps 2" in source
    assert "--distill_latent_schedule cosine_decay" in source
    assert "--distill_latent_decay_end_fraction 0.5" in source
    assert "--distill_latent_block_decay 0.5" in source
    assert "--distill_latent_pool_height 45" in source
    assert "--distill_latent_pool_width 90" in source


def test_latent_selection_precedes_ood_spectra_and_hres() -> None:
    source = SCRIPT.read_text()

    selection = source.index("select_specialist_distillation.py")
    ood = source.index('run_eval 1 2021 "$EVAL_2021"')
    spectra = source.index('run_spectra 1 "$year"')
    hres = source.index("batch_eval_forecast_anchor.py")
    assert selection < ood < spectra < hres
    assert "--candidate \"q_latent:$EVAL_2020/q_latent.json\"" in source
    assert 'not any("teacher" in key' in source


def test_latent_queue_forwards_selection_after_early_rejection() -> None:
    source = SCRIPT.read_text()

    assert "latent_rejected.json" in source
    assert "record_rejected_latent_selection" in source
    assert 'if [[ -s "$LATENT_REJECTION" ]]' in source
    assert 'selection["excluded_candidates"] = {"q_latent": rejection}' in source
    assert 'selection["selection_extension_role"]' in source
