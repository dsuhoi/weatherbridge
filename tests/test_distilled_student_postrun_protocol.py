from __future__ import annotations

from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "tools"
    / "eval"
    / "run_distilled_student_postrun_cloudru.sh"
)


def test_pipeline_freezes_selection_before_ood() -> None:
    source = SCRIPT.read_text()

    strict = source.index("select_distilled_champion.py")
    freeze = source.index("freeze_distilled_student_selection.py")
    full_year = source.index("--full-year")
    hres = source.index("batch_eval_forecast_anchor.py")
    assert strict < freeze < full_year < hres
    assert '[[ "$status" != pass ]]' in source
    assert '[[ ! -e "$SCREEN_ROOT/state/screen.complete" ]]' in source
    assert '[[ ! -e "$GATED_ROOT/state/all.complete" ]]' in source
    assert '--control "$SCREEN_EVAL/qhead_gt.json"' in source
    assert '--matched-screen "$MATCHED_SELECTION"' in source
    assert '--candidate "qhead_flow_gated:$GATED_EVAL/qhead_flow_gated.json"' in source


def test_pipeline_accepts_only_frozen_distilled_qhead_variant() -> None:
    source = SCRIPT.read_text()

    assert "qhead_flow) STUDENT=$QHEAD_FLOW" in source
    assert "qhead_flow_gated) STUDENT=$QHEAD_FLOW_GATED" in source
    assert "qhead_gt) STUDENT=" not in source
    assert 'distill.get("high_gate") == expected_gate' in source
    assert 'protocol.get("trainable_scope") == "q_output_head"' in source
    assert 'protocol.get("optimizer_weight_decay") == 0.0' in source
    assert 'torch.equal(state[key][:12], reference[key][:12])' in source
    assert 'torch.equal(state[key][16:], reference[key][16:])' in source


def test_pipeline_covers_journal_metrics_and_stress_tests() -> None:
    source = SCRIPT.read_text()

    assert "--acc-mode enabled" in source
    assert "--require-physical-metrics" in source
    assert "--require-temporal-metrics" in source
    assert "paired_block_bootstrap.py" in source
    assert "paired_rmse_moisture.json" in source
    assert "paired_rmse_non_moisture.json" in source
    assert "--channels Q1000,Q925,Q850,Q700" in source
    assert "paired_aux_block_bootstrap.py" in source
    assert "eval_anchor_exchange_consistency.py" in source
    assert "region_season_12h_eval.py" in source
    assert "sh_energy_spectra_12h.py" in source
    assert "--lmax 359" in source
    assert "--hf-ell-min 180" in source
    assert "eval_physics_consistency.py" in source
    assert "eval_diurnal_amplitude.py" in source
    assert "benchmark_capmatched_inference.py" in source


def test_pipeline_uses_heldout_hours_and_two_years() -> None:
    source = SCRIPT.read_text()

    assert "for year in 2020 2021" in source
    assert "--seen-tau 1,3,5 --unseen-tau 2,4" in source
    assert "--eval-hours 2,4" in source
    assert "--max-inits 16" in source
    assert "pipeline.complete.json" in source
