from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_lastmile_training_is_low_lr_full_network_and_sparse_tau() -> None:
    source = (
        REPO_ROOT
        / "tools/train/run_flow_lagrange_allnet_lastmile_cloudru.sh"
    ).read_text()

    assert "SCALE_SCREEN_COMPLETE=" in source
    assert 'while [[ ! -e "$SCALE_SCREEN_COMPLETE" ]]' in source
    assert "--arch flow_compact_lagrange_l" in source
    assert "--trainable_scope all" in source
    assert "--lr 1e-5" in source
    assert "--max_epochs 1" in source
    assert "--lambda_hf_override 0.02" in source
    assert "--train_tau_subset 1 3 5" in source
    assert "--eval_tau 1 2 3 4 5" in source
    assert "--years 2014 2015 2016 2017 2018 2019" in source
    assert "--val_years 2020" in source
    assert "--samples_per_date_train 4" in source
    assert "--train_batches_per_epoch 6568" in source
    assert "[base]=\"base_balanced\"" in source
    assert "[edge]=\"base_edge_balanced\"" in source
    assert "INITIAL_CHECKPOINT_SHA256=" in source
    assert "verify_initial_checkpoint" in source


def test_lastmile_evaluation_requires_strict_2020_and_2021_gates() -> None:
    source = (
        REPO_ROOT
        / "tools/eval/run_lagrange_allnet_lastmile_eval_cloudru.sh"
    ).read_text()

    assert "--eval-days-per-month 8" in source
    assert "--eval-hours 1,2,3,4,5" in source
    assert "--seen-tau 1,3,5" in source
    assert "--unseen-tau 2,4" in source
    assert "--full-year" in source
    assert "run_full_eval 2020" in source
    assert "run_full_eval 2021" in source
    assert source.index("run_full_eval 2020") < source.index(
        'if [[ -z "$(selected_from "$FULL_REPORT")" ]]'
    )
    assert source.index(
        'if [[ -z "$(selected_from "$FULL_REPORT")" ]]'
    ) < source.index("run_full_eval 2021")
    for gate in (
        "--base-mean-limit 0",
        "--seen-mean-limit 0",
        "--held-mean-limit 0",
        "--per-tau-limit 0",
        "--surface-mean-limit 0",
    ):
        assert source.count(gate) == 2
