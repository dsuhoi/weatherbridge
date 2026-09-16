from pathlib import Path


def test_factorial_queue_fills_both_missing_cells_for_three_seeds() -> None:
    queue = Path("scripts/run_loss_architecture_factorial_cloudru.sh").read_text()
    trainer = Path("tools/train/run_npj_seed_replicates_cloudru.sh").read_text()

    for family in ("flow_pixel", "dcae_spectral"):
        for seed in (202707, 202708, 202709):
            assert f"{family}:6:{seed}" in queue
    assert "RUN_IFS=0" in queue
    assert 'LAMBDA_HF=0\n      LAMBDA_SPEC=0' in trainer
    assert 'LAMBDA_HF=0.05\n      LAMBDA_SPEC=0.02' in trainer
    assert 'SPECTRAL_MASK=advected' in trainer
    assert 'if [[ "${RUN_IFS:-1}" == 1 ]]' in trainer
    assert 'EXP="exp_flow_pp3_pixel_14m_6h_refinev1_s${SEED}_bs4"' in trainer
    assert "exp_flow_pp3_135_14m_6h_s202707_protocol_v2" not in trainer


def test_factorial_queue_recovers_stale_flow_pixel_seed_separately() -> None:
    source = Path("scripts/run_loss_architecture_factorial_cloudru.sh").read_text()
    assert "RECOVERY_STATE_DIR=" in source
    assert 'JOBS_OVERRIDE="flow_pixel:6:202707"' in source
    assert '"${RECOVER_ONLY:-0}" == 1' in source
    assert 'touch "$PRIMARY_STATE_DIR/flow_pixel_6_202707.done"' in source
    assert 'rm -f "$PRIMARY_STATE_DIR/flow_pixel_6_202707.failed"' in source
