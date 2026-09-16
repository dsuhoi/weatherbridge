from pathlib import Path


SCRIPT = Path("tools/train/run_npj_seed_replicates_cloudru.sh")


def test_npj_seed_queue_freezes_complete_matched_matrix() -> None:
    source = SCRIPT.read_text()
    for horizon in (6, 12):
        for family in ("refine", "flow_spectral", "dcae"):
            for seed in (202707, 202708, 202709):
                assert f"{family}:{horizon}:{seed}" in source

    assert "weatherbridge_universal_refine_v1_source" in source
    assert "--ckpt_every_n_epochs \"$MAX_EPOCHS\"" in source
    assert "--anchor_swap_probability 0" in source
    assert "--trainable_scope all" in source
    assert 'if [[ -n "${JOBS_OVERRIDE:-}" ]]' in source
    assert 'ARCH="flow_pp3_detail"' in source
    assert "exp_flow_pp3_detail_14m_12h_2017_19_refinev1" in source


def test_npj_seed_queue_includes_cross_source_ifs_evaluation() -> None:
    source = SCRIPT.read_text()
    assert "memmap_hres_2021" in source
    assert "batch_eval_forecast_anchor.py" in source
    assert "memmap_hres_2021_canonical_v3" in source
    assert "forecast_archive_manifest.json" in source
    assert "IFS_MAX_INITS=\"${IFS_MAX_INITS:-16}\"" in source
    assert "linear:6:0" in source
    assert "linear:12:0" in source
    assert "--era5-memmap-dir \"$MEMMAP\"" in source


def test_npj_seed_queue_does_not_duplicate_an_active_experiment() -> None:
    source = SCRIPT.read_text()
    assert 'pgrep -f -- "--exp_name ${EXP}( |$)"' in source
    assert '"$LOG_ROOT/$EXP.queue.lock"' in source
    assert '"$LOG_ROOT/.upr_lite_gpu${GPU}.lock"' in source
    assert 'LAUNCH_LOCK="${LAUNCH_LOCK:-$STATE_DIR/launch.lock}"' in source
    assert 'WORKER_GPUS="${WORKER_GPUS:-0 1}"' in source
    assert '&& ! -e "$STATE_DIR/$id.failed"' in source


def test_skip_pair_queue_does_not_hold_one_gpu_while_waiting() -> None:
    source = Path("scripts/run_journal_dcae_skip_queue_cloudru.sh").read_text()
    assert "acquire_both_gpu_locks" in source
    assert "if flock -n 7" in source
    assert "if flock -n 8" in source
    assert "flock -u 7" in source
    assert "flock -u 8" in source
    assert "flock 7\nflock 8" not in source


def test_detail_reference_waits_for_primary_queue() -> None:
    source = Path("scripts/run_detail_reference_queue_cloudru.sh").read_text()
    assert 'JOBS_OVERRIDE="detail:6:202707"' in source
    assert 'JOBS_OVERRIDE="detail:12:202707"' in source
    assert "wait_for_horizon 6 refine flow_spectral" in source
    assert "wait_for_horizon 12 refine flow_spectral dcae" in source
    assert "npj_seed_replicates_v2.state" in source
    assert "npj_seed_replicates_v1.state" in source
    assert "npj_seed_ifs_hres_2021_v3" in source
    assert "npj_detail_reference_v1_6h.state" in source
    assert "npj_detail_reference_v1_12h.state" in source
    assert "detail_6_202707.done" in source
    assert "detail_12_202707.done" in source
    assert 'WORKER_GPUS="${DETAIL_WORKER_GPUS:-1}"' in source
    assert "EXPECTED_SOURCE_SHA" in source


def test_seed_geometry_recheck_is_a_selection_barrier() -> None:
    source = Path("scripts/run_npj_seed_geometry_recheck_cloudru.sh").read_text()
    champion = Path("scripts/run_journal_champion_queue_cloudru.sh").read_text()
    assert "wb2_0p25_2x2_block_average_v1" in source
    assert "spherical_latitude_strip_area" in source
    assert "batch_eval_forecast_anchor.py" in source
    assert "source_files_sha256" in source
    assert "reclaiming stale geometry artifact" in source
    assert 'if ! artifact_current; then' in source
    assert "npj_seed_replicates_v2.state" in source
    assert "npj_seed_replicates_v1.state" in source
    assert "primary_job_done" in source
    assert "npj_detail_reference_v1" not in source
    assert "wait_for_hres_archive" in source
    assert "_forecast_manifest_provenance" in source
    assert 'len(init_files) != expected' in source
    assert 'forecast["manifest_sha256"] == current_forecast["manifest_sha256"]' in source
    assert "geometry_v3/primary.complete" in champion
    assert "SEED_STATE" not in champion
    assert "detail_confirmation.complete" in champion
