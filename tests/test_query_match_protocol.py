import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = (
    ROOT
    / "tools/train/run_upr_query_match_14m_6h_pilot_cloudru.sh"
)
EVAL_QUEUE = ROOT / "tools/eval/run_upr_lite_6h_eval_queue_cloudru.sh"
PROFILE = ROOT / "tools/train/architecture_candidate_profile.sh"
STRICT_UPR = (
    ROOT / "tools/train/run_upr_implicit_global_14m_6h_cloudru.sh"
)
FACTOR_SCRIPTS = (
    ROOT / "tools/train/run_flow_pp3_hf_6h_pilot_cloudru.sh",
    ROOT
    / "tools/train/run_upr_implicit_global_14m_nohf_6h_pilot_cloudru.sh",
)
SNAPSHOT = (
    ROOT
    / "legacy/training_snapshots/train_capacity_matched_6h_32f4ce54.py"
)
SNAPSHOT_SHA256 = (
    "32f4ce54ddd3d033ea4ea55112d6e656"
    "58caf73961e568680aa627e8794e1f30"
)
MEMMAP_DATASET_SHA256 = (
    "cc5c21f1150e9cdfa9acb1746557db88"
    "a080b09da3e7f8caa4207a25d60fc1fa"
)


def test_query_match_snapshot_is_hash_bound() -> None:
    source = SCRIPT.read_text()

    assert hashlib.sha256(SNAPSHOT.read_bytes()).hexdigest() == SNAPSHOT_SHA256
    assert str(SNAPSHOT.relative_to(ROOT)) in source
    assert SNAPSHOT_SHA256 in source
    assert (
        'TRAINER_ENTRYPOINT="${TRAINER_ENTRYPOINT:-$FROZEN_TRAINER}"'
        in source
    )


def test_query_match_is_in_full_grid_compute_screen() -> None:
    source = (
        ROOT / "tools/eval/benchmark_transport_candidate_compute.py"
    ).read_text()

    assert '"upr_query_match_14m",' in source
    assert '"upr_implicit_global_14m",' in source
    assert "elif arch in {" in source
    assert "upr_scaled_variant_kwargs(arch)" in source
    assert "weatherbridge_upr_lite_model.py" in source
    assert "weatherbridge_upr_scaled_model.py" in source


def test_query_match_uses_distinct_candidate_and_reference_provenance() -> None:
    source = SCRIPT.read_text()
    candidate = source.split(
        "candidate_protocol_args=(",
        maxsplit=1,
    )[1].split(")", maxsplit=1)[0]
    reference = source.split(
        "reference_protocol_args=(",
        maxsplit=1,
    )[1].split(")", maxsplit=1)[0]

    assert '"${common_protocol_args[@]}"' in candidate
    assert '"$FROZEN_TRAINER_SHA256"' in candidate
    assert '"${common_protocol_args[@]}"' in reference
    assert '"$REFERENCE_TRAINER_SHA256"' in reference
    assert '"${candidate_protocol_args[@]}"' in source
    assert '"${reference_protocol_args[@]}"' in source


def test_query_match_preserves_frozen_schedule_and_priority() -> None:
    source = SCRIPT.read_text()

    assert '--arch "$arch"' in source
    assert 'arch="upr_query_match_14m"' in source
    assert "--bs 4" in source
    assert "--accumulate 4" in source
    assert "--max_epochs 8" in source
    assert "run_stage 2" not in source
    assert "stop_after_screen_checkpoint.py" in source
    assert "epoch=1-step=3284.ckpt" in source
    assert "intrinsic OOM violates frozen bs=4" in source
    assert "OOM fallback" not in source
    assert "wait objective-factor priority" not in source
    assert source.index("stop_after_screen_checkpoint.py") < source.index(
        "wait two-epoch strict UPR reference for assessment"
    )


def test_query_match_has_prespecified_two_epoch_gate() -> None:
    source = SCRIPT.read_text()

    assert "assess_two_epoch_candidate.py" in source
    assert "--held-relative-limit 0.02" in source
    assert "--all-hour-relative-limit 0.02" in source
    assert "--per-hour-relative-limit 0.05" in source
    assert "--require-resume-lineage" in source
    assert 'touch "$started_marker"' in source
    assert 'touch "$terminal_marker"' in source
    assert "trap mark_failed_exit EXIT" in source


def test_strict_upr_yields_gpu_after_screen_for_query_match() -> None:
    source = STRICT_UPR.read_text()
    barrier = (
        'wait_for_architecture_screen_barrier "$exp_name" "$queue_log"'
    )

    assert "--max_epochs 8" in source
    assert "stop_after_screen_checkpoint.py" in source
    assert "--min-epochs 2" in source
    assert source.count("--require-resume-lineage") >= 2
    assert "source tools/train/architecture_screen_barrier.sh" in source
    assert source.index("stop_after_screen_checkpoint.py") < source.index(
        barrier
    )
    assert "release_candidate_gpu" in (
        ROOT / "tools/train/architecture_screen_barrier.sh"
    ).read_text()


def test_objective_factors_yield_gpu_after_screen_for_query_match() -> None:
    handoff = 'touch "$screened_marker"\nrelease_gpu'
    for script in FACTOR_SCRIPTS:
        source = script.read_text()
        tail = source.split(handoff, maxsplit=1)[1]

        assert "--max_epochs 8" in source
        assert "stop_after_screen_checkpoint.py" in source
        assert "--min-epochs 2" in source
        assert "--require-resume-lineage" in source
        assert "yield GPU until QueryMatch starts" in tail
        assert tail.index("yield GPU until QueryMatch starts") < tail.index(
            "acquire_gpu"
        )


def test_query_match_marks_screen_only_after_prespecified_assessment() -> None:
    source = SCRIPT.read_text()
    post_assessment = source.split(
        "tools/eval/assess_two_epoch_candidate.py",
        maxsplit=1,
    )[1]

    assert 'touch "$screened_marker"' in post_assessment
    assert post_assessment.index('touch "$screened_marker"') < (
        post_assessment.index('if [[ "$promoted" -ne 1 ]]')
    )


def test_query_match_reaches_hash_bound_full_year_selection() -> None:
    source = EVAL_QUEUE.read_text()

    assert (
        "[upr_query_match_14m]=\"$LOG_ROOT/"
        "exp_upr_query_match_14m_hf_135_14m_6h_s202707_protocol_v1/"
        "last.ckpt\""
        in source
    )
    assert SNAPSHOT_SHA256 in source
    assert "query_match_pilot=" in source
    assert "query_match_terminal=" in source
    assert "checkpoint_extra_args upr_query_match_14m" in source
    assert "--expected-arch upr_query_match_14m" in source
    assert "build_avg3_auxiliary upr_query_match_14m" in source
    assert source.count('"${query_match_models[@]}"') == 2
    assert '"${query_match_aux_models[@]}"' in source
    assert '"$name" == upr_query_match_14m*' in source
    assert "--require-resume-lineage" in source


def test_query_match_postselection_uses_its_immutable_snapshot() -> None:
    training_paths = (
        "tools/train/run_upr_lite_winner_replicates_cloudru.sh",
        "tools/train/run_upr_lite_winner_nohf_cloudru.sh",
        "tools/train/run_upr_lite_nohf_replicates_cloudru.sh",
        "tools/train/run_upr_lite_winner_12h_cloudru.sh",
    )
    profile = PROFILE.read_text()
    assert "upr_query_match_14m" in profile
    assert str(SNAPSHOT.relative_to(ROOT)) in profile
    assert SNAPSHOT_SHA256 in profile
    for relative in training_paths:
        source = (ROOT / relative).read_text()
        assert "source tools/train/architecture_candidate_profile.sh" in source
        assert "architecture_candidate_trainer" in source
        assert (
            '--expected-trainer-sha256 "$TRAINER_SHA256"' in source
            or '--expected-trainer-sha256 "$trainer_sha256"' in source
        )


def test_query_match_eval_keeps_candidate_and_pp3_provenance_separate() -> None:
    candidate_paths = (
        "tools/eval/run_upr_lite_seed_eval_queue_cloudru.sh",
        "tools/eval/run_upr_vs_pp3_ood_spectra_queue_cloudru.sh",
        "tools/eval/run_upr_nohf_vs_pp3_seed_eval_queue_cloudru.sh",
        "tools/eval/run_upr_nohf_vs_pp3_ood_spectra_queue_cloudru.sh",
    )
    profile = PROFILE.read_text()
    assert "upr_query_match_14m" in profile
    assert SNAPSHOT_SHA256 in profile
    for relative in candidate_paths:
        source = (ROOT / relative).read_text()
        assert "source tools/train/architecture_candidate_profile.sh" in source
        assert "architecture_candidate_trainer_sha256" in source

    for relative in candidate_paths[1:]:
        source = (ROOT / relative).read_text()
        assert '"$BASE_TRAINER_SHA256" 4 4; do' in source


def test_query_match_12h_transfer_uses_matched_layout_and_provenance() -> None:
    training = (
        ROOT / "tools/train/run_upr_lite_winner_12h_cloudru.sh"
    ).read_text()
    evaluation = (
        ROOT / "tools/eval/run_upr_lite_12h_transfer_queue_cloudru.sh"
    ).read_text()
    router = (
        ROOT / "tools/eval/run_temporal_expert_router_12h_queue_cloudru.sh"
    ).read_text()

    profile = PROFILE.read_text()
    assert "upr_query_match_14m" in profile
    assert "architecture_candidate_trainer" in training
    assert "architecture_candidate_model_arch" in evaluation
    assert "candidate_trainer_sha256()" in evaluation
    assert SNAPSHOT_SHA256 in profile
    assert (
        '--expected-trainer-sha256 '
        '"${EXPECTED_TRAINER_SHA256[$name]}"'
        in evaluation
    )
    assert "architecture_candidate_layout" in router


def test_12h_transfer_requires_correct_tau_normalization_source() -> None:
    for relative in (
        "tools/train/run_upr_lite_winner_12h_cloudru.sh",
        "tools/eval/run_upr_lite_12h_transfer_queue_cloudru.sh",
    ):
        source = (ROOT / relative).read_text()
        assert MEMMAP_DATASET_SHA256 in source
        assert 'sha256sum "$MEMMAP_DATASET"' in source
        assert "12h memmap tau-normalization SHA-256 mismatch" in source
