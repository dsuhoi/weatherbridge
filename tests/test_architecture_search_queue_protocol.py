import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _source(relative: str) -> str:
    return (REPO_ROOT / relative).read_text()


def test_architecture_candidate_profiles_are_consistent() -> None:
    profile = REPO_ROOT / "tools/train/architecture_candidate_profile.sh"
    command = r'''
source "$1"
for candidate in \
  upr_lite \
  upr_lite_column \
  upr_implicit_global_14m_nohf \
  upr_query_match_14m \
  upr_local_corr_14m \
  upr_spherical_implicit_global_14m \
  flow_pp3_hf \
  amt \
  amt_residual; do
  printf '%s|%s|%s|%s|%s|%s|%s|%s\n' \
    "$candidate" \
    "$(architecture_candidate_model_arch "$candidate")" \
    "$(architecture_candidate_layout "$candidate")" \
    "$(architecture_candidate_batch_size "$candidate")" \
    "$(architecture_candidate_accumulate "$candidate")" \
    "$(architecture_candidate_lambda_hf "$candidate")" \
    "$(architecture_candidate_highpass_boundary "$candidate")" \
    "$(architecture_candidate_trainer_sha256 "$candidate")"
done
'''
    result = subprocess.run(
        ["bash", "-c", command, "_", str(profile)],
        check=True,
        capture_output=True,
        text=True,
    )
    rows = result.stdout.splitlines()

    assert rows[0].split("|")[:7] == [
        "upr_lite",
        "upr_lite",
        "b8",
        "8",
        "2",
        "0",
        "periodic_lon_replicate_lat",
    ]
    assert rows[1].split("|")[2:6] == ["b8", "8", "2", "0.05"]
    assert rows[2].split("|")[:7] == [
        "upr_implicit_global_14m_nohf",
        "upr_implicit_global_14m",
        "eb16",
        "4",
        "4",
        "0",
        "periodic_lon_replicate_lat",
    ]
    assert rows[3].split("|")[1:3] == ["upr_query_match_14m", "eb16"]
    assert rows[3].split("|")[-1].startswith("32f4ce54")
    assert rows[4].split("|")[1:3] == ["upr_local_corr_14m", "eb16"]
    assert rows[4].split("|")[6] == "periodic_lon_replicate_lat"
    assert rows[4].split("|")[-1].startswith("d7a6bb0d")
    assert rows[5].split("|")[6] == "antipodal_vector_parity"
    assert rows[6].split("|")[1:3] == ["flow_pp3", "eb16"]
    assert rows[7].split("|")[1:3] == ["amt", "eb16"]
    assert rows[7].split("|")[6] == "antipodal_vector_parity"
    assert rows[8].split("|")[1:3] == ["amt_residual", "eb16"]
    assert rows[8].split("|")[6] == "antipodal_vector_parity"
    assert rows[8].split("|")[-1].startswith("62c9566f")


def test_weather_amt_uses_common_highpass_recipe() -> None:
    pilot = _source("tools/train/run_weather_amt_6h_pilot_cloudru.sh")
    evaluation = _source(
        "tools/eval/run_upr_lite_6h_eval_queue_cloudru.sh"
    )
    transfer = _source(
        "tools/train/run_upr_lite_winner_12h_cloudru.sh"
    )
    replicates = _source(
        "tools/train/run_upr_lite_winner_replicates_cloudru.sh"
    )

    assert "exp_weather_amt_l_14m_6h_s202707_protocol_v4" in pilot
    assert "weather_amt_l_pilot_v4.json" in pilot
    assert "weather_amt_l_polar_2020" in pilot
    assert "--pilot-name weather_amt_l" in pilot
    assert "--polar-pilot-checkpoint" in pilot
    assert "--polar-reference-checkpoint" in pilot
    assert "--lambda_hf_override 0.05" in pilot
    assert "--expected-lambda-hf 0.05" in pilot
    assert "exp_weather_amt_l_14m_6h_s202707_protocol_v4" in evaluation
    assert "weather_amt_l_pilot_v4.json" in evaluation
    for source in (transfer, replicates):
        assert '"$candidate" == "amt"' in source
        assert '"$candidate" == "amt_residual"' in source
        assert "loss_args=(--lambda_hf_override 0.05)" in source


def test_weather_amt_residual_is_an_independent_hash_bound_candidate() -> None:
    pilot = _source(
        "tools/train/run_weather_amt_residual_6h_pilot_cloudru.sh"
    )
    evaluation = _source(
        "tools/eval/run_upr_lite_6h_eval_queue_cloudru.sh"
    )
    loader = _source("tools/eval/capmatched_loader.py")
    benchmark = _source(
        "tools/eval/benchmark_transport_candidate_compute.py"
    )

    assert 'arch="amt_residual"' in pilot
    assert (
        "exp_weather_amt_residual_l_14m_6h_s202707_protocol_v1"
        in pilot
    )
    assert "weather_amt_residual_l_pilot_v1.json" in pilot
    assert "--pilot-name weather_amt_residual_l" in pilot
    assert "--lambda_hf_override 0.05" in pilot
    assert "62c9566ff7b69db0770afc510ade626b" in pilot
    assert 'TRAINER_ENTRYPOINT="${TRAINER_ENTRYPOINT:-$FROZEN_TRAINER}"' in pilot
    assert "wait_for_architecture_completion" in pilot
    assert "release_candidate_gpu" in pilot
    assert "acquire_candidate_gpu" in pilot

    assert "[amt_residual]=" in evaluation
    assert "weather_amt_residual_l_pilot_v1.json" in evaluation
    assert "--expected-arch amt_residual" in evaluation
    assert "AMT_RESIDUAL_TRAINER_SHA256" in evaluation
    assert 'if arch == "amt_residual":' in loader
    assert '"amt_residual",' in benchmark


def test_local_corr_is_a_hash_bound_post_barrier_candidate() -> None:
    pilot = _source(
        "tools/train/run_upr_local_corr_14m_6h_pilot_cloudru.sh"
    )
    evaluation = _source(
        "tools/eval/run_upr_lite_6h_eval_queue_cloudru.sh"
    )
    profile = _source("tools/train/architecture_candidate_profile.sh")
    loader = _source("tools/eval/capmatched_loader.py")
    benchmark = _source(
        "tools/eval/benchmark_transport_candidate_compute.py"
    )

    assert 'arch="upr_local_corr_14m"' in pilot
    assert (
        "exp_upr_local_corr_14m_hf_135_14m_6h_s202707_protocol_v1"
        in pilot
    )
    assert "upr_local_corr_14m_pilot.json" in pilot
    assert "d7a6bb0d0b801eadb3365bf7b84803c" in pilot
    assert "b52cdc0407c4e5efce70af7ada523fa" in pilot
    assert "a36c3092943fe6021d9ea0a710e73e6" in pilot
    assert "wait_for_architecture_completion" in pilot
    assert "wait_for_architecture_screen_barrier" not in pilot
    assert "--lambda_hf_override 0.05" in pilot
    assert "--expected-highpass-boundary periodic_lon_replicate_lat" in pilot
    assert "stop_after_screen_checkpoint.py" in pilot

    assert "[upr_local_corr_14m]=" in evaluation
    assert "upr_local_corr_14m_pilot.json" in evaluation
    assert "--expected-arch upr_local_corr_14m" in evaluation
    assert "LOCAL_CORR_TRAINER_SHA256" in evaluation
    assert "local_corr_sources_ready" in evaluation
    assert "wait frozen LocalCorr inference sources" in evaluation
    assert "upr_local_corr_14m)" in profile
    assert '"upr_local_corr_14m",' in loader
    assert '"upr_local_corr_14m",' in benchmark


def test_flow_pp3_highpass_completes_the_objective_factorial() -> None:
    source = _source(
        "tools/train/run_flow_pp3_hf_6h_pilot_cloudru.sh"
    )
    evaluation = _source(
        "tools/eval/run_upr_lite_6h_eval_queue_cloudru.sh"
    )
    transfer = _source(
        "tools/train/run_upr_lite_winner_12h_cloudru.sh"
    )
    replicates = _source(
        "tools/train/run_upr_lite_winner_replicates_cloudru.sh"
    )
    seed_evaluation = _source(
        "tools/eval/run_upr_lite_seed_eval_queue_cloudru.sh"
    )
    ood_spectra = _source(
        "tools/eval/run_upr_vs_pp3_ood_spectra_queue_cloudru.sh"
    )
    profile = _source("tools/train/architecture_candidate_profile.sh")

    assert "--arch \"$arch\"" in source
    assert 'arch="flow_pp3"' in source
    assert "--lambda_hf_override 0.05" in source
    assert "--expected-lambda-hf \"$lambda_hf\"" in source
    assert "--expected-batch-size-per-device 4" in source
    assert "--expected-accumulate-grad-batches 4" in source
    assert "--expected-trainer-sha256" in source
    assert "flow_pp3_hf_vs_nohf_2ep.json" in source
    assert 'started_marker="$LOG_ROOT/${exp_name}.started"' in source
    assert 'touch "$started_marker"' in source
    assert "upr_checkpoint=" not in source
    assert "flow_checkpoint=" not in source
    assert (
        '"$upr_two_epoch" upr_implicit_global_14m 2 3284 0.05'
        in source
    )
    assert '"$flow_two_epoch" flow_pp3 2 3284 0' in source
    assert 'upr_terminal="$LOG_ROOT/${upr_name}.terminal"' in source
    assert "strict UPR+HF reference failed" in source
    assert "assess_flow_spherical_ep_pilot.py" in source
    assert "stop_after_screen_checkpoint.py" in source
    assert "primary_reference_atmvfi_protocol_v2.started" in source
    assert "OOM fallback" not in source
    assert "[flow_pp3_hf]=" in evaluation
    assert "flow_pp3_hf_pilot.json" in evaluation
    assert "flow_hf_models=(flow_pp3_hf)" in evaluation
    assert "summarize_objective_factorial.py" in evaluation
    assert "upr_flow_objective_factorial.json" in evaluation
    assert '"${flow_hf_models[@]}"' in evaluation
    for downstream in (
        transfer,
        replicates,
        seed_evaluation,
        ood_spectra,
    ):
        assert "source tools/train/architecture_candidate_profile.sh" in downstream
        assert "architecture_candidate_model_arch" in downstream
    assert "flow_pp3_hf)" in profile
    assert "echo flow_pp3" in profile
    for consumer in (transfer, replicates, seed_evaluation):
        assert '--expected-arch "$model_arch"' in consumer
    assert (
        '"${CANDIDATE_CHECKPOINTS[$seed]}" "$model_arch" "$seed"'
        in ood_spectra
    )
    for trainer in (transfer, replicates):
        assert '"$candidate" == "amt"' in trainer
        assert '"$candidate" == "amt_residual"' in trainer
        assert '"$candidate" == "flow_pp3_hf"' in trainer
        assert '--arch "$model_arch"' in trainer


def test_upr_nohf_factorial_arm_is_preselection_and_locked() -> None:
    pilot = _source(
        "tools/train/"
        "run_upr_implicit_global_14m_nohf_6h_pilot_cloudru.sh"
    )
    postselection = _source(
        "tools/train/run_upr_lite_winner_nohf_cloudru.sh"
    )

    assert 'arch="upr_implicit_global_14m"' in pilot
    assert "--lambda_hf_override 0" in pilot
    assert "--expected-lambda-hf \"$lambda_hf\"" in pilot
    assert "stop_after_screen_checkpoint.py" in pilot
    assert "flow_pp3_nohf" in pilot
    assert "upr_implicit_global_14m_hf_vs_nohf_2ep.json" in pilot
    assert 'started_marker="$LOG_ROOT/${exp_name}.started"' in pilot
    assert 'touch "$started_marker"' in pilot
    assert "upr_checkpoint=" not in pilot
    assert "flow_checkpoint=" not in pilot
    assert '"$upr_two_epoch" "$arch" 2 3284 0.05' in pilot
    assert '"$flow_two_epoch" flow_pp3 2 3284 0' in pilot
    assert 'upr_terminal="$LOG_ROOT/${upr_name}.terminal"' in pilot
    assert "strict UPR+HF reference failed" in pilot
    assert "primary_reference_atmvfi_protocol_v2.started" in pilot
    assert "OOM fallback" not in pilot
    evaluation = _source(
        "tools/eval/run_upr_lite_6h_eval_queue_cloudru.sh"
    )
    assert "[upr_implicit_global_14m_nohf]=" in evaluation
    assert "upr_nohf_models=(upr_implicit_global_14m_nohf)" in evaluation
    assert '"${upr_nohf_models[@]}"' in evaluation
    shared_lock = '"$LOG_ROOT/${exp_name}.nohf_train.lock"'
    assert shared_lock in pilot
    assert shared_lock in postselection

    downstream_paths = (
        "tools/train/run_upr_lite_winner_12h_cloudru.sh",
        "tools/train/run_upr_lite_winner_replicates_cloudru.sh",
        "tools/eval/run_upr_lite_seed_eval_queue_cloudru.sh",
        "tools/eval/run_upr_vs_pp3_ood_spectra_queue_cloudru.sh",
    )
    profile = _source("tools/train/architecture_candidate_profile.sh")
    assert "upr_implicit_global_14m_nohf)" in profile
    assert "echo upr_implicit_global_14m" in profile
    for path in downstream_paths:
        source = _source(path)
        assert "source tools/train/architecture_candidate_profile.sh" in source
        assert "architecture_candidate_model_arch" in source


def test_nohf_avg3_keeps_zero_highpass_weight() -> None:
    evaluation = _source(
        "tools/eval/run_upr_lite_6h_eval_queue_cloudru.sh"
    )

    assert 'local name="${1%_avg3}"' in evaluation
    assert (
        "weatherbridge_ref|atmvfi_ref|upr_lite|"
        "upr_implicit_global_14m_nohf"
    ) in evaluation
    assert 'lambda_hf="$(expected_lambda_hf "$name")"' in evaluation


def test_pilot_promotions_are_recomputed_before_queue_allocation() -> None:
    evaluation = _source(
        "tools/eval/run_upr_lite_6h_eval_queue_cloudru.sh"
    )

    assert "validated_promotion()" in evaluation
    assert "tools/eval/pilot_assessment_status.py" in evaluation
    assert evaluation.index("export PYTHONPATH=") < evaluation.index(
        "validated_promotion()"
    )
    for candidate in (
        "upr_spherical_implicit_global_14m",
        "upr_endpoint_implicit_global_14m",
        "flow_spherical_ep",
        "upr_implicit_global_14m_nohf",
        "flow_pp3_hf",
        "upr_query_match_14m",
        "weather_amt_l",
    ):
        assert candidate in evaluation
    flow_hf_call = evaluation.split(
        'flow_hf_promoted="$(validated_promotion',
        maxsplit=1,
    )[1].split(')"', maxsplit=1)[0]
    assert "flow_pp3_hf" in flow_hf_call
    assert "upr_implicit_global_14m" in flow_hf_call
    assert "\n    1" not in flow_hf_call


def test_pilot_runners_validate_fresh_assessments() -> None:
    scripts = {
        "tools/train/run_flow_pp3_hf_6h_pilot_cloudru.sh": False,
        "tools/train/run_flow_spherical_ep_6h_pilot_cloudru.sh": True,
        "tools/train/"
        "run_upr_endpoint_implicit_global_14m_6h_pilot_cloudru.sh": False,
        "tools/train/"
        "run_upr_implicit_global_14m_nohf_6h_pilot_cloudru.sh": False,
        "tools/train/run_upr_query_match_14m_6h_pilot_cloudru.sh": False,
        "tools/train/"
        "run_upr_spherical_implicit_global_14m_6h_pilot_cloudru.sh": True,
        "tools/train/run_weather_amt_6h_pilot_cloudru.sh": True,
    }
    for path, allows_geometry in scripts.items():
        source = _source(path)
        assert "tools/eval/pilot_assessment_status.py" in source
        assert "--print-promoted" in source
        if allows_geometry:
            assert "--allow-geometry-rescue" in source


def test_architecture_screen_barrier_covers_all_modern_candidates() -> None:
    barrier = _source("tools/train/architecture_screen_barrier.sh")
    expected = (
        "exp_upr_query_match_14m_hf_135_14m_6h_s202707_protocol_v1",
        "exp_upr_spherical_implicit_global_14m_24ch_6h_2014_19_"
        "sparse135_lr1e4_eb16_s202707",
        "exp_flow_spherical_ep_14m_6h_s202707",
        "exp_weather_amt_l_14m_6h_s202707_protocol_v4",
    )

    for experiment in expected:
        assert barrier.count(experiment) == 1
    assert "release_candidate_gpu" in barrier
    assert 'touch "$LOG_ROOT/${current_experiment}.screened"' in barrier
    assert (
        '[[ ! -e "$LOG_ROOT/${experiment}.screened" && \\\n'
        '             ! -e "$LOG_ROOT/${experiment}.terminal" ]]'
    ) in barrier
    assert "acquire_candidate_gpu" in barrier


def test_architecture_pilots_yield_gpu_before_full_promotion() -> None:
    scripts = {
        "tools/train/run_upr_query_match_14m_6h_pilot_cloudru.sh":
            "promote QueryMatch to eight epochs",
        "tools/train/"
        "run_upr_spherical_implicit_global_14m_6h_pilot_cloudru.sh":
            'promote $arch to eight epochs',
        "tools/train/run_flow_spherical_ep_6h_pilot_cloudru.sh":
            'promote $arch to eight epochs',
        "tools/train/run_weather_amt_6h_pilot_cloudru.sh":
            'promote $arch to eight epochs',
    }

    for path, promotion_message in scripts.items():
        source = _source(path)
        barrier_call = (
            'wait_for_architecture_screen_barrier "$exp_name" "$queue_log"'
        )
        assert "source tools/train/architecture_screen_barrier.sh" in source
        assert 'screened_marker="$LOG_ROOT/${exp_name}.screened"' in source
        assert "trap mark_failed_exit EXIT" in source
        assert source.index(barrier_call) < source.index(promotion_message)


def test_strict_reference_yields_after_screen_checkpoint() -> None:
    source = _source(
        "tools/train/run_upr_implicit_global_14m_6h_cloudru.sh"
    )
    barrier_call = (
        'wait_for_architecture_screen_barrier "$exp_name" "$queue_log"'
    )

    assert "source tools/train/architecture_screen_barrier.sh" in source
    assert source.index("stop_after_screen_checkpoint.py") < source.index(
        barrier_call
    )
    assert source.index(barrier_call) < source.index(
        "\n  run_stage\n",
        source.index(barrier_call),
    )


def test_objective_ablations_wait_for_architecture_completion() -> None:
    paths = (
        "tools/train/run_flow_pp3_hf_6h_pilot_cloudru.sh",
        "tools/train/"
        "run_upr_implicit_global_14m_nohf_6h_pilot_cloudru.sh",
    )
    for path in paths:
        source = _source(path)
        wait_call = 'wait_for_architecture_completion "$queue_log"'
        first_acquire = source.index("\nacquire_gpu\n")
        assert "source tools/train/architecture_screen_barrier.sh" in source
        assert source.index(wait_call) < first_acquire


def test_incomplete_legacy_atmvfi_does_not_block_modern_selection() -> None:
    source = _source(
        "tools/eval/run_upr_lite_6h_eval_queue_cloudru.sh"
    )
    required = source.split(
        "required_models=(",
        maxsplit=1,
    )[1].split(")", maxsplit=1)[0]
    selection = source.split(
        "selection_models=(",
        maxsplit=1,
    )[1].split(")", maxsplit=1)[0]

    assert "atmvfi_ref" not in required
    assert "atmvfi_ref" not in selection
    assert "legacy_diagnostic_models=()" in source
    assert "skip incomplete optional ATM-VFI diagnostic" in source
    assert '"${legacy_diagnostic_models[@]}"' in source


def test_12h_legacy_atmvfi_is_optional_and_not_selectable() -> None:
    source = _source(
        "tools/eval/run_upr_lite_12h_transfer_queue_cloudru.sh"
    )
    selection = source.split(
        "selection_models=(",
        maxsplit=1,
    )[1].split(")", maxsplit=1)[0]

    assert "atmvfi_ref" not in selection
    assert "legacy_diagnostic_models=()" in source
    assert "skip incomplete optional 12h ATM-VFI diagnostic" in source
    assert 'models+=("${legacy_diagnostic_models[@]}")' in source
    assert 'models_csv="$(IFS=,; echo "${selection_models[*]}")"' in source


def test_12h_avg3_preserves_loss_and_trainer_provenance() -> None:
    source = _source(
        "tools/eval/run_upr_lite_12h_transfer_queue_cloudru.sh"
    )

    assert (
        'EXPECTED_LAMBDA_HF[$averaged_name]='
        '"${EXPECTED_LAMBDA_HF[$winner]}"'
    ) in source
    assert (
        'EXPECTED_TRAINER_SHA256[$averaged_name]='
        '"${EXPECTED_TRAINER_SHA256[$winner]}"'
    ) in source
    assert (
        '--expected-trainer-sha256 \\\n'
        '    "${EXPECTED_TRAINER_SHA256[$averaged_name]}"'
    ) in source


def test_12h_gate_preserves_optimizer_budget_across_batch_sizes() -> None:
    training = _source(
        "tools/train/run_upr_lite_winner_12h_cloudru.sh"
    )
    evaluation = _source(
        "tools/eval/run_upr_lite_12h_transfer_queue_cloudru.sh"
    )

    assert "optimizer_steps_per_epoch=1093" in training
    assert (
        "expected_train_batches_per_epoch="
        "$((optimizer_steps_per_epoch * accumulate))"
    ) in training
    assert "--expected-optimizer-steps-per-epoch 1093" in evaluation
    assert "--expected-train-batches-per-epoch" not in evaluation


def test_12h_trainers_pass_exact_microbatch_budget() -> None:
    paths = (
        "tools/train/run_upr_lite_winner_12h_cloudru.sh",
        "tools/train/run_weatherbridge_pp3_12h_matched_cloudru.sh",
        "tools/train/run_12h_reference_retrain_cloudru.sh",
    )

    for path in paths:
        source = _source(path)
        assert "optimizer_steps_per_epoch=1093" in source
        assert (
            "expected_train_batches_per_epoch="
            "$((optimizer_steps_per_epoch * accumulate))"
        ) in source
        assert (
            '--train_batches_per_epoch '
            '"$expected_train_batches_per_epoch"'
        ) in source
        assert "--expected-train-batches-per-epoch" in source


def test_12h_reference_trainers_use_frozen_entrypoint() -> None:
    expected_hash = (
        "e8d3bd5830ff7cc5a15dca986e94570c"
        "7e1b6a3707b8b7e3d5709dd17dcd6b6a"
    )
    paths = (
        "tools/train/run_weatherbridge_pp3_12h_matched_cloudru.sh",
        "tools/train/run_12h_reference_retrain_cloudru.sh",
    )

    for path in paths:
        source = _source(path)
        assert (
            'TRAINER_ENTRYPOINT="legacy/training_snapshots/'
            'train_capacity_matched_6h_e8d3bd58.py"'
        ) in source
        assert f'TRAINER_SHA256="{expected_hash}"' in source
        assert '"$PYTHON_BIN" -u "$TRAINER_ENTRYPOINT"' in source
        assert '--expected-trainer-sha256 "$TRAINER_SHA256"' in source


def test_12h_reference_retrain_freezes_training_protocol() -> None:
    source = _source(
        "tools/train/run_12h_reference_retrain_cloudru.sh"
    )
    expected_hash = (
        "ce3eac6d2f7eb6146c4880ce0e8ac2c"
        "201ed9710242003d86f1d724dc63a1e10"
    )

    assert 'TRAINING_PROTOCOL="tools/train/training_protocol.py"' in source
    assert (
        'TRAINING_PROTOCOL_SNAPSHOT="legacy/training_snapshots/'
        'training_protocol.py"'
    ) in source
    assert f'TRAINING_PROTOCOL_SHA256="{expected_hash}"' in source
    assert 'for protocol_path in "$TRAINING_PROTOCOL"' in source
    assert '"$TRAINING_PROTOCOL_SNAPSHOT"; do' in source
    assert (
        Path("legacy/training_snapshots/training_protocol.py").read_bytes()
        == Path("tools/train/training_protocol.py").read_bytes()
    )


def test_geometry_finalizer_releases_gpu_while_waiting() -> None:
    source = _source(
        "scripts/run_journal_baseline_geometry_v1_cloudru.sh"
    )
    acquire = "acquire_gpu"
    release = "release_gpu"
    finalizer = 'if [[ "$FINALIZE" -ne 1 ]]; then'
    wait_for_artifacts = "while ! ("

    assert source.count(f"{acquire}()") == 1
    assert source.count(f"{release}()") == 1
    evaluate_body = source[source.index("evaluate_model()") :]
    assert evaluate_body.index(acquire) < evaluate_body.index(
        '"$PY" -u tools/eval/batch_eval_12h_memmap_fast.py'
    )
    release_call = source.index(f"{release}\n", source.index("if has_horizon 12"))
    assert release_call < source.index(finalizer)
    assert release_call < source.index(wait_for_artifacts)


def test_12h_pp3_waits_for_frozen_schema13_selection() -> None:
    source = _source(
        "tools/train/run_weatherbridge_pp3_12h_matched_cloudru.sh"
    )

    assert (
        'SELECTION="${SELECTION:-metrics/upr_lite_candidate_selection.json}"'
        in source
    )
    assert "wait fresh selection=$SELECTION" in source
    assert "validate_frozen_selection_for_followup" in source
    assert source.index("validate_frozen_selection_for_followup") < source.index(
        'while [[ ! -e "$UPR_WORKER_MARKER" ]]'
    )


def test_12h_postselection_suite_prioritizes_quality_and_flow() -> None:
    source = _source(
        "tools/train/run_upr_lite_12h_postselection_suite_cloudru.sh"
    )

    assert "validate_frozen_selection_for_followup" in source
    assert "choose_transfer_candidate" in source
    assert "run_weatherbridge_pp3_12h_matched_cloudru.sh" in source
    assert "REFERENCE_ARCH=dcae_14m" in source
    assert "run_upr_lite_12h_transfer_queue_cloudru.sh" in source
    assert "run_temporal_expert_router_12h_queue_cloudru.sh" in source
    primary_wait = source.index("wait primary 12h quality=")
    efficiency_launch = source.index("TRANSFER_OBJECTIVE=efficiency")
    flow_launch = source.index("TRANSFER_OBJECTIVE=flow")
    assert primary_wait < efficiency_launch
    assert primary_wait < flow_launch
    assert "REFERENCE_ARCH=atmvfi" not in source


def test_compact_vp3_short_budget_screen_is_compute_and_flow_gated() -> None:
    source = _source(
        "tools/train/run_flow_compact_vp3_6h_followup_cloudru.sh"
    )

    assert "flow_compact_vp3" in source
    assert "flow_compact_vp3_m" in source
    assert "upr_lite_implicit_global" in source
    assert "UPR_SCREEN_CSV=" in source
    assert "UPR_FINAL_CSV=" in source
    assert "validate_frozen_selection_for_followup" not in source
    assert "flow_compact_vp3_compute_a5000.json" in source
    assert "flow_compact_vp3_m_compute_a5000.json" in source
    assert 'result["parameters"] <= int(sys.argv[3])' in source
    assert 'result["inference_ms_median"] <= float(sys.argv[4])' in source
    assert 'result["training_ms_median"] <= float(sys.argv[5])' in source
    assert "train_capacity_matched_6h_38ca4e17.py" in source
    assert "--train_batches_per_epoch 6568" in source
    assert "--expected-train-batches-per-epoch 6568" in source
    assert "stop_after_screen_checkpoint.py" in source
    assert "--held-relative-limit 0.03" in source
    assert "--all-hour-relative-limit 0.03" in source
    assert "--per-hour-relative-limit 0.08" in source
    assert "--epoch 3" in source
    assert "--held-relative-limit 0" in source
    assert "select_compact_flow_candidate.py" in source
    assert 'PREFLIGHT_ONLY="${PREFLIGHT_ONLY:-0}"' in source
    assert "exp_flow_pp3_135_14m_6h_s202707_protocol_v2" in source
    assert "--expected-highpass-boundary antipodal_vector_parity" in source
    assert "--expected-highpass-boundary periodic_lon_replicate_lat" in source
    assert "--test-year 2021" not in source


def test_compact_short_budget_eval_uses_frozen_winner_and_ood() -> None:
    source = _source(
        "tools/eval/run_compact_flow_short_budget_eval_cloudru.sh"
    )

    assert "four_epoch_selection.json" in source
    assert 'FINAL_REPORT="$METRICS_ROOT/' in source
    assert "epoch=3-step=6568.ckpt" in source
    assert "flow_4ep_ref" in source
    assert "for year in 2020 2021" in source
    assert "--save-window-metrics" in source
    assert "--save-physical-metrics" in source
    assert "--save-temporal-metrics" in source
    assert "paired_block_bootstrap.py" in source
    assert "spectral_block_bootstrap.py" in source
    assert "--taus 1,2,3,4,5" in source
    assert "compact_flow_spectra_6h_${year}" in source


def test_under10m_search_is_full_sample_compute_gated() -> None:
    source = _source(
        "tools/train/run_flow_under10m_6h_search_cloudru.sh"
    )

    assert "flow_compact_vp3_l" in source
    assert "flow_compact_hermite_l" in source
    assert "train_capacity_matched_6h_08c8a559.py" in source
    assert "08c8a559f84b363245c26607fe1a5047" in source
    assert "flow_compact_vp3_l_compute_a5000.json" in source
    assert "flow_compact_hermite_l_compute_a5000.json" in source
    assert "[flow_compact_vp3_l]=8800000" in source
    assert "[flow_compact_hermite_l]=8800000" in source
    assert "--train_batches_per_epoch 6568" in source
    assert "--expected-train-batches-per-epoch 6568" in source
    assert "--samples_per_date_train 4" in source
    assert "--samples_per_date_val 2" in source
    assert "--expected-effective-batch-size 16" in source
    assert "train_until flow_compact_vp3_l 2 3284" in source
    assert "train_until flow_compact_hermite_l 2 3284" in source
    assert 'train_until "$selected" 4 6568' in source
    assert "exp_flow_pp3_135_14m_6h_s202707_protocol_v2" in source
    assert "upr_lite_implicit_global" not in source
    assert "--test-year 2021" not in source


def test_under10m_eval_uses_only_the_frozen_four_epoch_winner() -> None:
    source = _source(
        "tools/eval/run_under10m_flow_eval_cloudru.sh"
    )

    assert "under10m_flow_screen_6h/four_epoch_selection.json" in source
    assert "under10m_flow_search.complete" in source
    assert "flow_compact_vp3_l)" in source
    assert "flow_compact_hermite_l)" in source
    assert source.count("epoch=3-step=6568.ckpt") == 3
    assert "flow_4ep_ref" in source
    assert "for year in 2020 2021" in source
    assert "--full-year" in source
    assert "--save-window-metrics" in source
    assert "--save-physical-metrics" in source
    assert "--save-temporal-metrics" in source
    assert "--channels all" in source
    assert "--taus 1,2,3,4,5" in source
    assert "paired_block_bootstrap.py" in source
    assert "spectral_block_bootstrap.py" in source
    assert "upr_lite" not in source
    assert "flow_compact_vp3_m" not in source


def test_base_field_finetune_keeps_sparse_full_sample_protocol() -> None:
    source = _source(
        "tools/train/run_flow_base_field_finetune_cloudru.sh"
    )

    assert "train_capacity_matched_6h_08e77586.py" in source
    assert "08e77586ae33e16c79582832561421da" in source
    assert "base_balanced" in source
    assert "base_edge_balanced" in source
    assert "--init_weights_path \"$INITIAL_CHECKPOINT\"" in source
    assert 'assert "resume_lineage" not in hparams' in source
    assert "--train_tau_subset 1 3 5" in source
    assert "--eval_tau 1 2 3 4 5" in source
    assert "--samples_per_date_train 4" in source
    assert "--samples_per_date_val 2" in source
    assert "--train_batches_per_epoch 6568" in source
    assert "--expected-effective-batch-size 16" in source
    assert "--max_epochs 1" in source
    assert "--warmup_steps 100" in source
    assert "--lambda_hf_override 0.02" in source
    assert "d71a7a5e803cd1fbc7d060010ce240ca" in source
    assert "flow_compact_vp3" not in source


def test_base_field_eval_uses_fixed_non_q_selection_protocol() -> None:
    source = _source(
        "tools/eval/run_base_field_finetune_eval_cloudru.sh"
    )

    assert "base_field_finetune.complete" in source
    assert "base_balanced:$BASE_CHECKPOINT" in source
    assert "base_edge_balanced:$EDGE_CHECKPOINT" in source
    assert "flow_4ep_ref:$FLOW_CHECKPOINT" in source
    assert "--test-year 2020" in source
    assert "--samples-per-date 4" in source
    assert "--eval-days-per-month 8" in source
    assert "--eval-hours 1,2,3,4,5" in source
    assert "--seen-tau 1,3,5" in source
    assert "--unseen-tau 2,4" in source
    assert "--no-acc" in source
    assert "select_base_field_candidate.py" in source
    assert "--base-mean-limit 0.03" in source
    assert "--held-mean-limit 0.01" in source
    assert "one_epoch_selection.json" in source
    assert "--full-year" not in source


def test_lagrange_expert_is_warm_started_under_10m_and_sparse_tau() -> None:
    source = _source(
        "tools/train/run_flow_lagrange_base_expert_cloudru.sh"
    )

    assert "train_capacity_matched_6h_cf4a8519.py" in source
    assert "cf4a85190bac7bcdf14e5b8d49de492" in source
    assert "flow_compact_lagrange_l" in source
    assert "result[\"parameters\"] == 8806801" in source
    assert "--init_weights_path \"$INITIAL_CHECKPOINT\"" in source
    assert 'compatibility["source_arch"] == "flow_compact_hermite_l"' in source
    assert 'compatibility["target_arch"] == "flow_compact_lagrange_l"' in source
    assert "--train_tau_subset 1 3 5" in source
    assert "--eval_tau 1 2 3 4 5" in source
    assert "--samples_per_date_train 4" in source
    assert "--train_batches_per_epoch 6568" in source
    assert "--max_epochs 1" in source
    assert "flow_compact_vp3" not in source


def test_lagrange_eval_uses_same_flow_and_non_q_gate() -> None:
    source = _source(
        "tools/eval/run_lagrange_base_expert_eval_cloudru.sh"
    )

    assert "lagrange_base_expert.complete" in source
    assert "lagrange_base_balanced:$BASE_CHECKPOINT" in source
    assert "lagrange_base_edge_balanced:$EDGE_CHECKPOINT" in source
    assert "flow_4ep_ref:$FLOW_CHECKPOINT" in source
    assert "--eval-days-per-month 8" in source
    assert "--eval-hours 1,2,3,4,5" in source
    assert "--seen-tau 1,3,5" in source
    assert "--unseen-tau 2,4" in source
    assert "select_base_field_candidate.py" in source
    assert "lagrange_base_expert_6h/one_epoch_selection.json" in source
    assert "--full-year" not in source


def test_lagrange_head_only_uses_frozen_four_epoch_backbone() -> None:
    source = _source(
        "tools/train/run_flow_lagrange_head_only_cloudru.sh"
    )

    assert "train_capacity_matched_6h_7c4c9279.py" in source
    assert "7c4c9279f5adf323cd9ee2f5bc40605" in source
    assert "epoch=3-step=6568.ckpt" in source
    assert "c567ff58c92725d2059d8c6efbab5f" in source
    assert "--trainable_scope base_knot_head" in source
    assert "--loss_profile base_edge_balanced" in source
    assert "--lambda_hf_override 0" in source
    assert '[lr3e4]="3e-4"' in source
    assert '[lr1e3]="1e-3"' in source
    assert 'lineage["epoch"] == 3' in source
    assert 'lineage["global_step"] == 6568' in source
    assert "--train_tau_subset 1 3 5" in source
    assert "--train_batches_per_epoch 6568" in source


def test_lagrange_head_only_requires_strict_gate_before_full_year() -> None:
    source = _source(
        "tools/eval/run_lagrange_head_only_eval_cloudru.sh"
    )

    assert "lagrange_head_only.complete" in source
    assert "lagrange_head_lr3e4:$BASE_CHECKPOINT" in source
    assert "lagrange_head_lr1e3:$EDGE_CHECKPOINT" in source
    assert source.count("--base-mean-limit 0") == 2
    assert source.count("--seen-mean-limit 0") == 2
    assert source.count("--held-mean-limit 0") == 2
    assert source.count("--per-tau-limit 0") == 2
    assert source.count("--surface-mean-limit 0") == 2
    assert source.index('if [[ -z "$selected" ]]') < source.index(
        "--full-year"
    )
    assert "full_year_selection.json" in source
    assert "lagrange_head_winner:$winner_checkpoint" in source


def test_primary_references_can_run_independently_of_geometry_pilot() -> None:
    source = _source(
        "tools/train/run_primary_reference_retrains_cloudru.sh"
    )

    assert (
        'WAIT_FOR_FLOW_SPHERICAL_TERMINAL="${'
        'WAIT_FOR_FLOW_SPHERICAL_TERMINAL:-1}"'
    ) in source
    assert 'case "$WAIT_FOR_FLOW_SPHERICAL_TERMINAL" in' in source
    assert "strict references run independently of Flow-Spherical" in source
    assert "WAIT_FOR_FLOW_SPHERICAL_TERMINAL must be 0 or 1" in source
    assert 'MIN_FREE_MIB="${MIN_FREE_MIB:-76000}"' in source
    assert "--expected-batch-size-per-device 4" in source
    assert "--expected-accumulate-grad-batches 4" in source
    assert "--expected-highpass-boundary periodic_lon_replicate_lat" in source
    assert "--expected-trainer-sha256" in source
    assert "intrinsic OOM violates frozen bs=4" in source
    assert "fallback bs=" not in source
    assert "primary_reference_atmvfi_protocol_v2.started" in source


def test_upr_hf_reference_is_fresh_and_frozen() -> None:
    source = _source(
        "tools/train/run_upr_implicit_global_14m_6h_cloudru.sh"
    )

    strict_name = (
        "exp_upr_implicit_global_14m_hf_135_14m_6h_"
        "s202707_protocol_v2"
    )
    assert strict_name in source
    assert "--lambda_hf_override 0.05" in source
    assert "--expected-lambda-hf 0.05" in source
    assert "--expected-batch-size-per-device 4" in source
    assert "--expected-accumulate-grad-batches 4" in source
    assert "--expected-highpass-boundary periodic_lon_replicate_lat" in source
    assert "--expected-trainer-sha256" in source
    assert 'touch "$started_marker"' in source
    assert "trap mark_failed_exit EXIT" in source
    assert 'touch "$terminal_marker"' in source
    assert "intrinsic OOM violates frozen bs=4" in source
    assert "fallback bs=" not in source

    consumers = (
        "tools/train/run_flow_pp3_hf_6h_pilot_cloudru.sh",
        "tools/train/"
        "run_upr_implicit_global_14m_nohf_6h_pilot_cloudru.sh",
        "tools/train/run_weather_amt_6h_pilot_cloudru.sh",
        "tools/train/"
        "run_upr_endpoint_implicit_global_14m_6h_pilot_cloudru.sh",
        "tools/train/"
        "run_upr_spherical_implicit_global_14m_6h_pilot_cloudru.sh",
        "tools/train/run_flow_spherical_ep_6h_pilot_cloudru.sh",
        "tools/eval/run_upr_lite_6h_eval_queue_cloudru.sh",
    )
    for path in consumers:
        assert strict_name in _source(path)


def test_geometry_pilots_use_frozen_microbatch_and_source_protocol() -> None:
    paths = (
        "tools/train/"
        "run_upr_spherical_implicit_global_14m_6h_pilot_cloudru.sh",
        "tools/train/run_flow_spherical_ep_6h_pilot_cloudru.sh",
        "tools/train/"
        "run_upr_endpoint_implicit_global_14m_6h_pilot_cloudru.sh",
        "tools/train/run_weather_amt_6h_pilot_cloudru.sh",
    )
    for path in paths:
        source = _source(path)
        assert 'MIN_FREE_MIB="${MIN_FREE_MIB:-76000}"' in source
        assert "--expected-batch-size-per-device 4" in source
        assert "--expected-accumulate-grad-batches 4" in source
        assert "--expected-trainer-sha256" in source
        assert "intrinsic OOM violates frozen bs=4" in source
        assert "OOM fallback" not in source
        assert "pilot OOM fallback" not in source
        assert "PP3_START_MARKER" in source
        assert "ATMVFI_START_MARKER" in source
        assert "wait primary reference priority marker" in source

    evaluation = _source(
        "tools/eval/run_upr_lite_6h_eval_queue_cloudru.sh"
    )
    assert "checkpoint_extra_args()" in evaluation
    assert "terminal without assessment" in evaluation
    assert (
        evaluation.count(
            '--expected-trainer-sha256 "$FROZEN_TRAINER_SHA256"'
        )
        >= 2
    )


def test_temporal_router_is_frozen_on_2020_before_ood() -> None:
    source = _source(
        "tools/eval/run_temporal_expert_router_6h_queue_cloudru.sh"
    )

    assert "select_temporal_expert_route.py" in source
    assert "--default-expert auto" in source
    assert "--min-relative-gain 0.005" in source
    assert "--bootstrap-draws 5000" in source
    assert source.count("--block-days 7") == 2
    assert "--extreme-quantile 0.95" in source
    assert "build_temporal_router_checkpoint.py" in source
    assert "temporal_router_checkpoint_status.py" in source
    selection_offset = source.index("select_temporal_expert_route.py")
    ood_offset = source.index("for year in 2020 2021")
    assert selection_offset < ood_offset
    assert "--test-year \"$year\"" in source
    assert "--full-year" in source
    assert "--save-window-metrics" in source
    assert "--save-physical-metrics" in source
    assert "--save-temporal-metrics" in source
    assert "--lmax 359" in source
    assert "temporal_router_inference_cost_batch1_all_tau.json" in source
    assert "temporal_router_inference_cost_batch5_mixed.json" in source
    assert source.count("--tau-values") == 2
    assert "assess_temporal_expert_router.py" in source
    assert "temporal_router_generalization.json" in source
    unlock_offset = source.index('flock -u "$lock_fd"')
    source_ood_wait_offset = source.index(
        'for source_name in "$UPR_NAME" "$FLOW_NAME"'
    )
    assert unlock_offset < source_ood_wait_offset


def test_temporal_router_12h_uses_transfer_artifacts_and_unseen_spectra() -> None:
    source = _source(
        "tools/eval/run_temporal_expert_router_12h_queue_cloudru.sh"
    )

    assert "choose_transfer_candidate(selection, \"quality\")" in source
    assert "upr_lite_transfer_12h_2020/quality_transfer.json" in source
    assert "upr_lite_transfer_12h_2020/weatherbridge_ref.json" in source
    assert "--default-expert auto" in source
    assert source.count("--block-days 7") == 2
    assert "--extreme-quantile 0.95" in source
    assert "--required-taus 1,2,3,4,5,6,7,8,9,10,11" in source
    assert "--seen-tau 1,2,3,5,7,9,10,11" in source
    assert "--unseen-tau 4,6,8" in source
    assert "--spectral-taus 4,6,8" in source
    assert "--tau-values \"$all_tau_values\"" in source
    assert "--tau-values \"$mixed_tau_values\"" in source
    selection_offset = source.index("select_temporal_expert_route.py")
    ood_offset = source.index("for year in 2020 2021")
    assert selection_offset < ood_offset
    unlock_offset = source.index('flock -u "$lock_fd"')
    source_ood_wait_offset = source.index(
        'wait_field_artifact quality_2021'
    )
    assert unlock_offset < source_ood_wait_offset
    assert "temporal_router_generalization_12h.json" in source


def test_12h_keeps_promoted_flow_as_geometry_control() -> None:
    training = _source(
        "tools/train/run_upr_lite_winner_12h_cloudru.sh"
    )
    evaluation = _source(
        "tools/eval/run_upr_lite_12h_transfer_queue_cloudru.sh"
    )

    assert 'if objective == "flow":' in training
    assert (
        'for name in ("flow_spherical_ep", "flow_pp3_hf")'
        in training
    )
    assert "flow_candidate=" in evaluation
    assert "CHECKPOINTS[flow_transfer]" in evaluation
    assert "models+=(flow_transfer)" in evaluation
    assert "candidate_arch()" in evaluation
    assert "candidate_lambda_hf()" in evaluation
    assert "EXPECTED_LAMBDA_HF" in evaluation
    assert "source tools/train/architecture_candidate_profile.sh" in evaluation
    assert "architecture_candidate_layout" in evaluation
    assert "architecture_candidate_model_arch" in evaluation


def test_primary_12h_trainers_require_corrected_anchor_index_source() -> None:
    expected_hash = (
        "cc5c21f1150e9cdfa9acb1746557db88"
        "a080b09da3e7f8caa4207a25d60fc1fa"
    )
    for relative in (
        "tools/train/run_weatherbridge_pp3_12h_matched_cloudru.sh",
        "tools/train/run_12h_reference_retrain_cloudru.sh",
        "tools/train/run_upr_lite_winner_12h_cloudru.sh",
    ):
        source = _source(relative)
        assert 'MEMMAP_DATASET="weather_time_interp/memmap_dataset.py"' in source
        assert f'MEMMAP_DATASET_SHA256="{expected_hash}"' in source
        assert 'sha256sum "$MEMMAP_DATASET"' in source
        assert (
            '"$actual_memmap_sha256" != "$MEMMAP_DATASET_SHA256"'
            in source
        )


def test_nohf_queue_reports_architecture_only_significance() -> None:
    source = _source(
        "tools/train/run_upr_lite_winner_nohf_cloudru.sh"
    )

    assert "summarize_highpass_ablation.py" in source
    assert "--primary-root-2021 metrics/upr_lite_screen_6h_2021" in source
    assert "--nohf-root-2021 metrics/upr_lite_nohf_6h_2021" in source
    assert "--unseen-taus 2,4" in source
    assert "--draws 5000" in source
    assert '--expected-batch-size-per-device "$batch_size"' in source
    assert '--expected-accumulate-grad-batches "$accumulate"' in source
    assert "--expected-highpass-boundary" in source
    assert "--expected-trainer-sha256" in source
    assert "intrinsic OOM violates frozen bs=$batch_size" in source
    assert "intrinsic OOM fallback" not in source


def test_nohf_multiseed_runs_only_after_matched_objective_promotion() -> None:
    training = _source(
        "tools/train/run_upr_lite_nohf_replicates_cloudru.sh"
    )
    evaluation = _source(
        "tools/eval/run_upr_nohf_vs_pp3_seed_eval_queue_cloudru.sh"
    )

    for source in (training, evaluation):
        assert "architecture_only_single_seed_promotion_passed" in source
        assert 'ablation.get("training_seed_count") == 1' in source
        assert '--expected-batch-size-per-device "$expected_batch_size"' in source
        assert '--expected-accumulate-grad-batches "$expected_accumulate"' in source
        assert "--expected-lambda-hf 0" in source
        assert "--expected-trainer-sha256" in source
    assert "run_seed 202708" in training
    assert "run_seed 202709" in training
    assert "run_seed 202707" not in training
    assert "no-HF promotion checkpoint failed protocol validation" in training
    assert "PP3 promotion checkpoint failed protocol validation" in training
    assert "intrinsic OOM violates frozen bs=$batch_size" in training
    assert "--candidate-artifact-name" in evaluation
    assert "--candidate-lambda-hf 0" in evaluation
    assert "--reference-lambda-hf 0" in evaluation
    assert "--save-temporal-metrics" in evaluation


def test_nohf_architecture_gets_independent_2021_spectral_confirmation() -> None:
    source = _source(
        "tools/eval/"
        "run_upr_nohf_vs_pp3_ood_spectra_queue_cloudru.sh"
    )

    assert "upr_nohf_vs_pp3_seed_comparison.json" in source
    assert 'objective.get("matched") is True' in source
    assert "primary_quality_superiority_seed_consistent" in source
    assert "--test-year 2021" in source
    assert "--candidate-artifact-name" in source
    assert "--candidate-lambda-hf 0" in source
    assert "--reference-lambda-hf 0" in source
    assert "--reference-spectra-root" in source
    assert "--seed-comparison" in source
    assert "--expected-trainer-sha256" in source


def test_12h_transfer_includes_forecast_and_exchange_generalization() -> None:
    evaluator = _source("tools/eval/batch_eval_forecast_anchor.py")
    evaluation_6h = _source(
        "tools/eval/run_upr_lite_6h_eval_queue_cloudru.sh"
    )
    source = _source(
        "tools/eval/run_upr_lite_12h_transfer_queue_cloudru.sh"
    )

    assert "select_anchor_pairs(" in evaluator
    assert "left_lead + delta_t_hours" in evaluator
    assert "--delta-t-hours 12" in source
    assert "upr_lite_forecast_anchor_12h_2021_v1" in source
    assert "summarize_forecast_anchor.py" in source
    assert "upr_lite_anchor_exchange_12h_2020.json" in source
    assert "upr_lite_anchor_exchange_12h_2021.json" in source
    assert "upr_lite_region_season_summary_12h.json" in source
    assert "upr_lite_transfer_spectra_12h_2021" in source
    assert "summarize_frozen_ood_spectra.py" in source
    assert "upr_lite_transfer_ood_spectra_12h_2021.json" in source
    assert "--samples-per-date 2" in source
    assert "--eval-days-per-month 8" in source
    assert "upr_lite_region_season_summary_6h.json" in evaluation_6h
    assert '--models "$exchange_models_csv"' in evaluation_6h
    assert '--models "$models_csv"' in evaluation_6h
    assert "--samples-per-date 2" in evaluation_6h
    assert "--eval-days-per-month 8" in evaluation_6h
    assert "--all-taus 1,2,3,4,5,6,7,8,9,10,11" in source
    assert "--unseen-taus 4,6,8" in source
    for evaluation in (evaluation_6h, source):
        assert "--save-temporal-metrics" in evaluation
        assert "--require-temporal-metrics" in evaluation


def test_journal_queue_requires_confirmed_6h_and_12h_selections() -> None:
    source = _source("scripts/run_journal_final_eval_queue_cloudru.sh")

    assert source.count("tools/eval/check_validation_gate.py") == 2
    assert source.count(
        "tools/eval/check_postselection_generalization_gate.py"
    ) == 2
    assert "upr_lite_candidate_validation.json" in source
    assert "upr_lite_transfer_validation_12h.json" in source
    assert "upr_lite_forecast_anchor_2021_v2/summary.json" in source
    assert "upr_lite_forecast_anchor_12h_2021_v1/summary.json" in source
    assert "upr_lite_region_season_summary_6h.json" in source
    assert "upr_lite_region_season_summary_12h.json" in source
    assert "--ood-spectral-summary" in source
    assert "exp_flow_pp3_135_14m_6h_s202707_protocol_v2" in source
    assert (
        "exp_atmvfi_24ch_6h_2014_19_sparse135_lr1e4_"
        "s202707_protocol_v2"
    ) in source


def test_upr14m_avg3_is_built_and_evaluated_as_auxiliary() -> None:
    source = _source(
        "tools/eval/run_upr_lite_6h_eval_queue_cloudru.sh"
    )

    assert '--checkpoint-dir "$upr14m_dir"' in source
    assert (
        '--output "${CHECKPOINTS[upr_implicit_global_14m_avg3]}"'
        in source
    )
    assert 'for name in "${eval_models[@]}"; do' in source
    assert "upr_implicit_global_14m_avg3" not in source.split(
        "selection_models=(",
        maxsplit=1,
    )[1].split(")", maxsplit=1)[0]


def test_endpoint_control_is_zero_shot_allocation_only() -> None:
    source = _source(
        "tools/eval/run_upr_endpoint_zero_shot_queue_cloudru.sh"
    )

    assert "--min-epochs 8" in source
    assert "--expected-arch upr_implicit_global_14m" in source
    assert "--expected-batch-size-per-device 4" in source
    assert "--expected-accumulate-grad-batches 4" in source
    assert "--expected-highpass-boundary periodic_lon_replicate_lat" in source
    assert "--expected-trainer-sha256" in source
    assert "make_state_compatible_control_checkpoint.py" in source
    assert "--target-arch upr_endpoint_implicit_global_14m" in source
    assert "--test-year 2020" in source
    assert "--eval-days-per-month 4" in source
    assert "--eval-hours 1,2,3,4,5" in source
    assert "assess_upr_endpoint_zero_shot.py" in source
    assert "--test-year 2021" not in source
    assert "--full-year" not in source
    assert "train_capacity_matched_6h.py" not in source


def test_endpoint_promotion_reaches_full_generalization_chain() -> None:
    pilot = _source(
        "tools/train/"
        "run_upr_endpoint_implicit_global_14m_6h_pilot_cloudru.sh"
    )
    evaluation = _source(
        "tools/eval/run_upr_lite_6h_eval_queue_cloudru.sh"
    )
    replicates = _source(
        "tools/train/run_upr_lite_winner_replicates_cloudru.sh"
    )
    nohf = _source(
        "tools/train/run_upr_lite_winner_nohf_cloudru.sh"
    )
    spectra = _source(
        "tools/eval/run_upr_vs_pp3_ood_spectra_queue_cloudru.sh"
    )
    transfer = _source(
        "tools/train/run_upr_lite_winner_12h_cloudru.sh"
    )
    transfer_eval = _source(
        "tools/eval/run_upr_lite_12h_transfer_queue_cloudru.sh"
    )

    name = "upr_endpoint_implicit_global_14m"
    assert name in pilot
    assert "--require-resume-lineage" in pilot
    assert 'actual_source_sha="$(sha256sum "$reference_checkpoint"' in pilot
    assert "endpoint allocation is stale" in pilot
    assert "run_stage 8 &" in pilot
    assert name in evaluation
    assert "build_avg3_auxiliary upr_endpoint_implicit_global_14m" in evaluation
    assert name in nohf
    profile = _source("tools/train/architecture_candidate_profile.sh")
    assert name in profile
    for source in (transfer, transfer_eval):
        assert "source tools/train/architecture_candidate_profile.sh" in source
    assert "choose_transfer_candidate" in replicates
    assert '--expected-arch "$model_arch"' in replicates
    assert "choose_transfer_candidate" in spectra
    assert '--expected-arch "$arch"' in spectra
    assert '"${CANDIDATE_CHECKPOINTS[$seed]}" "$model_arch"' in spectra


def test_promoted_geometry_and_amt_avg3_are_deployment_only() -> None:
    source = _source(
        "tools/eval/run_upr_lite_6h_eval_queue_cloudru.sh"
    )

    assert "build_avg3_auxiliary upr_spherical_implicit_global_14m" in source
    assert "build_avg3_auxiliary flow_spherical_ep" in source
    assert "build_avg3_auxiliary amt" in source
    eval_models = source.split(
        "eval_models=(",
        maxsplit=1,
    )[1].split(")", maxsplit=1)[0]
    selection_models = source.split(
        "selection_models=(",
        maxsplit=1,
    )[1].split(")", maxsplit=1)[0]
    assert '"${spherical_upr_aux_models[@]}"' in eval_models
    assert '"${flow_aux_models[@]}"' in eval_models
    assert '"${amt_aux_models[@]}"' in eval_models
    assert "spherical_upr_aux_models" not in selection_models
    assert "flow_aux_models" not in selection_models
    assert "amt_aux_models" not in selection_models
    assert "tools/eval/assess_avg3_deployment.py" in source
    assert "upr_lite_avg3_deployment.json" in source


def test_12h_avg3_is_built_only_after_raw_selection() -> None:
    source = _source(
        "tools/eval/run_upr_lite_12h_transfer_queue_cloudru.sh"
    )

    selection_offset = source.index(
        "--out-json metrics/upr_lite_transfer_selection_12h.json"
    )
    average_offset = source.index(
        "tools/train/average_checkpoints.py"
    )
    assert selection_offset < average_offset
    assert 'averaged_name="${winner}_avg3"' in source
    assert 'for name in "${models[@]}" "$averaged_name"; do' in source
    assert "--unseen-taus 4,6,8" in source
    assert "--spectral-taus 4,6,8" in source
    assert "upr_lite_transfer_avg3_deployment_12h.json" in source
    raw_models = source.split(
        "models=(quality_transfer)",
        maxsplit=1,
    )[1].split(
        'models_csv="$(IFS=,; echo "${selection_models[*]}")"',
        maxsplit=1,
    )[0]
    assert "averaged_name" not in raw_models
