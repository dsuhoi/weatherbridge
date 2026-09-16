from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
EVAL_QUEUE = ROOT / "tools/eval/run_upr_lite_6h_eval_queue_cloudru.sh"
PILOT_SCRIPTS = (
    ROOT / "tools/train/run_weather_amt_6h_pilot_cloudru.sh",
    ROOT / "tools/train/run_flow_spherical_ep_6h_pilot_cloudru.sh",
    ROOT
    / "tools/train/run_upr_spherical_implicit_global_14m_6h_pilot_cloudru.sh",
)
ENDPOINT_PILOT = (
    ROOT
    / "tools/train/run_upr_endpoint_implicit_global_14m_6h_pilot_cloudru.sh"
)
ALL_PILOT_SCRIPTS = PILOT_SCRIPTS + (ENDPOINT_PILOT,)
IMPORTING_PILOT_SCRIPTS = ALL_PILOT_SCRIPTS + (
    ROOT / "tools/train/run_flow_pp3_hf_6h_pilot_cloudru.sh",
    ROOT
    / "tools/train/run_upr_implicit_global_14m_nohf_6h_pilot_cloudru.sh",
)


@pytest.mark.parametrize("script", ALL_PILOT_SCRIPTS)
def test_two_epoch_pilot_preserves_full_training_schedule(
    script: Path,
) -> None:
    text = script.read_text()

    assert "run_stage 2" not in text
    assert "run_stage 8 &" in text
    assert "stop_after_screen_checkpoint.py" in text
    assert "--min-global-step 3284" in text
    assert "--expected-total-steps 3284" not in text


@pytest.mark.parametrize("script", ALL_PILOT_SCRIPTS)
def test_two_epoch_pilot_selects_completed_validation_csv(
    script: Path,
) -> None:
    text = script.read_text()

    assert "select_validation_metrics_csv.py" in text
    assert "--required-epoch 1" in text
    assert "--required-step 3283" in text
    assert '--pilot-csv "$pilot_csv"' in text
    assert (
        '--reference-csv "$upr_csv"' in text
        or '--reference-csv "$reference_csv"' in text
    )
    assert "lightning_logs/version_0/metrics.csv" not in text


def test_weather_amt_matches_reference_high_pass_loss() -> None:
    text = PILOT_SCRIPTS[0].read_text()

    assert "--expected-lambda-hf 0.05" in text
    assert "--lambda_hf_override 0.05" in text


@pytest.mark.parametrize("script", PILOT_SCRIPTS)
def test_spherical_pilot_binds_antipodal_highpass_boundary(
    script: Path,
) -> None:
    text = script.read_text()
    common = text.split("protocol_args=(", maxsplit=1)[1].split(
        "candidate_protocol_args=(",
        maxsplit=1,
    )[0]
    candidate = text.split(
        "candidate_protocol_args=(",
        maxsplit=1,
    )[1].split(")", maxsplit=1)[0]

    assert "--expected-highpass-boundary" not in common
    assert "--expected-highpass-boundary antipodal_vector_parity" in candidate
    assert text.count('"${candidate_protocol_args[@]}"') >= 5
    reference_variable = (
        "$reference_checkpoint"
        if "$reference_checkpoint" in text
        else "$upr_checkpoint"
    )
    reference_gate = text.split(
        f'"{reference_variable}"',
        maxsplit=1,
    )[1].split("--quiet", maxsplit=1)[0]
    pilot_gate = text.split(
        'while ! "$PYTHON_BIN" tools/train/checkpoint_status.py \\\n'
        '  "$checkpoint"',
        maxsplit=1,
    )[1].split("--quiet", maxsplit=1)[0]
    if "reference_protocol_args=(" in text:
        reference_protocol = text.split(
            "reference_protocol_args=(",
            maxsplit=1,
        )[1].split(")", maxsplit=1)[0]
        assert '"${reference_protocol_args[@]}"' in reference_gate
        assert (
            "--expected-highpass-boundary periodic_lon_replicate_lat"
            in reference_protocol
        )
        assert "--expected-trainer-sha256" in reference_protocol
    else:
        assert '"${protocol_args[@]}"' in reference_gate
    assert '"${candidate_protocol_args[@]}"' not in reference_gate
    assert '"${candidate_protocol_args[@]}"' in pilot_gate


@pytest.mark.parametrize("script", PILOT_SCRIPTS)
def test_spherical_pilot_binds_exact_screen_checkpoint_boundary(
    script: Path,
) -> None:
    text = script.read_text()
    exact_checkpoint = (
        "$pilot_two_epoch_checkpoint"
        if "$pilot_two_epoch_checkpoint" in text
        else "$checkpoint"
    )
    exact_gate = text.split(
        f'"{exact_checkpoint}"',
        maxsplit=1,
    )[1].split("--quiet", maxsplit=1)[0]

    assert "--min-epochs 2" in exact_gate
    assert '"${candidate_protocol_args[@]}"' in exact_gate


@pytest.mark.parametrize("script", PILOT_SCRIPTS)
def test_spherical_pilot_has_checkpoint_bound_polar_rescue(
    script: Path,
) -> None:
    text = script.read_text()

    assert "epoch=1-step=3284.ckpt" in text
    assert "--test-year 2020" in text
    assert "--eval-days-per-month 8" in text
    assert "--eval-hours 2,4" in text
    assert "--polar-pilot-json" in text
    assert "--polar-reference-json" in text
    assert "--polar-pilot-checkpoint" in text
    assert "--polar-reference-checkpoint" in text
    assert "--test-year 2021" not in text


@pytest.mark.parametrize("script", IMPORTING_PILOT_SCRIPTS)
def test_pilot_exports_repo_root_before_python_imports(script: Path) -> None:
    text = script.read_text()

    export = 'export PYTHONPATH="$EXTRA_PYTHONPATH:$PWD:${PYTHONPATH:-}"'
    assert export in text
    assert text.index(export) < text.index("exec ")


def test_upr_spherical_reuses_validated_polar_artifacts_before_gpu() -> None:
    text = PILOT_SCRIPTS[2].read_text()

    assert "prepare_existing_assessment" in text
    assert "reuse validated two-epoch polar assessment" in text
    assert text.index("prepare_existing_assessment()") < text.index('gpu=""')
    assert text.index("if prepare_existing_assessment") < text.index('gpu=""')


def test_eval_queue_accepts_weather_amt_training_loss() -> None:
    text = EVAL_QUEUE.read_text()
    amt_gate = text.split(
        'if [[ "$amt_promoted" -eq 1 ]]',
        maxsplit=1,
    )[1].split("amt_models=(amt)", maxsplit=1)[0]

    assert "--expected-lambda-hf 0.05" in amt_gate


def test_endpoint_retrain_is_zero_shot_gated_and_planar() -> None:
    text = ENDPOINT_PILOT.read_text()

    assert "upr_endpoint_zero_shot_2020.json" in text
    assert '"promote_matched_retrain"' in text
    assert "reference_checkpoint_sha256" in text
    assert 'sha256sum "$reference_checkpoint"' in text
    assert "--expected-highpass-boundary periodic_lon_replicate_lat" in text
    assert "--candidate-name \"$arch\"" in text
    assert "--held-relative-limit 0.02" in text
    assert "--all-hour-relative-limit 0.02" in text
    assert "--per-hour-relative-limit 0.05" in text
    assert "--test-year 2021" not in text
