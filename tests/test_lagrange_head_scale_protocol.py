from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_scale_screen_preserves_honest_temporal_and_year_gates() -> None:
    source = (
        REPO_ROOT
        / "tools/eval/run_lagrange_head_scale_screen_cloudru.sh"
    ).read_text()

    assert "SOURCE_SHA256=" in source
    assert "scale_lagrange_head_checkpoint.py" in source
    assert "for factor in 2 4 6 8" in source
    assert '--factor "$factor"' in source
    for factor in (2, 4, 6, 8):
        assert f"factor{factor}.ckpt" in source
    assert "--eval-hours 1,2,3,4,5" in source
    assert "--seen-tau 1,3,5" in source
    assert "--unseen-tau 2,4" in source
    assert "--eval-days-per-month 8" in source
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
