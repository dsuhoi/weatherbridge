from pathlib import Path


def _source() -> str:
    return (
        Path(__file__).parents[1]
        / "tools"
        / "train"
        / "run_weatherdcae_sparse_distillation_6h_cloudru.sh"
    ).read_text()


def test_weatherdcae_sparse_distillation_is_matched_and_validation_only() -> None:
    source = _source()

    assert "--arch dcae_14m" in source
    assert "train_one control 0 &" in source
    assert "train_one student 1 &" in source
    assert "--seed 202712" in source
    assert "--train_batches_per_epoch 1640" in source
    assert "--test-year 2020" in source
    assert "--test-year 2021" not in source
    assert "select_distilled_champion.py" in source
    assert 'touch "$STATE/strict.pass"' in source
    assert 'touch "$STATE/precheck.pass"' in source


def test_weatherdcae_sparse_distillation_skips_teacher_forwards() -> None:
    source = _source()

    assert "--distill_direct_blend 0.02" in source
    assert "--distill_every_n_steps 4" in source
    assert 'assert distill["every_n_steps"] == 4' in source
    assert 'assert distill["interval_corrected"] is True' in source
    assert 'assert route["active_routes"] == 95' in source
    assert "merge_a050.ckpt" in source


def test_weatherdcae_sparse_distillation_reuses_local_climatology() -> None:
    source = _source()

    assert "CLIM_CACHE=${CLIM_CACHE:-/tmp/wti_climatology_cache_detailed_6h}" in source
    assert 'ln -s "$CLIM_CACHE" "$OUT_ROOT/climatology_cache"' in source
