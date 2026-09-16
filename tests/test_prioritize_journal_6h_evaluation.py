from pathlib import Path


SCRIPT = Path("scripts/prioritize_journal_6h_evaluation_cloudru.sh")


def test_priority_wrapper_never_interrupts_active_jobs() -> None:
    source = SCRIPT.read_text()
    assert "kill " not in source
    assert "dcae_6_202709 flow_spectral_12_202707" in source
    assert 'until [[ -e "$STATE/$id.done" ]]' in source


def test_priority_wrapper_releases_only_its_own_claims() -> None:
    source = SCRIPT.read_text()
    assert '"$(cat "$marker")" == "$TOKEN"' in source
    assert 'flock 7' in source
    assert 'metrics/journal_spectra_v6/state/.complete_6h' in source
    assert 'metrics/detailed_benchmark_v2/6h/.complete' in source
    assert "run_npj_seed_replicates_cloudru.sh" in source
