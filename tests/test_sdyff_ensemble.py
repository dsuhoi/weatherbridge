import pytest

def test_prefix_means_average_predictions_not_errors():
    torch = pytest.importorskip("torch")
    from tools.eval.eval_sdyff_ensemble import prefix_means

    draws = iter([torch.tensor([0.0]), torch.tensor([4.0]), torch.tensor([2.0])])
    means = prefix_means(lambda: next(draws), (1, 2, 3))
    assert means[1].item() == 0.0
    assert means[2].item() == means[3].item() == 2.0
    assert (means[2] - 2).square().item() == 0.0
    assert means[1].item() == 0.0  # Earlier prefixes must not alias the sum.


def test_prefix_means_reject_invalid_sizes_and_nonfinite_draws():
    torch = pytest.importorskip("torch")
    from tools.eval.eval_sdyff_ensemble import prefix_means

    for sizes in [(), (0,), (2, 1), (1, 1)]:
        with pytest.raises(ValueError):
            prefix_means(lambda: torch.ones(1), sizes)
    with pytest.raises(ValueError, match="non-finite"):
        prefix_means(lambda: torch.tensor([float("nan")]), (1,))


def test_summary_pairs_prefixes_and_checks_reference(tmp_path):
    import json
    import numpy as np
    from tools.eval.paired_block_bootstrap import sha256_file, window_index_sha256
    from tools.eval.summarize_sdyff_ensemble import summarize

    window = tmp_path / "windows.npz"
    years = np.full(4, 2020, dtype=np.int16)
    starts = np.array([0, 0, 24 * 14, 24 * 14], dtype=np.int32)
    taus = np.array([1, 2, 1, 2], dtype=np.int8)
    np.savez(window, year=years, t0=starts, tau=taus, channel_names=np.array(["T850"]),
             mse_norm_ens1=np.full((4, 1), 4.0), mse_norm_model=np.full((4, 1), 1.0))
    def scores(rmse):
        return {"rmse_norm_T850": rmse, "rmse_phys_T850": rmse, "acc_T850": 0.8}
    payload = {
        "channel_names": ["T850"], "years": [2020], "n_per_tau": {"1": 2, "2": 2},
        "seen_tau": [1], "unseen_tau": [2], "delta_t_hours": 3,
        "checkpoint_provenance": {"sha256": "same-checkpoint"},
        "window_metrics_file": "windows.npz",
        "window_metrics_provenance": {"sha256": sha256_file(window)},
        "evaluation_protocol": {"index_sha256": window_index_sha256(years, starts, taus)},
        "ensemble_protocol": {"method_sizes": {"ens1": 1, "model": 2}},
        "per_tau": {str(h): {"model": scores(1.0), "ens1": scores(2.0),
                             "bilinear": scores(3.0)} for h in (1, 2)},
    }
    source = tmp_path / "sdyff_ens.json"
    reference = tmp_path / "reference.json"
    source.write_text(json.dumps(payload))
    reference.write_text(json.dumps(payload))
    rows, summary = summarize(source, reference, draws=20)
    assert len(rows) == 4
    assert rows[-1]["gain_vs_single_pct"] == 50.0
    assert summary["paired_vs_single"]["2"]["all"]["relative_delta_pct"] == -50.0
    payload["evaluation_protocol"]["index_sha256"] = "different-window-index"
    reference.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="window index"):
        summarize(source, reference, draws=20)
    payload["evaluation_protocol"]["index_sha256"] = window_index_sha256(years, starts, taus)
    payload["checkpoint_provenance"]["sha256"] = "another-checkpoint"
    reference.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="checkpoint"):
        summarize(source, reference, draws=20)
