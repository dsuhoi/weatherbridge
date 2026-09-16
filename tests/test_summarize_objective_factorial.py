from __future__ import annotations

import numpy as np
import pytest

import tools.eval.summarize_objective_factorial as factorial
from tools.eval.paired_block_bootstrap import WindowMetrics
from tools.eval.summarize_objective_factorial import (
    _validate_run_contract,
    factorial_block_bootstrap,
)


def _metrics(rmse: float) -> WindowMetrics:
    year = np.full(16, 2020, dtype=np.int16)
    t0 = np.repeat(np.arange(0, 8 * 24, 24, dtype=np.int32), 2)
    tau = np.tile(np.asarray([2, 4], dtype=np.int8), 8)
    mse = np.full((16, 3), rmse**2, dtype=np.float64)
    return WindowMetrics(year, t0, tau, mse, ("t2m", "z500", "q850"))


def test_factorial_recovers_main_effects_without_interaction() -> None:
    result = factorial_block_bootstrap(
        {
            "upr_hf": _metrics(1.0),
            "upr_nohf": _metrics(2.0),
            "flow_hf": _metrics(4.0),
            "flow_nohf": _metrics(8.0),
        },
        taus=np.asarray([2, 4]),
        block_days=1,
        draws=200,
        seed=7,
    )

    assert result["cell_rmse"] == {
        "upr_hf": 1.0,
        "upr_nohf": 2.0,
        "flow_hf": 4.0,
        "flow_nohf": 8.0,
    }
    assert result["effects"]["architecture"]["relative_effect_pct"] == (
        pytest.approx(-75.0)
    )
    assert result["effects"]["highpass"]["relative_effect_pct"] == (
        pytest.approx(-50.0)
    )
    assert result["effects"]["interaction"]["relative_effect_pct"] == (
        pytest.approx(0.0)
    )
    assert result["effects"]["architecture"]["log_effect_ci95"][2] < 0.0
    assert set(result["per_tau"]) == {"2", "4"}
    assert result["worst_architecture_field"]["channel"] in {
        "t2m",
        "z500",
        "q850",
    }


def test_factorial_rejects_unpaired_windows() -> None:
    cells = {
        "upr_hf": _metrics(1.0),
        "upr_nohf": _metrics(2.0),
        "flow_hf": _metrics(4.0),
        "flow_nohf": _metrics(8.0),
    }
    mismatched = cells["flow_nohf"]
    cells["flow_nohf"] = WindowMetrics(
        mismatched.year,
        mismatched.t0 + 1,
        mismatched.tau,
        mismatched.mse,
        mismatched.channels,
    )

    with pytest.raises(ValueError, match="window index mismatch"):
        factorial_block_bootstrap(
            cells,
            taus=np.asarray([2, 4]),
            block_days=1,
            draws=20,
            seed=7,
        )


def test_factorial_contract_requires_pre_ood_selection_and_fixed_taus() -> None:
    selection = {
        "schema_version": 12,
        "ood_attached_at_selection_time": False,
    }
    groups = {
        "all": np.asarray([1, 2, 3, 4, 5]),
        "seen": np.asarray([1, 3, 5]),
        "unseen": np.asarray([2, 4]),
    }

    assert _validate_run_contract(selection, groups) == 12

    leaked = dict(selection, ood_attached_at_selection_time=True)
    with pytest.raises(ValueError, match="frozen selection"):
        _validate_run_contract(leaked, groups)

    wrong_groups = dict(groups, unseen=np.asarray([2]))
    with pytest.raises(ValueError, match="tau sets"):
        _validate_run_contract(selection, wrong_groups)


def test_load_year_binds_frozen_checkpoint_and_window_hashes(
    tmp_path,
    monkeypatch,
) -> None:
    root = tmp_path / "2020"
    root.mkdir()
    selection_models = {}
    for cell, model in factorial.DEFAULT_MODELS.items():
        checkpoint = f"{model}-checkpoint"
        selection_models[model] = {"checkpoint_sha256": checkpoint}
        window_path = root / f"{model}.npz"
        metrics = _metrics(
            {
                "upr_hf": 1.0,
                "upr_nohf": 2.0,
                "flow_hf": 4.0,
                "flow_nohf": 8.0,
            }[cell]
        )
        np.savez(
            window_path,
            year=metrics.year,
            t0=metrics.t0,
            tau=metrics.tau,
            mse_norm_model=metrics.mse,
            channel_names=np.asarray(metrics.channels),
        )
        (root / f"{model}.json").write_text(
            f'{{"window_metrics_file": "{window_path.name}"}}'
        )

    def fake_summary(path):
        model = path.stem
        return {
            "evaluation_full_year": True,
            "evaluation_dataset_provenance": {
                "years": [2020],
                "identity": "dataset",
            },
            "evaluation_input_provenance": {"identity": "inputs"},
            "checkpoint_sha256": f"{model}-checkpoint",
        }

    monkeypatch.setattr(factorial, "load_field_metrics", fake_summary)
    cells, artifacts = factorial._load_year(
        selection={"models": selection_models},
        root=root,
        year=2020,
        models=factorial.DEFAULT_MODELS,
    )

    assert set(cells) == set(factorial.CELL_ORDER)
    assert len(
        {
            artifact["window_index_sha256"]
            for artifact in artifacts.values()
        }
    ) == 1
    assert all(
        len(artifact["window_metrics_sha256"]) == 64
        for artifact in artifacts.values()
    )
