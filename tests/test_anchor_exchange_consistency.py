import numpy as np
import pytest
import torch

from tools.eval.eval_anchor_exchange_consistency import (
    _index_sha256,
    _parse_models,
    _summary,
    _weighted_mse,
    anchor_exchange_evaluation_source_paths,
)


def test_parse_models_requires_unique_complete_entries() -> None:
    assert _parse_models("upr:/a.ckpt,pp3:/b.ckpt") == [
        ("upr", "/a.ckpt"),
        ("pp3", "/b.ckpt"),
    ]
    with pytest.raises(ValueError, match="invalid"):
        _parse_models("upr")
    with pytest.raises(ValueError, match="unique"):
        _parse_models("upr:/a.ckpt,upr:/b.ckpt")


def test_weighted_mse_returns_per_sample_channel_values() -> None:
    left = torch.tensor(
        [
            [
                [[1.0, 1.0], [2.0, 2.0]],
                [[2.0, 2.0], [1.0, 1.0]],
            ]
        ]
    )
    right = torch.zeros_like(left)
    latitude_weight = torch.tensor([1.5, 0.5]).view(1, 1, 2, 1)

    result = _weighted_mse(left, right, latitude_weight)

    torch.testing.assert_close(result, torch.tensor([[1.75, 3.25]]))


def test_summary_distinguishes_pooled_and_channel_macro_rmse() -> None:
    values = np.asarray([[1.0, 9.0], [1.0, 9.0]])

    result = _summary(values)

    assert result["pooled_rmse"] == pytest.approx(np.sqrt(5.0))
    assert result["channel_macro_rmse"] == pytest.approx(2.0)
    assert result["per_channel_rmse"] == pytest.approx([1.0, 3.0])


def test_index_hash_binds_order_and_all_columns() -> None:
    year = np.asarray([2020, 2020])
    t0 = np.asarray([0, 24])
    tau = np.asarray([1, 2])

    baseline = _index_sha256(year, t0, tau)

    assert baseline != _index_sha256(year, t0 + 1, tau)
    assert baseline != _index_sha256(year[::-1], t0[::-1], tau[::-1])


def test_anchor_exchange_source_manifest_is_complete() -> None:
    sources = anchor_exchange_evaluation_source_paths()

    assert "tools/eval/capmatched_loader.py" in sources
    assert "weather_time_interp/model/weatherbridge_flow_model.py" in sources
    assert all(path.is_file() for path in sources.values())
    assert "tools/eval/eval_anchor_exchange_consistency.py" in sources
    assert "tools/eval/batch_eval_12h_memmap.py" in sources
