"""Document the single-draw inference policy used by the published S-DYff row."""

import pytest

torch = pytest.importorskip('torch')
pytest.importorskip('torch_harmonics')

from weather_time_interp.model.sdyff_baseline_model import SDyffBlock


def test_eval_keeps_stochastic_depth_but_disables_dropout():
    block = SDyffBlock(
        embed_dim=4, time_dim=4, nlat=8, nlon=16,
        n_modes_lat=2, n_modes_lon=3, dropout=0.1, drop_path=0.1,
    ).eval()
    x = torch.ones(128, 4, 2, 2)
    assert torch.equal(block.dropout(x), x)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(20260911)
        first = block._drop_path(x)
        second = block._drop_path(x)
        assert not torch.equal(first, second)
        assert (first == 0).any() and (first > 1).any()
        torch.manual_seed(20260911)
        assert torch.equal(first, block._drop_path(x))
