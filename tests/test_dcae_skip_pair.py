import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("pytorch_lightning")

from tools.train.train_capacity_matched_6h import build_net
from tools.eval.capmatched_loader import _build_net_for_checkpoint


def test_weatherdcae_skip_changes_only_skip_gates():
    noskip, _, _ = build_net("dcae_14m", "unused.pt")
    skip, _, _ = build_net("wb_skip", "unused.pt")

    for attr in (
        "latent_channels",
        "block_out_channels",
        "layers_per_block",
        "block_type",
        "qkv_multiscales",
        "attention_head_dim",
        "in_channels",
        "out_channels",
        "n_static_features",
        "lat_crop",
    ):
        assert getattr(skip, attr) == getattr(noskip, attr)

    noskip_params = sum(parameter.numel() for parameter in noskip.parameters())
    skip_params = sum(parameter.numel() for parameter in skip.parameters())
    assert skip_params - noskip_params == 2


def test_legacy_skip_checkpoint_shape_selects_matched_13m_backbone():
    legacy = _build_net_for_checkpoint(
        "wb_skip",
        {"encoder.conv_in.weight": torch.empty(128, 51, 1, 1)},
        "unused.pt",
    )

    assert legacy.block_out_channels == (128, 128, 256, 256)
    assert legacy.layers_per_block == (2, 2, 2)
    assert sum(p.numel() for p in legacy.parameters()) == 13_389_532
