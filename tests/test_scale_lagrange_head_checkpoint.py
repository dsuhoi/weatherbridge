from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
import torch

from tools.train.scale_lagrange_head_checkpoint import scale_checkpoint


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_scales_only_lagrange_head_and_records_source(tmp_path) -> None:
    source = tmp_path / "source.ckpt"
    output = tmp_path / "scaled.ckpt"
    torch.save(
        {
            "hyper_parameters": {
                "arch": "flow_compact_lagrange_l",
                "trainable_scope": "base_knot_head",
            },
            "state_dict": {
                "net.base_knot_head.weight": torch.ones(2, 2),
                "net.base_knot_head.bias": torch.ones(2),
                "net.encoder.weight": torch.full((2, 2), 3.0),
            },
            "optimizer_states": [{"state": {}}],
            "lr_schedulers": [{"last_epoch": 1}],
            "callbacks": {"checkpoint": {}},
        },
        source,
    )

    scale_checkpoint(source, output, factor=4.0)
    checkpoint = torch.load(output, map_location="cpu", weights_only=False)

    assert torch.equal(
        checkpoint["state_dict"]["net.base_knot_head.weight"],
        torch.full((2, 2), 4.0),
    )
    assert torch.equal(
        checkpoint["state_dict"]["net.base_knot_head.bias"],
        torch.full((2,), 4.0),
    )
    assert torch.equal(
        checkpoint["state_dict"]["net.encoder.weight"],
        torch.full((2, 2), 3.0),
    )
    transform = checkpoint["hyper_parameters"]["deployment_transform"]
    assert transform["factor"] == 4.0
    assert transform["source_checkpoint_sha256"] == _sha256(source)
    assert checkpoint["optimizer_states"] == []
    assert checkpoint["lr_schedulers"] == []
    assert checkpoint["callbacks"] == {}


@pytest.mark.parametrize("factor", (0.0, -1.0, float("inf")))
def test_rejects_invalid_scale(tmp_path, factor: float) -> None:
    source = tmp_path / "source.ckpt"
    source.write_bytes(b"unused")

    with pytest.raises(ValueError, match="finite and positive"):
        scale_checkpoint(source, tmp_path / "output.ckpt", factor=factor)
