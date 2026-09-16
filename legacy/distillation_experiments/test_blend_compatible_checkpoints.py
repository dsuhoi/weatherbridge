from pathlib import Path

import pytest
import torch

from tools.train.blend_compatible_checkpoints import blend_checkpoints


def _write(
    path: Path,
    *,
    epoch: int,
    weight: float,
    distill_weight: float,
) -> None:
    torch.save(
        {
            "epoch": epoch,
            "global_step": (epoch + 1) * 10,
            "state_dict": {
                "net.weight": torch.tensor([weight]),
                "net.counter": torch.tensor(7),
            },
            "hyper_parameters": {
                "arch": "flow_pp3_compact_l",
                "delta_t": 6.0,
                "total_steps": 40,
                "training_seed": 202707,
                "distill_weight": distill_weight,
            },
            "optimizer_states": [{"state": {}}],
            "lr_schedulers": [{"last_epoch": epoch}],
        },
        path,
    )


def test_blends_same_epoch_runs_and_records_provenance(tmp_path) -> None:
    left = tmp_path / "plain.ckpt"
    right = tmp_path / "distill.ckpt"
    output = tmp_path / "blend.ckpt"
    _write(left, epoch=3, weight=2.0, distill_weight=0.0)
    _write(right, epoch=3, weight=6.0, distill_weight=0.2)

    metadata = blend_checkpoints(
        left,
        right,
        output,
        left_weight=0.25,
    )
    checkpoint = torch.load(output, map_location="cpu", weights_only=False)

    assert checkpoint["state_dict"]["net.weight"].item() == pytest.approx(5.0)
    assert checkpoint["state_dict"]["net.counter"].item() == 7
    assert checkpoint["optimizer_states"] == []
    assert checkpoint["lr_schedulers"] == []
    assert metadata["left_weight"] == 0.25
    assert metadata["epoch"] == 3
    assert len(metadata["inputs"]) == 2
    assert checkpoint["hyper_parameters"]["deployment_transform"] == metadata


@pytest.mark.parametrize("left_weight", (0.0, 1.0, -0.1, float("inf")))
def test_rejects_invalid_weight(tmp_path, left_weight: float) -> None:
    with pytest.raises(ValueError, match="strictly between"):
        blend_checkpoints(
            tmp_path / "left.ckpt",
            tmp_path / "right.ckpt",
            tmp_path / "output.ckpt",
            left_weight=left_weight,
        )


def test_rejects_epoch_mismatch(tmp_path) -> None:
    left = tmp_path / "left.ckpt"
    right = tmp_path / "right.ckpt"
    _write(left, epoch=1, weight=1.0, distill_weight=0.0)
    _write(right, epoch=3, weight=2.0, distill_weight=0.2)

    with pytest.raises(ValueError, match="same epoch"):
        blend_checkpoints(
            left,
            right,
            tmp_path / "output.ckpt",
            left_weight=0.5,
        )
