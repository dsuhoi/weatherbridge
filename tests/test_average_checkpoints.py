from pathlib import Path

import pytest
import torch

from tools.train.average_checkpoints import (
    average_checkpoints,
    discover_epoch_checkpoints,
)


def _write_checkpoint(
    path: Path,
    *,
    epoch: int,
    weight: float,
    counter: int = 7,
) -> None:
    torch.save(
        {
            "epoch": epoch,
            "global_step": (epoch + 1) * 10,
            "state_dict": {
                "net.weight": torch.tensor([weight], dtype=torch.float32),
                "net.counter": torch.tensor(counter, dtype=torch.int64),
            },
            "hyper_parameters": {
                "arch": "upr_lite_implicit_global",
                "total_steps": 80,
                "delta_t": 6.0,
            },
            "optimizer_states": [{"state": "not reusable"}],
            "lr_schedulers": [{"state": "not reusable"}],
        },
        path,
    )


def test_average_final_consecutive_epochs(tmp_path: Path) -> None:
    for epoch, weight in enumerate((1.0, 2.0, 4.0, 8.0)):
        _write_checkpoint(
            tmp_path / f"epoch={epoch}-step={(epoch + 1) * 10}.ckpt",
            epoch=epoch,
            weight=weight,
        )

    selected = discover_epoch_checkpoints(tmp_path, last_n=3)
    output = tmp_path / "avg_last3.ckpt"
    metadata = average_checkpoints(selected, output)
    averaged = torch.load(output, map_location="cpu", weights_only=False)

    assert metadata["epochs"] == [1, 2, 3]
    assert averaged["epoch"] == 3
    assert averaged["global_step"] == 40
    assert averaged["state_dict"]["net.weight"].item() == pytest.approx(
        14.0 / 3.0
    )
    assert averaged["state_dict"]["net.counter"].item() == 7
    assert averaged["optimizer_states"] == []
    assert averaged["lr_schedulers"] == []
    assert averaged["checkpoint_average"]["evaluation_only"] is True
    assert len(averaged["checkpoint_average"]["generator_sha256"]) == 64
    assert len(averaged["checkpoint_average"]["inputs"]) == 3
    assert output.stat().st_mode & 0o777 == 0o644
    assert discover_epoch_checkpoints(tmp_path, last_n=3) == selected


def test_rejects_nonconsecutive_final_epochs(tmp_path: Path) -> None:
    for epoch in (0, 2, 3):
        _write_checkpoint(
            tmp_path / f"epoch={epoch}.ckpt",
            epoch=epoch,
            weight=float(epoch),
        )

    with pytest.raises(ValueError, match="not consecutive"):
        discover_epoch_checkpoints(tmp_path, last_n=3)


def test_rejects_changed_nonfloating_state(tmp_path: Path) -> None:
    left = tmp_path / "epoch=0.ckpt"
    right = tmp_path / "epoch=1.ckpt"
    _write_checkpoint(left, epoch=0, weight=1.0, counter=7)
    _write_checkpoint(right, epoch=1, weight=2.0, counter=8)

    with pytest.raises(ValueError, match="non-floating tensors"):
        average_checkpoints([left, right], tmp_path / "invalid.ckpt")
