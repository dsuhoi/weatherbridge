import pytest

torch = pytest.importorskip("torch")

from tools.eval.validate_dcae_skip_pair import validate_pair


def _checkpoint(arch, state):
    protocol = {
        "requested_arch": arch,
        "canonical_arch": (
            "WeatherDCAE-Skip" if arch == "wb_skip" else "WeatherDCAE"
        ),
        "optimizer_steps_per_epoch": 820,
        "seed": 202707,
        "train_years": [2017, 2018, 2019],
        "train_tau_hours": [1, 3, 5],
        "batch_size_per_device": 4,
    }
    return {
        "epoch": 7,
        "global_step": 6560,
        "state_dict": state,
        "hyper_parameters": {
            "arch": arch,
            "training_protocol": protocol,
            "training_input_provenance": {"data": "same"},
            "training_code_provenance": {"code": "same"},
        },
    }


def _write_pair(tmp_path):
    base_state = {"net.weight": torch.zeros(2, 3)}
    skip_state = {
        "net.weight": torch.zeros(2, 3),
        "net.skip_gates.0": torch.zeros(()),
        "net.skip_gates.1": torch.zeros(()),
    }
    base_path = tmp_path / "base.ckpt"
    skip_path = tmp_path / "skip.ckpt"
    torch.save(_checkpoint("dcae_14m", base_state), base_path)
    torch.save(_checkpoint("wb_skip", skip_state), skip_path)
    return base_path, skip_path


def test_validate_dcae_skip_pair_accepts_only_two_scalar_gates(tmp_path):
    base_path, skip_path = _write_pair(tmp_path)

    result = validate_pair(base_path, skip_path)

    assert result["matched"] is True
    assert result["optimizer_updates"] == 6560
    assert result["skip_parameter_tensors_numel"] == 8
    assert result["base_parameter_tensors_numel"] == 6


def test_validate_dcae_skip_pair_rejects_update_mismatch(tmp_path):
    base_path, skip_path = _write_pair(tmp_path)
    checkpoint = torch.load(skip_path, weights_only=False)
    checkpoint["global_step"] = 6561
    torch.save(checkpoint, skip_path)

    with pytest.raises(ValueError, match="6560 updates"):
        validate_pair(base_path, skip_path)
