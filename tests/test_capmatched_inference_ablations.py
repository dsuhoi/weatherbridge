from __future__ import annotations

import torch.nn as nn

from tools.eval.capmatched_loader import _configure_inference_ablations


class _WeatherBridgeStub(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.ablate_acceleration = False
        self.ablate_transport = False
        self.ablate_hydrostatic = False


def test_inference_ablation_environment_is_reset_per_model(monkeypatch) -> None:
    model = _WeatherBridgeStub()
    monkeypatch.setenv("WEATHERBRIDGE_ABLATE_ACCELERATION", "1")
    monkeypatch.setenv("WEATHERBRIDGE_ABLATE_TRANSPORT", "0")
    monkeypatch.setenv("WEATHERBRIDGE_ABLATE_HYDROSTATIC", "0")
    _configure_inference_ablations(model)
    assert model.ablate_acceleration
    assert not model.ablate_transport
    assert not model.ablate_hydrostatic

    monkeypatch.setenv("WEATHERBRIDGE_ABLATE_ACCELERATION", "0")
    monkeypatch.setenv("WEATHERBRIDGE_ABLATE_TRANSPORT", "1")
    monkeypatch.setenv("WEATHERBRIDGE_ABLATE_HYDROSTATIC", "1")
    _configure_inference_ablations(model)
    assert not model.ablate_acceleration
    assert model.ablate_transport
    assert model.ablate_hydrostatic
