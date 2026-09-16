import pytest
import torch

from weather_time_interp.metrics.physical_consistency import (
    CANONICAL_24_CHANNELS,
    DIAGNOSTIC_COMPONENTS,
    _EARTH_RADIUS_M,
    _latitude,
    _spherical_wind_diagnostics,
    physical_diagnostics,
    validate_physical_channel_order,
)


def _state(batch: int = 2, height: int = 32, width: int = 64) -> torch.Tensor:
    torch.manual_seed(7)
    state = torch.randn(batch, 24, height, width)
    state[:, 0:4] = 270.0 + 5.0 * state[:, 0:4]
    state[:, 4:12] *= 10.0
    state[:, 12:16] = (
        0.005 + 0.001 * state[:, 12:16]
    ).clamp_min(1e-5)
    state[:, 16:20] = torch.tensor(
        [100.0, 800.0, 1500.0, 3000.0]
    ).view(1, 4, 1, 1) + 20.0 * state[:, 16:20]
    state[:, 20] = 285.0 + 3.0 * state[:, 20]
    state[:, 21:23] *= 5.0
    state[:, 23] = 101_000.0 + 500.0 * state[:, 23]
    return state


def test_identical_state_has_zero_consistency_error() -> None:
    state = _state()
    diagnostics = physical_diagnostics(state, state)
    assert set(diagnostics) == set(DIAGNOSTIC_COMPONENTS)
    for name, values in diagnostics.items():
        assert values.shape == (2, len(DIAGNOSTIC_COMPONENTS[name]))
        assert torch.isfinite(values).all()
        assert torch.count_nonzero(values) == 0


def test_wind_error_changes_dynamic_diagnostics() -> None:
    target = _state()
    prediction = target.clone()
    longitude = torch.linspace(0.0, 2.0 * torch.pi, target.size(-1))
    prediction[:, 4] += 3.0 * torch.sin(longitude).view(1, 1, -1)
    diagnostics = physical_diagnostics(prediction, target)
    assert diagnostics["wind_divergence_nmse"][:, 0].mean() > 0.0
    assert diagnostics["kinetic_energy_nmse"][:, 0].mean() > 0.0
    assert torch.count_nonzero(diagnostics["hydrostatic_balance_mse"]) == 0


def test_negative_humidity_is_reported_per_level() -> None:
    target = _state(batch=1)
    prediction = target.clone()
    prediction[:, 12, :, :8] = -1e-3
    diagnostics = physical_diagnostics(prediction, target)
    assert diagnostics["q_negative_fraction"][0, 0] > 0.0
    assert torch.count_nonzero(
        diagnostics["q_negative_fraction"][0, 1:]
    ) == 0


def test_global_mass_and_moisture_biases_detect_uniform_offsets() -> None:
    target = _state(batch=1)
    prediction = target.clone()
    prediction[:, 12:16] += 1e-3
    prediction[:, 23] += 100.0

    diagnostics = physical_diagnostics(prediction, target)

    assert diagnostics["global_mslp_bias"].shape == (1, 1)
    assert diagnostics["lower_tropospheric_moisture_bias"].shape == (1, 1)
    assert diagnostics["global_mslp_bias"].item() > 0.0
    assert diagnostics["lower_tropospheric_moisture_bias"].item() > 0.0


def test_spherical_vorticity_matches_solid_body_rotation() -> None:
    height, width = 180, 360
    latitude = _latitude(
        height,
        device=torch.device("cpu"),
        dtype=torch.float64,
    )
    omega = 7.2921159e-5
    u = (
        omega
        * _EARTH_RADIUS_M
        * torch.cos(torch.deg2rad(latitude))
    ).view(1, 1, height, 1).expand(-1, -1, -1, width)
    v = torch.zeros_like(u)

    divergence, vorticity = _spherical_wind_diagnostics(u, v, latitude)

    keep = latitude.abs() <= 70.0
    expected = (
        2.0 * omega * torch.sin(torch.deg2rad(latitude[keep]))
    ).view(1, 1, -1, 1).expand(-1, -1, -1, width)
    torch.testing.assert_close(
        divergence[:, :, keep],
        torch.zeros_like(divergence[:, :, keep]),
        rtol=0,
        atol=1e-12,
    )
    torch.testing.assert_close(
        vorticity[:, :, keep],
        expected,
        rtol=2e-4,
        atol=1e-10,
    )


def test_latitude_uses_equiangular_cell_centres() -> None:
    latitude = _latitude(
        180,
        device=torch.device("cpu"),
        dtype=torch.float64,
    )

    torch.testing.assert_close(
        latitude[0],
        torch.tensor(89.5, dtype=torch.float64),
    )
    torch.testing.assert_close(
        latitude[-1],
        torch.tensor(-89.5, dtype=torch.float64),
    )
    torch.testing.assert_close(
        latitude[:-1] - latitude[1:],
        torch.ones(179, dtype=torch.float64),
    )


def test_physical_channel_order_fails_closed() -> None:
    validate_physical_channel_order(list(CANONICAL_24_CHANNELS))
    swapped = list(CANONICAL_24_CHANNELS)
    swapped[0], swapped[1] = swapped[1], swapped[0]

    with pytest.raises(ValueError, match="canonical channel order"):
        validate_physical_channel_order(swapped)
