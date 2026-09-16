"""Physical diagnostics for the canonical 24-field interpolation protocol."""
from __future__ import annotations

import math

import torch

from weather_time_interp.grid import (
    latitude_strip_weights,
    wb2_block_average_latitudes,
)


DIAGNOSTIC_COMPONENTS = {
    "wind_divergence_nmse": ("1000", "925", "850", "700", "10m"),
    "wind_vorticity_nmse": ("1000", "925", "850", "700", "10m"),
    "kinetic_energy_nmse": ("1000", "925", "850", "700", "10m"),
    "hydrostatic_balance_mse": ("1000-925", "925-850", "850-700"),
    "q_negative_fraction": ("1000", "925", "850", "700"),
    "global_mslp_bias": ("global",),
    "lower_tropospheric_moisture_bias": ("1000-700",),
}
SELECTION_DIAGNOSTICS = (
    "wind_divergence_nmse",
    "wind_vorticity_nmse",
    "kinetic_energy_nmse",
    "hydrostatic_balance_mse",
)
GENERALIZATION_DIAGNOSTICS = SELECTION_DIAGNOSTICS + (
    "q_negative_fraction",
    "global_mslp_bias",
    "lower_tropospheric_moisture_bias",
)

_EARTH_RADIUS_M = 6_371_000.0
_DRY_AIR_GAS_CONSTANT = 287.05
_PRESSURE_HPA = (1000.0, 925.0, 850.0, 700.0)
CANONICAL_24_CHANNELS = tuple(
    f"{variable}{level}"
    for variable in ("T", "U", "V", "Q", "Z")
    for level in (1000, 925, 850, 700)
) + ("t2m", "u10", "v10", "mslp")


def validate_physical_channel_order(channel_names: list[str]) -> None:
    actual = tuple(channel_names[: len(CANONICAL_24_CHANNELS)])
    if actual != CANONICAL_24_CHANNELS:
        raise ValueError(
            "physical diagnostics require canonical channel order "
            f"{CANONICAL_24_CHANNELS}, got {actual}"
        )


def _latitude(
    height: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if height < 2:
        raise ValueError("physical diagnostics require at least two latitudes")
    if height == 360:
        return torch.as_tensor(
            wb2_block_average_latitudes(height),
            device=device,
            dtype=dtype,
        )
    half_cell_degrees = 90.0 / height
    return torch.linspace(
        90.0 - half_cell_degrees,
        -90.0 + half_cell_degrees,
        height,
        device=device,
        dtype=dtype,
    )


def _weighted_spatial_mean(
    values: torch.Tensor,
    latitude: torch.Tensor,
    *,
    latitude_limit: float | None = None,
) -> torch.Tensor:
    if latitude.numel() == 360:
        weights = torch.as_tensor(
            latitude_strip_weights(wb2_block_average_latitudes()),
            device=latitude.device,
            dtype=latitude.dtype,
        )
    else:
        weights = torch.cos(torch.deg2rad(latitude)).clamp_min(0.0)
    if latitude_limit is not None:
        weights = weights * (latitude.abs() <= latitude_limit)
    weights = weights / weights.sum().clamp_min(1e-12)
    return (values * weights.view(1, 1, -1, 1)).sum(dim=(-2, -1)) / values.size(-1)


def _horizontal_derivatives(
    field: torch.Tensor,
    latitude: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return eastward and northward centered derivatives on a regular grid."""
    height, width = field.shape[-2:]
    dlon = 2.0 * math.pi / width
    dlat = math.pi / height
    dx = (
        _EARTH_RADIUS_M
        * torch.cos(torch.deg2rad(latitude)).abs().clamp_min(1e-3)
        * dlon
    )
    east = (
        torch.roll(field, shifts=-1, dims=-1)
        - torch.roll(field, shifts=1, dims=-1)
    ) / (2.0 * dx.view(1, 1, -1, 1))
    # Latitude indices increase southward, hence north - south.
    north = (
        torch.roll(field, shifts=1, dims=-2)
        - torch.roll(field, shifts=-1, dims=-2)
    ) / (2.0 * _EARTH_RADIUS_M * dlat)
    return east, north


def _wind_fields(fields: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    u = torch.cat((fields[:, 4:8], fields[:, 21:22]), dim=1)
    v = torch.cat((fields[:, 8:12], fields[:, 22:23]), dim=1)
    return u, v


def _spherical_wind_diagnostics(
    u: torch.Tensor,
    v: torch.Tensor,
    latitude: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return horizontal divergence and relative vorticity on the sphere."""
    u_x, u_y = _horizontal_derivatives(u, latitude)
    v_x, v_y = _horizontal_derivatives(v, latitude)
    metric = (
        torch.tan(torch.deg2rad(latitude))
        / _EARTH_RADIUS_M
    ).view(1, 1, -1, 1)
    divergence = u_x + v_y - v * metric
    vorticity = v_x - u_y + u * metric
    return divergence, vorticity


def _normalized_mse(
    prediction: torch.Tensor,
    target: torch.Tensor,
    latitude: torch.Tensor,
    *,
    latitude_limit: float | None = None,
) -> torch.Tensor:
    numerator = _weighted_spatial_mean(
        (prediction - target).square(),
        latitude,
        latitude_limit=latitude_limit,
    )
    denominator = _weighted_spatial_mean(
        target.square(),
        latitude,
        latitude_limit=latitude_limit,
    )
    scale = denominator.detach().median().clamp_min(1e-20) * 1e-8
    return numerator / denominator.clamp_min(scale)


def _hydrostatic_residual(fields: torch.Tensor) -> torch.Tensor:
    temperature = fields[:, 0:4]
    humidity = fields[:, 12:16]
    geopotential = fields[:, 16:20]
    virtual_temperature = temperature * (1.0 + 0.61 * humidity)
    residuals = []
    for lower in range(3):
        upper = lower + 1
        thickness = geopotential[:, upper] - geopotential[:, lower]
        expected = (
            _DRY_AIR_GAS_CONSTANT
            * 0.5
            * (virtual_temperature[:, lower] + virtual_temperature[:, upper])
            * math.log(_PRESSURE_HPA[lower] / _PRESSURE_HPA[upper])
        )
        residuals.append((thickness - expected) / expected.abs().clamp_min(1.0))
    return torch.stack(residuals, dim=1)


def _global_relative_bias(
    prediction: torch.Tensor,
    target: torch.Tensor,
    latitude: torch.Tensor,
) -> torch.Tensor:
    prediction_mean = _weighted_spatial_mean(prediction, latitude)
    target_mean = _weighted_spatial_mean(target, latitude)
    return (
        (prediction_mean - target_mean).abs()
        / target_mean.abs().clamp_min(1e-12)
    )


def _lower_tropospheric_moisture(fields: torch.Tensor) -> torch.Tensor:
    humidity = fields[:, 12:16]
    delta_pressure = humidity.new_tensor((75.0, 75.0, 150.0)).view(
        1,
        3,
        1,
        1,
    )
    return (
        0.5
        * (humidity[:, :-1] + humidity[:, 1:])
        * delta_pressure
    ).sum(dim=1, keepdim=True)


def physical_diagnostics(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Return per-sample, per-component dimensionless physical diagnostics.

    Inputs are physical-unit canonical fields in the order
    ``T4, U4, V4, Q4, Z4, t2m, u10, v10, mslp``.
    """
    if prediction.shape != target.shape:
        raise ValueError("prediction and target must have identical shapes")
    if prediction.dim() != 4 or prediction.size(1) < 24:
        raise ValueError("physical diagnostics require (B, >=24, H, W)")
    prediction = prediction[:, :24].float()
    target = target[:, :24].float()
    latitude = _latitude(
        prediction.size(-2),
        device=prediction.device,
        dtype=prediction.dtype,
    )

    pred_u, pred_v = _wind_fields(prediction)
    tgt_u, tgt_v = _wind_fields(target)
    pred_divergence, pred_vorticity = _spherical_wind_diagnostics(
        pred_u,
        pred_v,
        latitude,
    )
    tgt_divergence, tgt_vorticity = _spherical_wind_diagnostics(
        tgt_u,
        tgt_v,
        latitude,
    )
    pred_ke = 0.5 * (pred_u.square() + pred_v.square())
    tgt_ke = 0.5 * (tgt_u.square() + tgt_v.square())

    pred_hydro = _hydrostatic_residual(prediction)
    tgt_hydro = _hydrostatic_residual(target)
    pred_moisture = _lower_tropospheric_moisture(prediction)
    tgt_moisture = _lower_tropospheric_moisture(target)
    return {
        "wind_divergence_nmse": _normalized_mse(
            pred_divergence,
            tgt_divergence,
            latitude,
            latitude_limit=75.0,
        ),
        "wind_vorticity_nmse": _normalized_mse(
            pred_vorticity,
            tgt_vorticity,
            latitude,
            latitude_limit=75.0,
        ),
        "kinetic_energy_nmse": _normalized_mse(pred_ke, tgt_ke, latitude),
        "hydrostatic_balance_mse": _weighted_spatial_mean(
            (pred_hydro - tgt_hydro).square(),
            latitude,
        ),
        "q_negative_fraction": _weighted_spatial_mean(
            (prediction[:, 12:16] < 0.0).float(),
            latitude,
        ),
        "global_mslp_bias": _global_relative_bias(
            prediction[:, 23:24],
            target[:, 23:24],
            latitude,
        ),
        "lower_tropospheric_moisture_bias": _global_relative_bias(
            pred_moisture,
            tgt_moisture,
            latitude,
        ),
    }
