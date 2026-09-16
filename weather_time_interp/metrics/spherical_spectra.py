"""Spherical spectral operators for cell-centred latitude-longitude grids.

ERA5-derived fields are sampled at latitude cell centres, not at the endpoints
used by the ``torch_harmonics`` equiangular transform. This module uses Fejer's
first quadrature rule on its native cell-centred grid, or explicit spherical
strip-area quadrature when the source-grid latitudes are supplied.
"""

from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn

from weather_time_interp.grid import latitude_strip_weights


def fejer1_nodes_weights(nlat: int) -> tuple[np.ndarray, np.ndarray]:
    """Return north-to-south colatitudes and Fejer-I integration weights."""
    if nlat < 2:
        raise ValueError("nlat must be at least 2")
    theta = (np.arange(nlat, dtype=np.float64) + 0.5) * np.pi / nlat
    k = np.arange(1, nlat // 2 + 1, dtype=np.float64)
    series = np.cos(2.0 * np.outer(theta, k)) / (4.0 * np.square(k) - 1.0)
    weights = (2.0 / nlat) * (1.0 - 2.0 * series.sum(axis=1))
    if not np.isclose(weights.sum(), 2.0, rtol=0.0, atol=5.0e-14):
        raise RuntimeError("Fejer-I weights do not integrate a constant exactly")
    return theta, weights


class CellCenteredRealSHT(nn.Module):
    """Forward real SHT on an equiangular cell-centred latitude grid.

    ``lmax`` and ``mmax`` are exclusive bounds, matching ``torch_harmonics``.
    """

    def __init__(
        self,
        nlat: int,
        nlon: int,
        lmax: int | None = None,
        mmax: int | None = None,
        *,
        norm: str = "ortho",
        csphase: bool = True,
        latitude_degrees: np.ndarray | None = None,
    ) -> None:
        super().__init__()
        from torch_harmonics.legendre import _precompute_legpoly

        self.nlat = int(nlat)
        self.nlon = int(nlon)
        self.lmax = int(lmax or nlat)
        self.mmax = int(mmax or nlon // 2 + 1)
        self.grid = (
            "equiangular-cell-centered-fejer1"
            if latitude_degrees is None
            else "explicit-latitude-strip-area"
        )
        self.norm = norm
        self.csphase = bool(csphase)
        if not 1 <= self.lmax <= self.nlat:
            raise ValueError("lmax must lie in [1, nlat]")
        if not 1 <= self.mmax <= self.nlon // 2 + 1:
            raise ValueError("mmax exceeds the real-FFT longitudinal bound")

        if latitude_degrees is None:
            theta, quadrature_weights = fejer1_nodes_weights(self.nlat)
        else:
            latitude = np.asarray(latitude_degrees, dtype=np.float64)
            if latitude.shape != (self.nlat,):
                raise ValueError("latitude_degrees has the wrong shape")
            theta = np.deg2rad(90.0 - latitude)
            quadrature_weights = latitude_strip_weights(latitude)
        theta_tensor = torch.as_tensor(theta, dtype=torch.float64)
        legendre = torch.as_tensor(
            _precompute_legpoly(
                self.mmax,
                self.lmax,
                theta_tensor,
                norm=self.norm,
                csphase=self.csphase,
            ),
            dtype=torch.float64,
        )
        weights = torch.from_numpy(quadrature_weights)
        weights = torch.einsum("mlk,k->mlk", legendre, weights)
        self.register_buffer("weights", weights, persistent=False)

    def forward(self, field: torch.Tensor) -> torch.Tensor:
        if field.ndim < 2:
            raise ValueError("SHT input must have at least two dimensions")
        if field.shape[-2:] != (self.nlat, self.nlon):
            raise ValueError(
                f"expected (...,{self.nlat},{self.nlon}), got {tuple(field.shape)}"
            )
        fourier = 2.0 * torch.pi * torch.fft.rfft(
            field,
            dim=-1,
            norm="forward",
        )
        split = torch.view_as_real(fourier)
        out_shape = [*split.shape]
        out_shape[-3] = self.lmax
        out_shape[-2] = self.mmax
        output = torch.zeros(out_shape, dtype=split.dtype, device=split.device)
        weights = self.weights.to(dtype=split.dtype)
        output[..., 0] = torch.einsum(
            "...km,mlk->...lm",
            split[..., : self.mmax, 0],
            weights,
        )
        output[..., 1] = torch.einsum(
            "...km,mlk->...lm",
            split[..., : self.mmax, 1],
            weights,
        )
        return torch.view_as_complex(output)


class CellCenteredRealVectorSHT(nn.Module):
    """Forward vector SHT for ``(..., 2, nlat, nlon)`` zonal/meridional winds."""

    def __init__(
        self,
        nlat: int,
        nlon: int,
        lmax: int | None = None,
        mmax: int | None = None,
        *,
        norm: str = "ortho",
        csphase: bool = True,
        latitude_degrees: np.ndarray | None = None,
    ) -> None:
        super().__init__()
        from torch_harmonics.legendre import _precompute_dlegpoly

        self.nlat = int(nlat)
        self.nlon = int(nlon)
        self.lmax = int(lmax or nlat)
        self.mmax = int(mmax or nlon // 2 + 1)
        self.grid = (
            "equiangular-cell-centered-fejer1-vector"
            if latitude_degrees is None
            else "explicit-latitude-strip-area-vector"
        )
        self.norm = norm
        self.csphase = bool(csphase)
        if not 1 <= self.lmax <= self.nlat:
            raise ValueError("lmax must lie in [1, nlat]")
        if not 1 <= self.mmax <= self.nlon // 2 + 1:
            raise ValueError("mmax exceeds the real-FFT longitudinal bound")

        if latitude_degrees is None:
            theta, quadrature_weights = fejer1_nodes_weights(self.nlat)
        else:
            latitude = np.asarray(latitude_degrees, dtype=np.float64)
            if latitude.shape != (self.nlat,):
                raise ValueError("latitude_degrees has the wrong shape")
            theta = np.deg2rad(90.0 - latitude)
            quadrature_weights = latitude_strip_weights(latitude)
        theta_tensor = torch.as_tensor(theta, dtype=torch.float64)
        dlegendre = torch.as_tensor(
            _precompute_dlegpoly(
                self.mmax,
                self.lmax,
                theta_tensor,
                norm=self.norm,
                csphase=self.csphase,
            ),
            dtype=torch.float64,
        )
        ell = torch.arange(self.lmax, dtype=torch.float64)
        norm_factor = torch.ones_like(ell)
        norm_factor[1:] = 1.0 / (ell[1:] * (ell[1:] + 1.0))
        weights = torch.einsum(
            "dmlk,k,l->dmlk",
            dlegendre,
            torch.from_numpy(quadrature_weights),
            norm_factor,
        )
        weights[1] *= -1.0
        self.register_buffer("weights", weights, persistent=False)

    def forward(self, field: torch.Tensor) -> torch.Tensor:
        if field.ndim < 3 or field.shape[-3] != 2:
            raise ValueError("vector SHT input must have shape (...,2,H,W)")
        if field.shape[-2:] != (self.nlat, self.nlon):
            raise ValueError(
                f"expected (...,2,{self.nlat},{self.nlon}), got {tuple(field.shape)}"
            )
        fourier = 2.0 * torch.pi * torch.fft.rfft(
            field,
            dim=-1,
            norm="forward",
        )
        split = torch.view_as_real(fourier)
        out_shape = [*split.shape]
        out_shape[-3] = self.lmax
        out_shape[-2] = self.mmax
        output = torch.zeros(out_shape, dtype=split.dtype, device=split.device)
        weights = self.weights.to(dtype=split.dtype)
        u = split[..., 0, :, : self.mmax, :]
        v = split[..., 1, :, : self.mmax, :]
        output[..., 0, :, :, 0] = torch.einsum(
            "...km,mlk->...lm", u[..., 0], weights[0]
        ) - torch.einsum("...km,mlk->...lm", v[..., 1], weights[1])
        output[..., 0, :, :, 1] = torch.einsum(
            "...km,mlk->...lm", u[..., 1], weights[0]
        ) + torch.einsum("...km,mlk->...lm", v[..., 0], weights[1])
        output[..., 1, :, :, 0] = -torch.einsum(
            "...km,mlk->...lm", u[..., 1], weights[1]
        ) - torch.einsum("...km,mlk->...lm", v[..., 0], weights[0])
        output[..., 1, :, :, 1] = torch.einsum(
            "...km,mlk->...lm", u[..., 0], weights[1]
        ) - torch.einsum("...km,mlk->...lm", v[..., 1], weights[0])
        return torch.view_as_complex(output)


def longitudinal_multiplicity(
    mmax: int,
    *,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """Weights restoring the negative-m energy omitted by a real FFT."""
    if mmax < 1:
        raise ValueError("mmax must be positive")
    weights = torch.full((mmax,), 2.0, device=device, dtype=dtype)
    weights[0] = 1.0
    return weights


def coefficient_power(coefficients: torch.Tensor) -> torch.Tensor:
    """Return angular power by degree while restoring negative-m modes."""
    if not torch.is_complex(coefficients) or coefficients.ndim < 2:
        raise ValueError("coefficients must be complex with (...,ell,m) shape")
    multiplicity = longitudinal_multiplicity(
        coefficients.shape[-1],
        device=coefficients.device,
        dtype=coefficients.real.dtype,
    )
    return (
        (coefficients.real.square() + coefficients.imag.square())
        * multiplicity
    ).sum(dim=-1)


def coefficient_cross(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """Return the complex cross-spectrum by spherical-harmonic degree."""
    if prediction.shape != target.shape or not torch.is_complex(prediction):
        raise ValueError("prediction and target coefficients must match")
    multiplicity = longitudinal_multiplicity(
        prediction.shape[-1],
        device=prediction.device,
        dtype=prediction.real.dtype,
    )
    return (prediction * torch.conj(target) * multiplicity).sum(dim=-1)


def aligned_sht_band_objective(
    prediction: torch.Tensor,
    target: torch.Tensor,
    sht: CellCenteredRealSHT,
    *,
    ell_min: int,
    channel_weights: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Amplitude, shape, and signed-cospectral losses in one SHT band."""
    if prediction.shape != target.shape or prediction.ndim != 4:
        raise ValueError("prediction and target must have matching (B,C,H,W) shape")
    if not 0 <= ell_min < sht.lmax:
        raise ValueError("ell_min must lie inside the retained SHT range")
    pred_coeff = sht(prediction.double())
    with torch.no_grad():
        target_coeff = sht(target.double())
    pred_power = coefficient_power(pred_coeff)[..., ell_min:]
    target_power = coefficient_power(target_coeff)[..., ell_min:]
    cross = coefficient_cross(pred_coeff, target_coeff)[..., ell_min:]

    eps = torch.finfo(pred_power.dtype).eps
    pred_energy = pred_power.sum(dim=-1).clamp_min(eps)
    target_energy = target_power.sum(dim=-1).clamp_min(eps)
    amplitude = (torch.log(pred_energy) - torch.log(target_energy)).abs()
    pred_shape = pred_power / pred_energy.unsqueeze(-1)
    target_shape = target_power / target_energy.unsqueeze(-1)
    shape = (
        torch.log(pred_shape.clamp_min(1.0e-12))
        - torch.log(target_shape.clamp_min(1.0e-12))
    ).abs().mean(dim=-1)
    signed = cross.sum(dim=-1).real / torch.sqrt(pred_energy * target_energy)
    phase = 1.0 - signed.clamp(-1.0, 1.0)

    typical_energy = target_energy.detach().median(dim=0).values.clamp_min(eps)
    reliability = target_energy.detach() / (
        target_energy.detach() + 0.01 * typical_energy.unsqueeze(0)
    )
    if channel_weights is not None:
        weights = channel_weights.to(
            device=prediction.device,
            dtype=reliability.dtype,
        ).reshape(1, -1)
        if weights.shape[-1] != prediction.shape[1]:
            raise ValueError("channel_weights must contain one value per channel")
        reliability = reliability * weights
    denominator = reliability.sum().clamp_min(eps)

    def weighted_mean(value: torch.Tensor) -> torch.Tensor:
        return (value * reliability).sum() / denominator

    amplitude_loss = weighted_mean(amplitude)
    shape_loss = weighted_mean(shape)
    phase_loss = weighted_mean(phase)
    return {
        "amplitude": amplitude_loss,
        "shape": shape_loss,
        "phase": phase_loss,
        "total": 0.4 * amplitude_loss + 0.3 * shape_loss + 0.3 * phase_loss,
    }


def cell_center_latitudes(nlat: int) -> np.ndarray:
    """Return the degrees-north coordinates represented by Fejer-I nodes."""
    theta, _ = fejer1_nodes_weights(nlat)
    return 90.0 - np.rad2deg(theta)


def angular_wavelength_km(ell: int, radius_km: float = 6371.0) -> float:
    """Approximate full wavelength corresponding to spherical degree ``ell``."""
    if ell < 1 or not math.isfinite(radius_km) or radius_km <= 0.0:
        raise ValueError("ell and radius_km must be positive")
    return 2.0 * math.pi * radius_km / ell
