"""Radial-averaged 2D power spectrum for weather fields.

Energy spectrum E(k) = average over rings in wavenumber space of |FFT(field)|².

Why this metric matters:
- bilinear interpolation kills high-k (small-scale) features → undersmoothed spectrum
- a good neural model matches truth's E(k) across all wavenumbers
- spectral fidelity = physical realism (turbulent cascade, mesoscale features)

Reference: Skamarock 2004 "Evaluating mesoscale NWP models using kinetic energy spectra".

Usage:
    >>> from weather_time_interp.metrics.energy_spectra import radial_psd
    >>> field = torch.randn(B, H, W)
    >>> k, e = radial_psd(field)              # k: (n_bins,), e: (B, n_bins)
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
import torch


def radial_psd(
    field: torch.Tensor,
    n_bins: int | None = None,
    lat_weights: torch.Tensor | None = None,
    detrend: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Radial-averaged power spectral density of a 2D field.

    Parameters
    ----------
    field : (B, H, W) or (H, W) tensor
        Spatial field. For ERA5 on (181, 360), recommend pre-cropping to (180, 360).
    n_bins : int, default = min(H,W) // 2
        Number of radial wavenumber bins.
    lat_weights : (H,) tensor or None
        Per-row weights (e.g. cos(lat)) applied before FFT to handle equiangular grid bias.
    detrend : bool
        Subtract spatial mean per sample (remove DC) before FFT.

    Returns
    -------
    k_centers : np.ndarray (n_bins,)
        Wavenumber bin centers (normalized to grid units, k ∈ [1, k_nyquist]).
    psd : np.ndarray (B, n_bins)  or (n_bins,) for 2D input
        Energy E(k) per sample.
    """
    if field.dim() == 2:
        squeeze = True
        field = field.unsqueeze(0)
    else:
        squeeze = False

    B, H, W = field.shape
    if n_bins is None:
        n_bins = min(H, W) // 2

    x = field.float()
    if detrend:
        x = x - x.mean(dim=(-1, -2), keepdim=True)

    if lat_weights is not None:
        lw = lat_weights.float().view(1, -1, 1)
        x = x * lw

    # 2D FFT and shift so 0-freq at center
    fft = torch.fft.fft2(x, norm="ortho")
    psd_2d = (fft.real ** 2 + fft.imag ** 2)  # (B, H, W)

    # Wavenumber grid (centered)
    ky = torch.fft.fftfreq(H, d=1.0)
    kx = torch.fft.fftfreq(W, d=1.0)
    ky_g, kx_g = torch.meshgrid(ky, kx, indexing="ij")
    k_rad = torch.sqrt(ky_g ** 2 + kx_g ** 2)  # (H, W) in [0, k_nyq]

    # Bin edges in normalized wavenumber (skip k=0)
    k_max = k_rad.max().item()
    edges = np.linspace(0.0, k_max, n_bins + 1)
    centers = 0.5 * (edges[1:] + edges[:-1])

    k_flat = k_rad.flatten().cpu().numpy()
    psd_flat = psd_2d.view(B, -1).cpu().numpy()

    # Vectorized binning: digitize once, accumulate per bin
    bin_idx = np.digitize(k_flat, edges) - 1
    bin_idx = np.clip(bin_idx, 0, n_bins - 1)

    psd_out = np.zeros((B, n_bins), dtype=np.float64)
    counts = np.zeros(n_bins, dtype=np.int64)
    for b in range(n_bins):
        mask = bin_idx == b
        if mask.any():
            counts[b] = mask.sum()
            psd_out[:, b] = psd_flat[:, mask].mean(axis=1)

    # Replace empty bins with NaN (no samples in that ring)
    psd_out[:, counts == 0] = np.nan

    if squeeze:
        psd_out = psd_out[0]
    return centers, psd_out


def spectral_distance(
    psd_pred: np.ndarray,
    psd_truth: np.ndarray,
    log: bool = True,
    weight_high_k: bool = False,
) -> float:
    """Mean absolute difference of (log-)spectra.

    Parameters
    ----------
    psd_pred, psd_truth : (B, n_bins) or (n_bins,)
    log : if True, compare log10(E(k)) — more physically meaningful
    weight_high_k : if True, weight bins by k (emphasize small-scale fidelity)

    Returns
    -------
    distance : float — mean absolute diff (lower = better spectral match)
    """
    p = np.asarray(psd_pred, dtype=np.float64)
    t = np.asarray(psd_truth, dtype=np.float64)
    if log:
        p = np.log10(np.clip(p, 1e-12, None))
        t = np.log10(np.clip(t, 1e-12, None))
    diff = np.abs(p - t)
    valid = np.isfinite(diff)
    if weight_high_k:
        k = np.arange(diff.shape[-1], dtype=np.float64) + 1.0
        diff = diff * k
    return float(diff[valid].mean())
