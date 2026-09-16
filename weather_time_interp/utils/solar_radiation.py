"""Analytic TOA Incident Solar Radiation (TISR) on a lat/lon grid.

Replaces ERA5-stored `tisr` field with on-the-fly compute. Saves
~8 GiB/year on disk and ~10 GiB/year in RAM cache, and gives exact values
for arbitrary timestamps (useful for time-interpolation supervision).

Output unit: J/m² accumulated over a 1-hour interval (matches ERA5 storage),
i.e. `S0 * eccentricity * ∫ max(0, cos(zenith)) dt` over the hour, with the
integral approximated with mid-point rule on N substeps.

Reference: Spencer 1971 ("Fourier series representation of the position of
the sun"), used by ECMWF/IFS for low-frequency solar declination.
"""
from __future__ import annotations

from typing import Sequence, Union

import numpy as np

# Solar constant (W/m²). ERA5 uses 1361 ± a tiny seasonal modulation
# (TSI). For interpolation we treat S0 as constant.
S0 = 1361.0
SECONDS_PER_HOUR = 3600.0
DEG = np.pi / 180.0


def _spencer_decl(doy_frac: np.ndarray) -> np.ndarray:
    """Solar declination (rad) as function of fractional day-of-year (Spencer 1971)."""
    g = 2.0 * np.pi * (doy_frac - 1.0) / 365.25
    return (
        0.006918
        - 0.399912 * np.cos(g) + 0.070257 * np.sin(g)
        - 0.006758 * np.cos(2 * g) + 0.000907 * np.sin(2 * g)
        - 0.002697 * np.cos(3 * g) + 0.001480 * np.sin(3 * g)
    )


def _spencer_ecc(doy_frac: np.ndarray) -> np.ndarray:
    """(R̄/R(t))² Earth-Sun distance squared correction (Spencer 1971)."""
    g = 2.0 * np.pi * (doy_frac - 1.0) / 365.25
    return (
        1.000110
        + 0.034221 * np.cos(g) + 0.001280 * np.sin(g)
        + 0.000719 * np.cos(2 * g) + 0.000077 * np.sin(2 * g)
    )


def _spencer_eot_minutes(doy_frac: np.ndarray) -> np.ndarray:
    """Equation of Time, in minutes (Spencer 1971). Range ~ −14..+16 min.

    EOT = solar_time − mean_solar_time. Without it, terminator pixels far from
    equinox can have several-degree lon offset → huge relative TISR error.
    """
    g = 2.0 * np.pi * (doy_frac - 1.0) / 365.25
    return 229.18 * (
        0.000075
        + 0.001868 * np.cos(g) - 0.032077 * np.sin(g)
        - 0.014615 * np.cos(2 * g) - 0.040849 * np.sin(2 * g)
    )


def _to_doy_and_hour(times: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """times: ndarray of np.datetime64[ns] -> (doy_frac, hour_utc)."""
    times = np.asarray(times, dtype="datetime64[ns]")
    days = times.astype("datetime64[D]")
    year_starts = days.astype("datetime64[Y]").astype("datetime64[D]")
    doy = (days - year_starts).astype("int64") + 1  # 1..366
    hours = ((times - days).astype("timedelta64[s]").astype("int64")) / SECONDS_PER_HOUR
    return doy.astype(np.float64), hours.astype(np.float64)


def _to_jd(times: np.ndarray) -> np.ndarray:
    """Julian date for UTC times. Reference J2000 = 2451545.0."""
    times = np.asarray(times, dtype="datetime64[ns]")
    # ns since J2000 epoch (2000-01-01T12:00 TT, here approximated as UTC)
    j2000 = np.datetime64("2000-01-01T12:00:00", "ns")
    delta_ns = (times - j2000).astype("int64")
    return 2451545.0 + delta_ns / (1e9 * 86400.0)


def _meeus_decl_eot(times: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Astronomical-Almanac low-precision algorithm (Meeus 1998 §25).
    Accuracy ≈ 0.01° (≈ 36 arcsec) for declination and ≈ 1 sec for EOT.
    Returns (declination_rad, eot_minutes, eccentricity_factor).

    50× more accurate than Spencer 1971 (~0.5°). No external deps.
    """
    jd = _to_jd(times)
    n = jd - 2451545.0                                  # days from J2000
    L = (280.46646 + 0.9856474 * n) % 360.0             # mean longitude (deg)
    g = np.deg2rad((357.52911 + 0.9856003 * n) % 360.0) # mean anomaly (rad)
    # ecliptic longitude with equation of centre + minor higher-order terms
    lam = np.deg2rad(L + 1.914602 * np.sin(g) + 0.019993 * np.sin(2 * g))
    eps = np.deg2rad(23.43928 - 3.563e-7 * n)            # obliquity (rad)

    # declination
    sin_dec = np.sin(eps) * np.sin(lam)
    decl = np.arcsin(sin_dec)

    # right ascension (atan2 to keep correct quadrant)
    alpha = np.arctan2(np.cos(eps) * np.sin(lam), np.cos(lam))
    # equation of time = (L_sun − α) in degrees, modulo 360 → minutes
    L_rad = np.deg2rad(L % 360.0)
    eot_deg = np.rad2deg(L_rad - alpha)
    eot_deg = (eot_deg + 180.0) % 360.0 - 180.0          # wrap to [-180,180]
    eot_min = eot_deg * 4.0                              # 1° = 4 min

    # Earth-Sun distance squared correction (R̄/R)²
    # R = 1.00014 − 0.01671·cos(g) − 0.00014·cos(2g) AU; (1/R)²
    R_au = 1.00014 - 0.01671 * np.cos(g) - 0.00014 * np.cos(2 * g)
    ecc = 1.0 / (R_au * R_au)

    return decl, eot_min, ecc


def hourly_tisr_accumulated(
    times: np.ndarray,
    latitudes: Union[np.ndarray, Sequence[float]],
    longitudes: Union[np.ndarray, Sequence[float]],
    n_substeps: int = 16,
) -> np.ndarray:
    """Compute ERA5-style hour-accumulated TISR (J/m²) for grid (T, H, W).

    Each output sample is the integral of incoming top-of-atmosphere irradiance
    over the 1-hour window ENDING at `times[t]` (ERA5 convention: accumulated
    fields are stamped at the end of the accumulation period; verified to give
    median 0.6% / p95 11% rel. error vs ERA5 stored TISR on 2018-06-21).

    Parameters
    ----------
    times : array of np.datetime64[ns], shape (T,)
        End of each hourly window (UTC).
    latitudes : array, shape (H,)
        Latitudes in degrees, [-90, 90].
    longitudes : array, shape (W,)
        Longitudes in degrees, [0, 360) or (-180, 180].
    n_substeps : int
        Mid-point rule substeps within each 1-hour window. 16 ⇒ ~1% accuracy
        on terminator pixels; 32 ⇒ ~0.3%.

    Returns
    -------
    tisr : ndarray, float32, shape (T, H, W)
        Hour-accumulated TISR in J/m².
    """
    times = np.asarray(times, dtype="datetime64[ns]")
    lat = np.asarray(latitudes, dtype=np.float64)
    lon = np.asarray(longitudes, dtype=np.float64)
    T = len(times)
    H = len(lat)
    W = len(lon)

    sin_lat_1d = np.sin(lat * DEG)                    # (H,)
    cos_lat_1d = np.cos(lat * DEG)                    # (H,)
    lon_rad_1d = lon * DEG                            # (W,)

    _, hour_utc = _to_doy_and_hour(times)             # only need UTC hour-of-day
    # Use Meeus low-precision (Astronomical Almanac) instead of Spencer 1971.
    # Spencer ≈ 0.5° declination; Meeus ≈ 0.01° declination, sub-second EOT.
    decl, eot_min, ecc = _meeus_decl_eot(times)       # (T,), (T,) min, (T,)
    eot_h = eot_min / 60.0                            # (T,) hours
    sin_decl = np.sin(decl)                           # (T,)
    cos_decl = np.cos(decl)                           # (T,)

    # Integrate cos(zenith) over [t-1, t] hour using mid-point rule.
    # Vectorise substeps inside one timestamp; loop over time (chunking T avoids
    # the (T, N, H, W) tensor blowing past available RAM).
    out = np.zeros((T, H, W), dtype=np.float32)
    sub_offsets = (np.arange(n_substeps) + 0.5) / n_substeps               # (N,)
    base_factor = (S0 * SECONDS_PER_HOUR) / n_substeps

    # Pre-broadcast static lat/lon to 3D shapes (axes: substep, lat, lon)
    sin_lat_b = sin_lat_1d[None, :, None]                                  # (1, H, 1)
    cos_lat_b = cos_lat_1d[None, :, None]                                  # (1, H, 1)
    lon_rad_b = lon_rad_1d[None, None, :]                                  # (1, 1, W)

    for it in range(T):
        # ERA5 stamp = end of period → integrate over [t-1, t]
        sub_h = hour_utc[it] - 1.0 + sub_offsets                           # (N,)
        # Hour angle (with EOT)
        H_ang = (sub_h + eot_h[it] - 12.0)[:, None, None] * (np.pi / 12.0) + lon_rad_b  # (N, 1, W)
        cos_z = sin_lat_b * sin_decl[it] + cos_lat_b * cos_decl[it] * np.cos(H_ang)     # (N, H, W)
        accum = np.clip(cos_z, 0.0, None).sum(axis=0)                      # (H, W)
        out[it] = (base_factor * ecc[it] * accum).astype(np.float32)
    return out


def hourly_tisr_accumulated_torch(
    times: np.ndarray,
    latitudes,
    longitudes,
    n_substeps: int = 64,
    device: str = "cuda:0",
    chunk_T: int = 256,
) -> np.ndarray:
    """GPU-accelerated version using PyTorch. Returns numpy float32 (T, H, W).

    Algorithm identical to `hourly_tisr_accumulated` (Meeus + EOT + ERA5 end-of-period).
    Memory: chunk_T * n_substeps * H * W * 4 B per chunk. Default chunk_T=256
    on (181×360, n=64) ~ 5.3 GB peak.
    """
    import torch
    times = np.asarray(times, dtype="datetime64[ns]")
    lat = np.asarray(latitudes, dtype=np.float64)
    lon = np.asarray(longitudes, dtype=np.float64)
    T = len(times)
    H = len(lat)
    W = len(lon)

    # Astronomical scalars on CPU (cheap, T scalars only)
    decl_np, eot_min_np, ecc_np = _meeus_decl_eot(times)
    _, hour_utc_np = _to_doy_and_hour(times)
    eot_h_np = eot_min_np / 60.0

    dev = torch.device(device)
    dtype = torch.float32  # GPU compute fp32 is sufficient (median <0.06%)

    sin_decl = torch.tensor(np.sin(decl_np), dtype=dtype, device=dev)        # (T,)
    cos_decl = torch.tensor(np.cos(decl_np), dtype=dtype, device=dev)        # (T,)
    eot_h = torch.tensor(eot_h_np, dtype=dtype, device=dev)                  # (T,)
    ecc = torch.tensor(ecc_np, dtype=dtype, device=dev)                      # (T,)
    hour_utc = torch.tensor(hour_utc_np, dtype=dtype, device=dev)            # (T,)

    sin_lat = torch.tensor(np.sin(lat * DEG), dtype=dtype, device=dev)       # (H,)
    cos_lat = torch.tensor(np.cos(lat * DEG), dtype=dtype, device=dev)       # (H,)
    lon_rad = torch.tensor(lon * DEG, dtype=dtype, device=dev)               # (W,)
    sub_offsets = (torch.arange(n_substeps, device=dev, dtype=dtype) + 0.5) / n_substeps  # (N,)
    base_factor = (S0 * SECONDS_PER_HOUR) / n_substeps

    out = np.empty((T, H, W), dtype=np.float32)
    for c0 in range(0, T, chunk_T):
        c1 = min(c0 + chunk_T, T)
        Tc = c1 - c0
        # (Tc, N) hour offsets
        sub_h = hour_utc[c0:c1, None] - 1.0 + sub_offsets[None, :]           # (Tc, N)
        # Hour angle (Tc, N, 1, W)
        H_ang = (sub_h + eot_h[c0:c1, None] - 12.0)[:, :, None, None] * (np.pi / 12.0) \
                + lon_rad[None, None, None, :]
        # cos(zenith) (Tc, N, H, W)
        cos_z = (
            sin_lat[None, None, :, None] * sin_decl[c0:c1, None, None, None]
            + cos_lat[None, None, :, None] * cos_decl[c0:c1, None, None, None] * torch.cos(H_ang)
        )
        # accumulate over substeps with clamp
        accum = cos_z.clamp_min(0.0).sum(dim=1)                              # (Tc, H, W)
        chunk = (base_factor * ecc[c0:c1, None, None]) * accum               # (Tc, H, W)
        out[c0:c1] = chunk.cpu().numpy().astype(np.float32)
    return out
