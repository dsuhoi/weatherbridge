"""Canonical normalised → physical RMSE conversion for paper figures.

The stored `rmse_model_phys` field in legacy eval JSONs was denormalised
with inconsistent σ per channel (mslp/Z off by 30-70×, Q by ~5000×).
Always go through this module instead of reading rmse_phys directly:

    from paper_units import canonical_stds, UNITS, denorm_phys
    σ = canonical_stds()
    phys = denorm_phys(rmse_norm, channel, σ)   # native physical units

Channels list and the storage-unit → display-unit scale are:
  T*, t2m            K        ×1
  U*, V*, u10, v10   m s⁻¹    ×1
  Z*                 m² s⁻²   ×1
  mslp               Pa       ×1
  Q*                 g kg⁻¹   ×1000   (σ_for_denorm is kg/kg)
"""
from __future__ import annotations
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

CHANNELS_ORDER = [
    "T1000", "T925", "T850", "T700",
    "U1000", "U925", "U850", "U700",
    "V1000", "V925", "V850", "V700",
    "Q1000", "Q925", "Q850", "Q700",
    "Z1000", "Z925", "Z850", "Z700",
    "t2m", "u10", "v10", "mslp",
]

UNITS = {
    "T": r"K",
    "U": r"m s$^{-1}$",
    "V": r"m s$^{-1}$",
    "Q": r"g kg$^{-1}$",
    "Z": r"m$^2$ s$^{-2}$",
    "t2m": r"K",
    "u10": r"m s$^{-1}$",
    "v10": r"m s$^{-1}$",
    "mslp": r"Pa",
}

UNIT_SCALE = {
    "T": 1.0, "U": 1.0, "V": 1.0, "Z": 1.0,
    "Q": 1000.0,                     # kg/kg → g/kg
    "t2m": 1.0, "u10": 1.0, "v10": 1.0, "mslp": 1.0,
}


def channel_family(ch: str) -> str:
    if ch in ("t2m", "u10", "v10", "mslp"):
        return ch
    return ch[0]


def channel_unit(ch: str) -> str:
    return UNITS[channel_family(ch)]


_CACHED_STDS: dict[str, float] | None = None


def canonical_stds() -> dict[str, float]:
    """Single canonical σ map (24-channel ERA5 0.5° σ) shared across all
    figures. Sourced from the DC-AE NoSkip 6yr eval JSON."""
    global _CACHED_STDS
    if _CACHED_STDS is None:
        p = ROOT / "metrics" / "eval_0p5_2020_cloudpipe_normalized" / \
            "weatherdcae_noskip_24ch_6yr_ep8.json"
        d = json.load(open(p))
        _CACHED_STDS = {k: float(v) for k, v in d["stds_for_denorm"].items()}
    return _CACHED_STDS


def denorm_phys(rmse_norm: float, channel: str, stds: dict[str, float]) -> float:
    """Convert a normalised RMSE for `channel` into its display physical
    unit (K / m·s⁻¹ / Pa / m²·s⁻² / g·kg⁻¹). Q gets the kg→g ×1000 factor."""
    return float(rmse_norm) * stds[channel] * UNIT_SCALE[channel_family(channel)]
