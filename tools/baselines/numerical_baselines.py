"""Physics-informed numerical baselines for 6h temporal interpolation.

Two methods, both use u,v wind fields from x0 and xT to add physics to the
trivial bilinear blend:

  1. semi_lagrangian_interp:  Lagrangian view — follow particle trajectories.
                              Symmetric blend of forward (from x0) and
                              backward (from xT) advected fields.
  2. hermite_advection_interp: Eulerian view — analytic tendency
                              ẋ = -u·∂x/∂lon - v·∂x/∂lat at endpoints,
                              cubic Hermite using (x0, ẋ0, xT, ẋT).

Per-variable wind assignment (matches our 27-channel ordering):
  - Pressure-level vars Tlev, Ulev, Vlev, Qlev, Zlev → U_lev, V_lev (same level)
  - t2m, u10, v10, mslp                              → U1000, V1000 (lowest PL)
  - tcc, tcwv                                        → U700, V700 (mid-trop)
  - sst                                              → no advection (slow boundary)

Both expect input tensors normalised (μ=0, σ=1) — denorm/renorm handled by caller.
"""
from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn.functional as F

from weather_time_interp.grid import (
    wb2_block_average_latitudes,
    wb2_block_average_longitudes,
)


# Channel name -> index in (20 PL + 7 surf = 27) layout. PL block first, by var
# (T,U,V,Q,Z) × levels (1000,925,850,700); then surface in order.
PL_VARS = ("T", "U", "V", "Q", "Z")
PL_LEVELS = (1000, 925, 850, 700)
SURF_VARS = ("t2m", "u10", "v10", "mslp", "sst", "tcc", "tcwv")


def build_channel_index(channel_names: List[str]) -> Dict[str, int]:
    return {c: i for i, c in enumerate(channel_names)}


def build_uv_assignment(channel_names: List[str]) -> Dict[int, tuple]:
    """For each channel idx → (u_idx, v_idx). None means no advection (zero wind).

    PL variable at level L is advected by U_L / V_L. Surface uses 1000hPa winds
    (closest to ground). Column-integrated (tcc, tcwv) use 700hPa winds (mid-trop
    representative). SST gets no advection (ocean currents not in our data).
    """
    idx = build_channel_index(channel_names)
    mapping: Dict[int, tuple] = {}
    # Pressure-level vars
    for v in PL_VARS:
        for lvl in PL_LEVELS:
            name = f"{v}{lvl}"
            if name not in idx:
                continue
            u_name, v_name = f"U{lvl}", f"V{lvl}"
            if u_name in idx and v_name in idx:
                mapping[idx[name]] = (idx[u_name], idx[v_name])
            else:
                mapping[idx[name]] = (-1, -1)
    # Surface vars use 10-m winds (closer to surface than 1000hPa due to BL friction).
    u10 = idx.get("u10", -1); v10 = idx.get("v10", -1)
    u700 = idx.get("U700", -1); v700 = idx.get("V700", -1)
    for sv in ("t2m", "u10", "v10", "mslp"):
        if sv in idx:
            mapping[idx[sv]] = (u10, v10)
    for sv in ("tcc", "tcwv"):
        if sv in idx:
            mapping[idx[sv]] = (u700, v700)
    if "sst" in idx:
        mapping[idx["sst"]] = (-1, -1)  # no advection: ocean currents not in data
    return mapping


def _build_lat_lon_grids(H: int, W: int, device, dtype) -> tuple:
    """Return physical centres for the evaluated latitude--longitude grid."""
    if H == 360:
        lat = torch.as_tensor(
            wb2_block_average_latitudes(H), device=device, dtype=dtype
        )
    elif H == 181:
        lat = torch.linspace(90.0, -90.0, H, device=device, dtype=dtype)
    else:
        # generic equal-spaced from +90-Δ/2 to -90+Δ/2
        dlat = 180.0 / H
        lat = torch.linspace(90.0 - dlat / 2.0, -90.0 + dlat / 2.0, H, device=device, dtype=dtype)
    if W == 720:
        lon = torch.as_tensor(
            wb2_block_average_longitudes(W), device=device, dtype=dtype
        )
    else:
        dlon = 360.0 / W
        lon = torch.linspace(
            0.0, 360.0 - dlon, W, device=device, dtype=dtype
        )
    return lat, lon


def _sample_field_at_departure(
    field: torch.Tensor,  # (B, C, H, W)
    dep_lat: torch.Tensor,  # (B, C, H, W) departure latitudes in deg
    dep_lon: torch.Tensor,  # (B, C, H, W) departure longitudes in deg, wrapped
) -> torch.Tensor:
    """Bilinear sample with cyclic-lon, clamped-lat boundaries.

    Returns same shape as field, sampled at (dep_lat, dep_lon).
    """
    B, C, H, W = field.shape
    if (H, W) != (360, 720):
        # Preserve the historical generic-grid path used by legacy 1-degree
        # experiments and compact synthetic regression fixtures.
        if H == 181:
            lat_min, lat_max = -90.0, 90.0
        else:
            dlat = 180.0 / H
            lat_min = -90.0 + dlat / 2.0
            lat_max = 90.0 - dlat / 2.0
        lat_norm = (
            (dep_lat - lat_max) / (lat_min - lat_max)
        ) * 2.0 - 1.0
        lon_norm = ((dep_lon % 360.0) / 360.0) * 2.0 - 1.0
        field_flat = field.reshape(B * C, 1, H, W)
        grid = torch.stack(
            (
                lon_norm.reshape(B * C, H, W),
                lat_norm.clamp(-1.0, 1.0).reshape(B * C, H, W),
            ),
            dim=-1,
        )
        return F.grid_sample(
            field_flat,
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=False,
        ).reshape(B, C, H, W)

    lat, lon = _build_lat_lon_grids(H, W, field.device, field.dtype)
    dlat = torch.abs(lat[1] - lat[0]) if H > 1 else field.new_tensor(1.0)
    dlon = field.new_tensor(360.0 / W)
    source_y = ((lat[0] - dep_lat) / dlat).clamp(0.0, float(H - 1))
    source_x = torch.remainder((dep_lon - lon[0]) / dlon, W)

    # Append the first longitude column. With align_corners=True, x=W is the
    # appended copy and interpolation across 359.625/0.125 degrees is cyclic.
    field_flat = field.reshape(B * C, 1, H, W)
    periodic = torch.cat((field_flat, field_flat[..., :1]), dim=-1)
    x_norm = source_x.reshape(B * C, H, W) / max(W, 1) * 2.0 - 1.0
    y_norm = source_y.reshape(B * C, H, W) / max(H - 1, 1) * 2.0 - 1.0
    grid = torch.stack((x_norm, y_norm), dim=-1)
    sampled = F.grid_sample(
        periodic,
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )
    return sampled.reshape(B, C, H, W)


def semi_lagrangian_interp(
    x0: torch.Tensor, xT: torch.Tensor,
    u0: torch.Tensor, v0: torch.Tensor,
    uT: torch.Tensor, vT: torch.Tensor,
    tau: torch.Tensor, dt_hours: float = 6.0,
    *, n_iter: int = 2,
) -> torch.Tensor:
    """Symmetric semi-Lagrangian advection (forward + backward), Stationary fields.

    Departure-point iteration (n_iter=2 is operational standard):
      p_fwd ≈ p - τ·Δt · ½(u(p_fwd_prev) + u(p))    # arrives at p starting from x0
      p_bwd ≈ p + (1-τ)·Δt · ½(u(p_bwd_prev) + u(p))  # arrived at p ending at xT

    Then:
      x̂(p) = (1-τ)·x0(p_fwd) + τ·xT(p_bwd)

    Wind unit assumed m/s. dt_hours converted internally to seconds.
    Grid coordinate updates use spherical geometry (R_earth = 6371 km).
    """
    B, C, H, W = x0.shape
    device = x0.device
    dtype = x0.dtype
    lat_deg_1d, lon_deg_1d = _build_lat_lon_grids(H, W, device, dtype)
    # Per-pixel lat, lon broadcast to (B, C, H, W)
    lat_grid = lat_deg_1d.view(1, 1, H, 1).expand(B, C, H, W)
    lon_grid = lon_deg_1d.view(1, 1, 1, W).expand(B, C, H, W)

    R_earth_m = 6_371_000.0
    dt_sec = dt_hours * 3600.0

    cos_lat = torch.cos(torch.deg2rad(lat_grid))                  # (B,C,H,W)
    # Avoid div-by-zero near poles
    cos_lat = cos_lat.clamp_min(1e-6)
    # Pre-compute degree-per-meter at each lat: dλ/m = 1 / (R cos φ); dφ/m = 1/R
    deg_per_meter_lat = torch.rad2deg(torch.tensor(1.0 / R_earth_m, device=device, dtype=dtype))
    deg_per_meter_lon = torch.rad2deg(1.0 / (R_earth_m * cos_lat))  # (B,C,H,W)

    # Reshape tau to broadcast (B,1,1,1) or (B,C,1,1)
    if tau.dim() == 1:
        tau_b = tau.view(B, 1, 1, 1).to(dtype)
    elif tau.dim() == 4:
        tau_b = tau.to(dtype)
    else:
        tau_b = tau.view(B, 1, 1, 1).to(dtype)

    one_minus_tau = (1.0 - tau_b)

    # --- Forward departure (from x0 view) ---
    # We're at p at time tau. Where did we come from at t=0?
    # p_0 = p - ∫_0^{τΔt} u(particle) dt  ≈  p - τΔt · u(p_0)
    # Iterate with mid-point: p_0^{k+1} = p - τΔt · u(½(p + p_0^k))
    dep_lat_fwd = lat_grid - tau_b * dt_sec * v0 * deg_per_meter_lat   # initial guess (≈ explicit Euler)
    dep_lon_fwd = lon_grid - tau_b * dt_sec * u0 * deg_per_meter_lon
    for _ in range(max(0, n_iter - 1)):
        # sample u, v at mid-point of (p, p_dep)
        mid_lat = 0.5 * (lat_grid + dep_lat_fwd)
        mid_lon = 0.5 * (lon_grid + dep_lon_fwd)
        u_mid = _sample_field_at_departure(u0, mid_lat, mid_lon)
        v_mid = _sample_field_at_departure(v0, mid_lat, mid_lon)
        dep_lat_fwd = lat_grid - tau_b * dt_sec * v_mid * deg_per_meter_lat
        dep_lon_fwd = lon_grid - tau_b * dt_sec * u_mid * deg_per_meter_lon
    x_fwd = _sample_field_at_departure(x0, dep_lat_fwd, dep_lon_fwd)
    # For zero-wind channels, grid_sample introduces blurring even at identity
    # positions — fall through to direct x0 (exact, matches bilinear pathway).
    no_adv_mask = ((u0.abs() < 1e-12) & (v0.abs() < 1e-12)).all(dim=(-2, -1), keepdim=True)  # (B,C,1,1)
    x_fwd = torch.where(no_adv_mask, x0, x_fwd)

    # --- Backward departure (from xT view) ---
    # We're at p at time τ. Where do we end up at t=Δt?
    # p_T = p + (1-τ)·Δt · u(p_T)
    dep_lat_bwd = lat_grid + one_minus_tau * dt_sec * vT * deg_per_meter_lat
    dep_lon_bwd = lon_grid + one_minus_tau * dt_sec * uT * deg_per_meter_lon
    for _ in range(max(0, n_iter - 1)):
        mid_lat = 0.5 * (lat_grid + dep_lat_bwd)
        mid_lon = 0.5 * (lon_grid + dep_lon_bwd)
        u_mid = _sample_field_at_departure(uT, mid_lat, mid_lon)
        v_mid = _sample_field_at_departure(vT, mid_lat, mid_lon)
        dep_lat_bwd = lat_grid + one_minus_tau * dt_sec * v_mid * deg_per_meter_lat
        dep_lon_bwd = lon_grid + one_minus_tau * dt_sec * u_mid * deg_per_meter_lon
    x_bwd = _sample_field_at_departure(xT, dep_lat_bwd, dep_lon_bwd)
    no_adv_mask_T = ((uT.abs() < 1e-12) & (vT.abs() < 1e-12)).all(dim=(-2, -1), keepdim=True)
    x_bwd = torch.where(no_adv_mask_T, xT, x_bwd)

    return one_minus_tau * x_fwd + tau_b * x_bwd


def _advection_tendency(
    x: torch.Tensor, u: torch.Tensor, v: torch.Tensor,
    lat_grid: torch.Tensor,
) -> torch.Tensor:
    """ẋ = -u·∂x/∂lon - v·∂x/∂lat  (advection-only, ignoring diffusion/sources).

    Spatial derivatives via central differences on the regular lat/lon grid.
    Units: u, v in m/s; ∂x/∂(lon|lat) in (1/deg); convert to (1/m) via R, cos φ.
    Returns ẋ in (x-units / second).
    """
    B, C, H, W = x.shape
    R_earth_m = 6_371_000.0
    # Central diff on lon (cyclic), lat (clamped at poles)
    dx_dlon = (torch.roll(x, shifts=-1, dims=-1) - torch.roll(x, shifts=1, dims=-1)) / 2.0
    dx_dlat = torch.zeros_like(x)
    dx_dlat[..., 1:-1, :] = (x[..., 2:, :] - x[..., :-2, :]) / 2.0
    dx_dlat[..., 0, :] = x[..., 1, :] - x[..., 0, :]
    dx_dlat[..., -1, :] = x[..., -1, :] - x[..., -2, :]
    # convert per-cell deltas to per-degree, then per-meter
    dlon_deg = 360.0 / W
    dlat_deg = 180.0 / H if H not in (181,) else 180.0 / (H - 1)
    cos_lat = torch.cos(torch.deg2rad(lat_grid)).clamp_min(1e-6)
    # ∂x/∂s_lon (per metre) = ∂x/∂lon[cells] / (R cos φ · dλ[radians])
    dlon_rad = torch.deg2rad(torch.tensor(dlon_deg, device=x.device, dtype=x.dtype))
    dlat_rad = torch.deg2rad(torch.tensor(dlat_deg, device=x.device, dtype=x.dtype))
    dx_ds_lon = dx_dlon / (R_earth_m * cos_lat * dlon_rad)        # (B,C,H,W)
    dx_ds_lat = dx_dlat / (R_earth_m * dlat_rad)
    # Note: lat decreases with row index → ∂x/∂s_lat for north-positive convention
    # gets a minus when latitudes are descending. Our lat_grid is descending, so the
    # raw ∂x/∂row gives ∂x/∂(−φ), invert sign for true northward derivative:
    dx_ds_lat_phys = -dx_ds_lat
    return -(u * dx_ds_lon + v * dx_ds_lat_phys)


def hermite_advection_interp(
    x0: torch.Tensor, xT: torch.Tensor,
    u0: torch.Tensor, v0: torch.Tensor,
    uT: torch.Tensor, vT: torch.Tensor,
    tau: torch.Tensor, dt_hours: float = 6.0,
) -> torch.Tensor:
    """Cubic Hermite using endpoint values + advective tendencies.

    Tendencies ẋ_0, ẋ_T are computed analytically from each endpoint's u, v
    via central differences (no observation of mid-window required).
    Hermite basis (Δt-normalised, τ ∈ [0,1]):
      x̂(τ) = h00·x0 + h10·Δt·ẋ_0 + h01·xT + h11·Δt·ẋ_T
      h00 = 2τ³ - 3τ² + 1
      h10 = τ³ - 2τ² + τ
      h01 = -2τ³ + 3τ²
      h11 = τ³ - τ²
    """
    B, C, H, W = x0.shape
    device, dtype = x0.device, x0.dtype
    lat_deg_1d, _ = _build_lat_lon_grids(H, W, device, dtype)
    lat_grid = lat_deg_1d.view(1, 1, H, 1).expand(B, C, H, W)
    dt_sec = dt_hours * 3600.0
    xdot0 = _advection_tendency(x0, u0, v0, lat_grid) * dt_sec   # in x-units per Δt
    xdotT = _advection_tendency(xT, uT, vT, lat_grid) * dt_sec
    if tau.dim() == 1:
        tau_b = tau.view(B, 1, 1, 1).to(dtype)
    elif tau.dim() == 4:
        tau_b = tau.to(dtype)
    else:
        tau_b = tau.view(B, 1, 1, 1).to(dtype)
    t2 = tau_b * tau_b
    t3 = t2 * tau_b
    h00 = 2 * t3 - 3 * t2 + 1.0
    h10 = t3 - 2 * t2 + tau_b
    h01 = -2 * t3 + 3 * t2
    h11 = t3 - t2
    return h00 * x0 + h10 * xdot0 + h01 * xT + h11 * xdotT


def settls_interp(
    x0: torch.Tensor, xT: torch.Tensor,
    u0: torch.Tensor, v0: torch.Tensor,
    uT: torch.Tensor, vT: torch.Tensor,
    tau: torch.Tensor, dt_hours: float = 6.0,
    *, n_iter: int = 3,
) -> torch.Tensor:
    """SETTLS-style semi-Lagrangian (Hortal 2002, ECMWF QJRMS 128).

    Improvement over plain semi-Lagrangian: trajectory uses TIME-blended wind
    at the trajectory mid-point rather than endpoint-only wind. For the
    forward leg from x_0 to time τ:
        u_along(p_mid, t_mid=τ/2) ≈ (1-τ/2)·u_0(p_mid) + (τ/2)·u_T(p_mid)
    Then iterate departure-point estimation (3 iterations is operational).
    """
    B, C, H, W = x0.shape
    device, dtype = x0.device, x0.dtype
    lat_deg_1d, lon_deg_1d = _build_lat_lon_grids(H, W, device, dtype)
    lat_grid = lat_deg_1d.view(1, 1, H, 1).expand(B, C, H, W)
    lon_grid = lon_deg_1d.view(1, 1, 1, W).expand(B, C, H, W)
    R_earth_m = 6_371_000.0
    dt_sec = dt_hours * 3600.0
    cos_lat = torch.cos(torch.deg2rad(lat_grid)).clamp_min(1e-6)
    deg_per_meter_lat = torch.rad2deg(torch.tensor(1.0 / R_earth_m, device=device, dtype=dtype))
    deg_per_meter_lon = torch.rad2deg(1.0 / (R_earth_m * cos_lat))
    if tau.dim() == 1:
        tau_b = tau.view(B, 1, 1, 1).to(dtype)
    elif tau.dim() == 4:
        tau_b = tau.to(dtype)
    else:
        tau_b = tau.view(B, 1, 1, 1).to(dtype)
    one_minus_tau = 1.0 - tau_b
    half_tau = 0.5 * tau_b
    half_one_minus_tau = 0.5 * one_minus_tau

    # --- Forward leg (x_0 → x_τ) ---
    # Mid-time blend weight for trajectory midpoint: w_T = τ/2
    dep_lat = lat_grid - tau_b * dt_sec * v0 * deg_per_meter_lat
    dep_lon = lon_grid - tau_b * dt_sec * u0 * deg_per_meter_lon
    for _ in range(max(0, n_iter - 1)):
        mid_lat = 0.5 * (lat_grid + dep_lat)
        mid_lon = 0.5 * (lon_grid + dep_lon)
        u0_mid = _sample_field_at_departure(u0, mid_lat, mid_lon)
        v0_mid = _sample_field_at_departure(v0, mid_lat, mid_lon)
        uT_mid = _sample_field_at_departure(uT, mid_lat, mid_lon)
        vT_mid = _sample_field_at_departure(vT, mid_lat, mid_lon)
        # Time-blended at trajectory mid-time τ/2
        u_blend = (1.0 - half_tau) * u0_mid + half_tau * uT_mid
        v_blend = (1.0 - half_tau) * v0_mid + half_tau * vT_mid
        dep_lat = lat_grid - tau_b * dt_sec * v_blend * deg_per_meter_lat
        dep_lon = lon_grid - tau_b * dt_sec * u_blend * deg_per_meter_lon
    x_fwd = _sample_field_at_departure(x0, dep_lat, dep_lon)
    no_adv = ((u0.abs() < 1e-12) & (v0.abs() < 1e-12)).all(dim=(-2, -1), keepdim=True)
    x_fwd = torch.where(no_adv, x0, x_fwd)

    # --- Backward leg (x_T → x_τ) ---
    # Mid-time blend at t = τ + (1-τ)/2 = (1+τ)/2, weight on x_T side = (1+τ)/2
    w_T_bwd = 0.5 * (1.0 + tau_b)
    dep_lat = lat_grid + one_minus_tau * dt_sec * vT * deg_per_meter_lat
    dep_lon = lon_grid + one_minus_tau * dt_sec * uT * deg_per_meter_lon
    for _ in range(max(0, n_iter - 1)):
        mid_lat = 0.5 * (lat_grid + dep_lat)
        mid_lon = 0.5 * (lon_grid + dep_lon)
        u0_mid = _sample_field_at_departure(u0, mid_lat, mid_lon)
        v0_mid = _sample_field_at_departure(v0, mid_lat, mid_lon)
        uT_mid = _sample_field_at_departure(uT, mid_lat, mid_lon)
        vT_mid = _sample_field_at_departure(vT, mid_lat, mid_lon)
        u_blend = (1.0 - w_T_bwd) * u0_mid + w_T_bwd * uT_mid
        v_blend = (1.0 - w_T_bwd) * v0_mid + w_T_bwd * vT_mid
        dep_lat = lat_grid + one_minus_tau * dt_sec * v_blend * deg_per_meter_lat
        dep_lon = lon_grid + one_minus_tau * dt_sec * u_blend * deg_per_meter_lon
    x_bwd = _sample_field_at_departure(xT, dep_lat, dep_lon)
    no_adv_T = ((uT.abs() < 1e-12) & (vT.abs() < 1e-12)).all(dim=(-2, -1), keepdim=True)
    x_bwd = torch.where(no_adv_T, xT, x_bwd)

    return one_minus_tau * x_fwd + tau_b * x_bwd


def _laplacian(x: torch.Tensor, lat_grid: torch.Tensor) -> torch.Tensor:
    """∇²x on spherical lat/lon grid (per metre²). Central differences."""
    B, C, H, W = x.shape
    R_earth_m = 6_371_000.0
    dlon_deg = 360.0 / W
    dlat_deg = 180.0 / H if H not in (181,) else 180.0 / (H - 1)
    cos_lat = torch.cos(torch.deg2rad(lat_grid)).clamp_min(1e-6)
    dlon_rad = torch.deg2rad(torch.tensor(dlon_deg, device=x.device, dtype=x.dtype))
    dlat_rad = torch.deg2rad(torch.tensor(dlat_deg, device=x.device, dtype=x.dtype))
    # ∂²x/∂lon² (cyclic)
    d2_lon = (torch.roll(x, -1, dims=-1) - 2.0 * x + torch.roll(x, 1, dims=-1))
    # ∂²x/∂lat² (clamped at poles)
    d2_lat = torch.zeros_like(x)
    d2_lat[..., 1:-1, :] = x[..., 2:, :] - 2.0 * x[..., 1:-1, :] + x[..., :-2, :]
    # convert per-cell to per-metre²
    inv_dlon_m_sq = 1.0 / ((R_earth_m * cos_lat * dlon_rad) ** 2)
    inv_dlat_m_sq = 1.0 / ((R_earth_m * dlat_rad) ** 2)
    return d2_lon * inv_dlon_m_sq + d2_lat * inv_dlat_m_sq


def hermite_diffusion_interp(
    x0: torch.Tensor, xT: torch.Tensor,
    u0: torch.Tensor, v0: torch.Tensor,
    uT: torch.Tensor, vT: torch.Tensor,
    tau: torch.Tensor, dt_hours: float = 6.0,
    *, kappa_m2_s: float = 1.0e4,
) -> torch.Tensor:
    """Hermite-advection augmented with eddy-diffusion term.

    Tendency: ẋ = -u·∂x/∂lon - v·∂x/∂lat + κ·∇²x
    Default κ = 1e4 m²/s (typical horizontal eddy diffusivity in atmosphere;
    Pielke 2002, Stull 1988).
    """
    B, C, H, W = x0.shape
    device, dtype = x0.device, x0.dtype
    lat_deg_1d, _ = _build_lat_lon_grids(H, W, device, dtype)
    lat_grid = lat_deg_1d.view(1, 1, H, 1).expand(B, C, H, W)
    dt_sec = dt_hours * 3600.0
    xdot0 = _advection_tendency(x0, u0, v0, lat_grid)
    xdotT = _advection_tendency(xT, uT, vT, lat_grid)
    xdot0 = xdot0 + kappa_m2_s * _laplacian(x0, lat_grid)
    xdotT = xdotT + kappa_m2_s * _laplacian(xT, lat_grid)
    xdot0 = xdot0 * dt_sec
    xdotT = xdotT * dt_sec
    if tau.dim() == 1:
        tau_b = tau.view(B, 1, 1, 1).to(dtype)
    elif tau.dim() == 4:
        tau_b = tau.to(dtype)
    else:
        tau_b = tau.view(B, 1, 1, 1).to(dtype)
    t2 = tau_b * tau_b
    t3 = t2 * tau_b
    h00 = 2 * t3 - 3 * t2 + 1.0
    h10 = t3 - 2 * t2 + tau_b
    h01 = -2 * t3 + 3 * t2
    h11 = t3 - t2
    return h00 * x0 + h10 * xdot0 + h01 * xT + h11 * xdotT


def expand_uv_to_channels(
    x: torch.Tensor,                # (B, C, H, W)
    uv_map: Dict[int, tuple],       # channel idx -> (u_idx, v_idx) or (-1,-1)
) -> tuple:
    """Return (u_per_channel, v_per_channel) tensors of same shape as x: each
    channel slot c holds u_field / v_field for the wind assigned to c. Channels
    with no advection get zero wind (so methods degenerate to linear-in-time).
    """
    B, C, H, W = x.shape
    u_out = torch.zeros_like(x)
    v_out = torch.zeros_like(x)
    for c in range(C):
        ui, vi = uv_map.get(c, (-1, -1))
        if ui >= 0 and vi >= 0:
            u_out[:, c] = x[:, ui]
            v_out[:, c] = x[:, vi]
    return u_out, v_out
