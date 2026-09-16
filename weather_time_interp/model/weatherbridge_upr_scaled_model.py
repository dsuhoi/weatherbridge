"""Capacity-matched UPR variants for quality-first comparisons."""
from __future__ import annotations


UPR_SCALED_VARIANTS = (
    "upr_implicit_global_14m",
    "upr_endpoint_implicit_global_14m",
    "upr_query_match_14m",
    "upr_local_corr_14m",
    "upr_universal_latent_q4_10m",
)


def upr_scaled_variant_kwargs(arch: str) -> dict[str, object]:
    """Return the fixed approximately 14M Implicit-Global configuration."""
    if arch not in UPR_SCALED_VARIANTS:
        raise ValueError(f"unknown scaled UPR variant {arch}")
    universal_q4 = arch == "upr_universal_latent_q4_10m"
    query_match = arch == "upr_query_match_14m"
    local_corr = arch == "upr_local_corr_14m"
    return {
        "in_channels": 24,
        "out_channels": 24,
        "n_static_features": 3,
        "hidden": 400 if universal_q4 else 500 if query_match else 504,
        "static_width": 32,
        "n_blocks": 10,
        "n_flow_modes": 1 if universal_q4 else 3,
        "laplacian_detail": True,
        "query_conditioned": False,
        "quadratic_trajectory": False,
        "cubic_trajectory": True,
        "global_tokens": 32,
        "global_token_dim": 128 if universal_q4 else 96 if query_match else 112,
        "global_transformer_depth": 2 if universal_q4 else 0,
        "global_transformer_heads": 4,
        "global_area_weighted": universal_q4,
        "hydrostatic_coupling": not universal_q4,
        "smooth_endpoint_blend": (
            arch == "upr_endpoint_implicit_global_14m" or universal_q4
        ),
        "local_matching": query_match,
        "local_correlation_radius": 2 if local_corr else 0,
        "matching_width": 32,
        "query_state_modulation": query_match or universal_q4,
        "periodic_longitude_resize": query_match or local_corr or universal_q4,
        "flow_scale": 2.0,
        "pyramid_divisors": (16, 8, 4) if universal_q4 else (8, 4, 2),
    }
