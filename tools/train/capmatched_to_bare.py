"""Convert a CapMatchedLit checkpoint → bare blob for the standard eval path.

The capacity-matched trainer wraps each backbone in ``CapMatchedLit``
(state_dict keys prefixed ``net.``). This strips the wrapper and writes a
``{arch, kwargs, state_dict}`` bare blob compatible with
``examples/_bare_loader.load_bare`` and the anchor / downstream eval
scripts.

  python capmatched_to_bare.py --ckpt exp_wb_mamba_14m_6h/last.ckpt \
    --arch mamba --out weights/wb_mamba_14m_6h_bare.pt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch

from tools.eval.capmatched_loader import (
    normalize_weatherbridge_decoder_state_dict,
)
from weather_time_interp.model.weatherbridge_upr_lite_model import (
    UPR_LITE_VARIANTS,
    upr_lite_variant_kwargs,
)
from weather_time_interp.model.weatherbridge_upr_scaled_model import (
    UPR_SCALED_VARIANTS,
    upr_scaled_variant_kwargs,
)

ARCH_META = {
    "mamba": {
        "cls": "WeatherBridgeMambaModel",
        "kwargs": dict(in_channels=24, out_channels=24, n_static_features=3,
                       hidden=64, n_levels=3, d_state=16, lat_crop=0),
    },
    "crossframe": {
        "cls": "WeatherDCAECrossFrameModel",
        "kwargs": dict(in_channels=24, out_channels=24, n_static_features=3,
                       latent_channels=32, attention_head_dim=32,
                       block_type=("ResBlock", "ResBlock", "EfficientViTBlock"),
                       qkv_multiscales=((), (), (5,)), lat_crop=-8,
                       block_out_channels=(128, 128, 256, 256),
                       layers_per_block=(2, 2, 2)),
    },
    "wb_vanilla": {
        "cls": "WeatherDCAEAdaLNModel",
        "kwargs": dict(in_channels=24, out_channels=24, n_static_features=3,
                       latent_channels=32, attention_head_dim=32,
                       block_type=("ResBlock", "ResBlock", "EfficientViTBlock"),
                       qkv_multiscales=((), (), (5,)), lat_crop=-8,
                       block_out_channels=(128, 128, 256, 256),
                       layers_per_block=(2, 2, 2)),
    },
    "dcae_14m": {
        "cls": "WeatherDCAEAdaLNModel",
        "kwargs": dict(
            in_channels=24,
            out_channels=24,
            n_static_features=3,
            latent_channels=256,
            attention_head_dim=32,
            block_type=("ResBlock", "ResBlock", "EfficientViTBlock"),
            qkv_multiscales=((), (), (5,)),
            lat_crop=-8,
            block_out_channels=(64, 128, 256),
            layers_per_block=(3, 3, 3),
        ),
    },
    "wb_skip": {
        "cls": "WeatherDCAEAdaLNSkipModel",
        "kwargs": dict(
            in_channels=24,
            out_channels=24,
            n_static_features=3,
            latent_channels=256,
            attention_head_dim=32,
            block_type=("ResBlock", "ResBlock", "EfficientViTBlock"),
            qkv_multiscales=((), (), (5,)),
            lat_crop=-8,
            block_out_channels=(64, 128, 256),
            layers_per_block=(3, 3, 3),
            skip_lateral_rank=0,
            tau_conditional_gates=False,
        ),
    },
    "atmvfi": {
        "cls": "PixelAttentionVFI",
        "kwargs": dict(in_channels=24, hidden=72, n_levels=3),
    },
    "amt": {
        "cls": "WeatherAMTModel",
        "kwargs": dict(
            in_channels=24,
            out_channels=24,
            n_static_features=3,
            corr_radius=3,
            corr_levels=4,
            num_flows=5,
            channels=(48, 64, 72, 110),
            skip_channels=48,
            endpoint_envelope=True,
            max_field_displacement=16.0,
        ),
    },
    "amt_residual": {
        "cls": "WeatherAMTResidualModel",
        "kwargs": dict(
            in_channels=24,
            out_channels=24,
            n_static_features=3,
            corr_radius=3,
            corr_levels=4,
            num_flows=5,
            channels=(48, 64, 72, 110),
            skip_channels=48,
            max_field_displacement=16.0,
        ),
    },
}


def _flow_meta(arch: str):
    """Mirror train_capacity_matched_6h.build_net for Flow ablation arms."""
    return {
        "cls": "WeatherBridgeModel",
        "kwargs": dict(
            in_channels=24,
            out_channels=24,
            n_static_features=3,
            hidden=(
                64
                if arch in {
                    "flow_dual",
                    "flow_universal_latent",
                    "flow_universal_latent_refine",
                    "flow_universal_fm",
                }
                else 72
            ),
            n_levels=3,
            use_skip=(arch != "flow_noskip"),
            gated_skip=(arch != "flow_ungated"),
            use_accel=arch in {
                "flow_accel",
                "flow_pp",
                "flow_pp2",
                "flow_pp3",
                "flow_pp3_spherical",
                "flow_pp3_multiband",
                "flow_pp3_detail",
                "flow_pp3_detail_fm",
                "flow_universal_detail",
                "flow_universal_latent",
                "flow_universal_latent_refine",
                "flow_universal_fm",
                "flow_dual",
                "flow_spherical_ep",
            },
            strong_residual=arch in {"flow_pp", "flow_pp2"},
            mass_aware_gate=(arch == "flow_pp2"),
            spectral_branch=arch in {
                "flow_pp3",
                "flow_pp3_spherical",
                "flow_pp3_multiband",
                "flow_pp3_detail",
                "flow_pp3_detail_fm",
                "flow_universal_detail",
                "flow_universal_latent",
                "flow_universal_latent_refine",
                "flow_universal_fm",
                "flow_spherical_ep",
            },
            hydro_couple=arch in {
                "flow_pp3",
                "flow_pp3_spherical",
                "flow_pp3_multiband",
                "flow_pp3_detail",
                "flow_pp3_detail_fm",
                "flow_spherical_ep",
            },
            dual_stream=(arch == "flow_dual"),
            spherical_ops=arch in {
                "flow_pp3_spherical",
                "flow_spherical_ep",
            },
            endpoint_preserving=arch in {
                "flow_spherical_ep",
                "flow_pp3_detail_fm",
                "flow_universal_detail",
                "flow_universal_latent",
                "flow_universal_latent_refine",
                "flow_universal_fm",
            },
            query_independent_trajectory=(
                arch == "flow_spherical_ep"
            ),
            multiband_calibration=(arch == "flow_pp3_multiband"),
            anchor_detail_bypass=arch in {
                "flow_pp3_detail",
                "flow_pp3_detail_fm",
                "flow_universal_detail",
                "flow_universal_latent",
                "flow_universal_latent_refine",
                "flow_universal_fm",
            },
            flow_matching=arch in {
                "flow_pp3_detail_fm",
                "flow_universal_fm",
            },
            flow_matching_steps=4,
            shared_field_controls=arch in {
                "flow_universal_detail",
                "flow_universal_latent",
                "flow_universal_latent_refine",
                "flow_universal_fm",
            },
            latent_transformer_tokens=(
                16
                if arch in {
                    "flow_universal_latent",
                    "flow_universal_latent_refine",
                    "flow_universal_fm",
                }
                else 0
            ),
            latent_transformer_dim=128,
            latent_transformer_depth=2,
            latent_transformer_heads=4,
            decoder_blocks_per_level=(
                2 if arch == "flow_universal_latent_refine" else 1
            ),
        ),
    }


for _arch in (
    "flow",
    "flow_noskip",
    "flow_ungated",
    "flow_accel",
    "flow_pp",
    "flow_pp2",
    "flow_pp3",
    "flow_pp3_spherical",
    "flow_pp3_multiband",
    "flow_pp3_detail",
    "flow_pp3_detail_fm",
    "flow_universal_detail",
    "flow_universal_latent",
    "flow_universal_latent_refine",
    "flow_universal_fm",
    "flow_dual",
    "flow_spherical_ep",
):
    ARCH_META[_arch] = _flow_meta(_arch)


for _arch in UPR_LITE_VARIANTS:
    ARCH_META[_arch] = {
        "cls": "WeatherBridgeUPRLiteModel",
        "kwargs": upr_lite_variant_kwargs(_arch),
    }

for _arch in UPR_SCALED_VARIANTS:
    ARCH_META[_arch] = {
        "cls": "WeatherBridgeUPRLiteModel",
        "kwargs": upr_scaled_variant_kwargs(_arch),
    }

ARCH_META["upr_spherical_implicit_global_14m"] = {
    "cls": "WeatherBridgeUPRSphericalModel",
    "kwargs": upr_scaled_variant_kwargs("upr_implicit_global_14m"),
}


ARCH_ALIASES = {
    "weatherbridge": "flow_pp3",
    "weatherbridge_detail": "flow_pp3_detail",
    "weatherbridge_flow_spectral": "flow_pp3",
    "weatherbridge_fm_detail": "flow_pp3_detail_fm",
    "weatherbridge_universal_detail": "flow_universal_detail",
    "weatherbridge_universal_latent": "flow_universal_latent",
    "weatherbridge_universal_latent_refine": "flow_universal_latent_refine",
    "weatherbridge_universal_pyramid": "upr_universal_latent_q4_10m",
    "weatherbridge_universal_fm": "flow_universal_fm",
    "weatherdcae": "dcae_14m",
    "pixelattn_vfi": "atmvfi",
    "weatheramt": "amt",
    "weatheramt_residual": "amt_residual",
}


def resolve_bare_arch(arch: str) -> str:
    """Resolve public paper names to immutable internal architectures."""
    return ARCH_ALIASES.get(arch, arch)


def checkpoint_arch_meta(arch: str, state: dict[str, torch.Tensor]):
    """Return exact bare metadata, including historical matched Skip shapes."""
    if (
        arch == "wb_skip"
        and "encoder.conv_in.weight" in state
        and state["encoder.conv_in.weight"].shape[0] == 128
    ):
        return {
            "cls": "WeatherDCAEAdaLNSkipModel",
            "kwargs": dict(
                in_channels=24,
                out_channels=24,
                n_static_features=3,
                latent_channels=32,
                attention_head_dim=32,
                block_type=("ResBlock", "ResBlock", "EfficientViTBlock"),
                qkv_multiscales=((), (), (5,)),
                lat_crop=-8,
                block_out_channels=(128, 128, 256, 256),
                layers_per_block=(2, 2, 2),
                skip_lateral_rank=0,
                tau_conditional_gates=False,
            ),
        }
    return ARCH_META[arch]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument(
        "--arch",
        required=True,
        choices=sorted([*ARCH_META, *ARCH_ALIASES]),
    )
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    internal_arch = resolve_bare_arch(args.arch)

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    sd = ck["state_dict"] if "state_dict" in ck else ck
    # Strip the "net." prefix from CapMatchedLit → backbone state_dict.
    net_sd = {}
    for k, v in sd.items():
        if k.startswith("net."):
            net_sd[k[len("net."):]] = v
    net_sd = normalize_weatherbridge_decoder_state_dict(net_sd)
    meta = checkpoint_arch_meta(internal_arch, net_sd)
    blob = {"arch": meta["cls"], "kwargs": meta["kwargs"], "state_dict": net_sd}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(blob, args.out)
    print(
        f"[bare] {args.arch} (internal={internal_arch}) -> "
        f"{args.out}  ({len(net_sd)} tensors)"
    )


if __name__ == "__main__":
    main()
