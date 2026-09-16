"""Inference loader for checkpoints produced by the capacity-matched trainer."""
from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path

import torch
from torch import nn

_LEGACY_DECODER_KEY = re.compile(r"^dec\.(\d+)\.(\d+)\.(.+)$")
_CURRENT_DECODER_KEY = re.compile(r"^dec\.\d+\.\d+\.\d+\.")
_HRES_LEAD_ARCHES = {
    "flow_pp3_hres_aug",
    "flow_pp3_hres_residual",
    # The degradation-aware arch is lead-conditioned too: without the real
    # anchor lead its reliability branch is evaluated at a constant lead and
    # the anchor-trust gate is mis-set.
    "flow_pp3_degrade",
}


def _environment_flag(name: str) -> bool:
    value = os.environ.get(name, "0")
    if value not in {"0", "1"}:
        raise ValueError(f"{name} must be 0 or 1")
    return value == "1"


def _configure_inference_ablations(net: nn.Module) -> None:
    acceleration = _environment_flag("WEATHERBRIDGE_ABLATE_ACCELERATION")
    transport = _environment_flag("WEATHERBRIDGE_ABLATE_TRANSPORT")
    hydrostatic = _environment_flag("WEATHERBRIDGE_ABLATE_HYDROSTATIC")
    if acceleration and not hasattr(net, "ablate_acceleration"):
        raise ValueError("acceleration lesion requires WeatherBridge")
    if transport and not hasattr(net, "ablate_transport"):
        raise ValueError("transport lesion requires WeatherBridge")
    if hydrostatic and not hasattr(net, "ablate_hydrostatic"):
        raise ValueError("hydrostatic lesion requires WeatherBridge")
    if hasattr(net, "ablate_acceleration"):
        net.ablate_acceleration = acceleration
    if hasattr(net, "ablate_transport"):
        net.ablate_transport = transport
    if hasattr(net, "ablate_hydrostatic"):
        net.ablate_hydrostatic = hydrostatic


def normalize_weatherbridge_decoder_state_dict(
    state: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Map pre-wrapper single-block decoder keys to the current layout."""
    decoder_keys = [key for key in state if key.startswith("dec.")]
    if not decoder_keys:
        return state
    current_keys = [
        key for key in decoder_keys if _CURRENT_DECODER_KEY.match(key)
    ]
    if len(current_keys) == len(decoder_keys):
        return state
    if current_keys:
        raise ValueError("mixed or unrecognised WeatherBridge decoder state layout")

    legacy_matches = {
        key: _LEGACY_DECODER_KEY.match(key)
        for key in decoder_keys
    }
    if not all(legacy_matches.values()):
        raise ValueError(
            "mixed or unrecognised WeatherBridge decoder state layout"
        )

    normalized: dict[str, torch.Tensor] = {}
    for key, value in state.items():
        match = legacy_matches.get(key)
        if match is None:
            normalized[key] = value
            continue
        level, module, suffix = match.groups()
        normalized[f"dec.{level}.0.{module}.{suffix}"] = value
    return normalized


def _build_net_for_checkpoint(
    arch: str,
    net_state: dict[str, torch.Tensor],
    static_path: str,
) -> nn.Module:
    """Build the recorded architecture, including the legacy matched Skip arm."""
    if arch in {
        "upr_query_match_14m",
        "upr_local_corr_14m",
    }:
        from weather_time_interp.model.weatherbridge_upr_lite_model import (
            WeatherBridgeUPRLiteModel,
        )
        from weather_time_interp.model.weatherbridge_upr_scaled_model import (
            upr_scaled_variant_kwargs,
        )

        return WeatherBridgeUPRLiteModel(
            **upr_scaled_variant_kwargs(arch)
        )

    if arch == "amt_residual":
        from weather_time_interp.model.weather_amt_residual_model import (
            WeatherAMTResidualModel,
        )

        return WeatherAMTResidualModel(
            in_channels=24,
            out_channels=24,
            n_static_features=3,
            corr_radius=3,
            corr_levels=4,
            num_flows=5,
            channels=(48, 64, 72, 110),
            skip_channels=48,
            max_field_displacement=16.0,
        )

    if (
        arch == "wb_skip"
        and "encoder.conv_in.weight" in net_state
        and net_state["encoder.conv_in.weight"].shape[0] == 128
    ):
        from weather_time_interp.model.dcae_adaln_skip_model import (
            WeatherDCAEAdaLNSkipModel,
        )

        return WeatherDCAEAdaLNSkipModel(
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
        )

    from tools.train.train_capacity_matched_6h import build_net

    net, _, _ = build_net(arch, static_path)
    return net


class CapMatchedInference(nn.Module):
    """Expose a paper-eval-compatible forward around a capacity-matched net."""

    def __init__(
        self,
        net: nn.Module,
        *,
        arch: str,
        delta_t: float,
        static: torch.Tensor,
    ) -> None:
        super().__init__()
        self.net = net
        self.arch = arch
        self.delta_t = float(delta_t)
        self.register_buffer(
            "static3",
            static[:3].float().unsqueeze(0),
            persistent=False,
        )

    def forward(
        self,
        x0: torch.Tensor,
        xT: torch.Tensor,
        tau_norm: torch.Tensor,
        cond: torch.Tensor | None = None,
        static: torch.Tensor | None = None,
    ):
        tau = tau_norm.reshape(-1)
        batch_size = x0.size(0)
        static_batch = static
        if static_batch is None:
            static_batch = self.static3
        if static_batch.dim() == 3:
            static_batch = static_batch.unsqueeze(0)
        if static_batch.size(0) != batch_size:
            static_batch = static_batch.expand(batch_size, -1, -1, -1)
        model_condition = (
            cond
            if self.arch in _HRES_LEAD_ARCHES
            else torch.full(
                (batch_size,),
                self.delta_t,
                device=x0.device,
                dtype=torch.float32,
            )
        )
        if self.arch in _HRES_LEAD_ARCHES and model_condition is None:
            raise ValueError("HRES-conditioned WeatherBridge requires forecast lead")
        return self.net(
            x0,
            xT,
            tau,
            model_condition,
            static=static_batch,
        )


class CapMatchedEnsemble(nn.Module):
    """Average independently trained capacity-matched members at inference."""

    def __init__(self, members: list[CapMatchedInference]) -> None:
        super().__init__()
        if len(members) < 2:
            raise ValueError("an ensemble requires at least two members")
        delta_t = {member.delta_t for member in members}
        if len(delta_t) != 1:
            raise ValueError("ensemble members must use the same delta_t")
        self.members = nn.ModuleList(members)
        self.delta_t = members[0].delta_t
        self.arch = "capmatched_output_ensemble"

    def forward(
        self,
        x0: torch.Tensor,
        xT: torch.Tensor,
        tau_norm: torch.Tensor,
        cond: torch.Tensor | None = None,
        static: torch.Tensor | None = None,
    ):
        total: torch.Tensor | None = None
        for member in self.members:
            output = member(x0, xT, tau_norm, cond, static)
            prediction = output[0] if isinstance(output, tuple) else output
            total = prediction if total is None else total + prediction
        if total is None:
            raise RuntimeError("ensemble has no members")
        mean = total / len(self.members)
        return mean, {"ensemble_size": mean.new_tensor(len(self.members))}


def load_capmatched_checkpoint(
    ckpt_path: str | Path,
    device: torch.device,
    *,
    static_path: str | Path,
) -> tuple[CapMatchedInference, str]:
    """Load a ``CapMatchedLit`` checkpoint without constructing a Trainer."""
    checkpoint_path = Path(ckpt_path)
    ckpt = torch.load(
        str(checkpoint_path),
        map_location=device,
        weights_only=False,
    )
    hparams = dict(ckpt.get("hyper_parameters", {}))
    arch = str(hparams.get("arch", ""))
    if not arch:
        raise ValueError(f"{checkpoint_path}: missing capacity-matched 'arch'")

    static_file = Path(static_path)
    state = ckpt.get("state_dict", ckpt)
    if arch == "temporal_expert_router":
        from weather_time_interp.model.temporal_expert_router import (
            TemporalExpertRouter,
        )

        expert_metadata = hparams.get("router_experts")
        route_metadata = hparams.get("router_route_by_tau")
        if (
            not isinstance(expert_metadata, dict)
            or not isinstance(route_metadata, dict)
        ):
            raise ValueError(
                f"{checkpoint_path}: missing temporal-router metadata"
            )
        experts: dict[str, nn.Module] = {}
        for name, metadata in expert_metadata.items():
            if (
                not isinstance(name, str)
                or not isinstance(metadata, dict)
                or not isinstance(metadata.get("arch"), str)
            ):
                raise TypeError(
                    f"{checkpoint_path}: invalid router expert metadata"
                )
            prefix = f"net.experts.{name}."
            expert_state = {
                key.removeprefix(prefix): value
                for key, value in state.items()
                if key.startswith(prefix)
            }
            if not expert_state:
                raise ValueError(
                    f"{checkpoint_path}: no state for router expert {name}"
                )
            expert_state = normalize_weatherbridge_decoder_state_dict(
                expert_state
            )
            expert = _build_net_for_checkpoint(
                metadata["arch"],
                expert_state,
                str(static_file),
            )
            missing, unexpected = expert.load_state_dict(
                expert_state,
                strict=False,
            )
            if missing or unexpected:
                raise RuntimeError(
                    f"{checkpoint_path}: incompatible router expert {name} "
                    f"(missing={missing[:5]}, unexpected={unexpected[:5]})"
                )
            experts[name] = expert
        net = TemporalExpertRouter(
            experts,
            {
                int(hour): str(name)
                for hour, name in route_metadata.items()
            },
            delta_t=float(hparams.get("delta_t", 6.0)),
        )
    else:
        net_state = {
            key.removeprefix("net."): value
            for key, value in state.items()
            if key.startswith("net.")
        }
        if not net_state:
            raise ValueError(f"{checkpoint_path}: no 'net.' parameters found")
        net_state = normalize_weatherbridge_decoder_state_dict(net_state)
        net = _build_net_for_checkpoint(arch, net_state, str(static_file))
        missing, unexpected = net.load_state_dict(net_state, strict=False)
        if missing or unexpected:
            raise RuntimeError(
                f"{checkpoint_path}: incompatible state "
                f"(missing={missing[:5]}, unexpected={unexpected[:5]})"
            )
        _configure_inference_ablations(net)

    static = torch.load(static_file, map_location="cpu", weights_only=False)
    model = CapMatchedInference(
        net,
        arch=arch,
        delta_t=float(hparams.get("delta_t", 6.0)),
        static=static,
    )
    model.to(device).eval()
    if arch == "flow_pp3_detail":
        model_type = "weatherbridge_detail"
    elif arch == "flow_pp3":
        model_type = "weatherbridge"
    else:
        model_type = f"capmatched_{arch}"
    return model, model_type


def load_capmatched_ensemble_manifest(
    manifest_path: str | Path,
    device: torch.device,
    *,
    static_path: str | Path,
) -> tuple[CapMatchedEnsemble, str]:
    """Load and verify a mean-output ensemble manifest."""
    path = Path(manifest_path)
    manifest = json.loads(path.read_text())
    if manifest.get("schema_version") != 1:
        raise ValueError(f"{path}: unsupported ensemble schema")
    if manifest.get("type") != "capmatched_output_ensemble":
        raise ValueError(f"{path}: unsupported ensemble type")
    if manifest.get("reduction") != "mean":
        raise ValueError(f"{path}: only mean reduction is supported")
    entries = manifest.get("members")
    if not isinstance(entries, list) or len(entries) < 2:
        raise ValueError(f"{path}: at least two members are required")

    members: list[CapMatchedInference] = []
    seen: set[Path] = set()
    for entry in entries:
        if not isinstance(entry, dict):
            raise TypeError(f"{path}: invalid ensemble member")
        checkpoint_path = Path(entry["checkpoint"]).expanduser().resolve()
        if checkpoint_path in seen:
            raise ValueError(f"{path}: duplicate member {checkpoint_path}")
        seen.add(checkpoint_path)
        actual_sha256 = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
        if actual_sha256 != entry.get("sha256"):
            raise ValueError(f"{path}: checksum mismatch for {checkpoint_path}")
        member, _ = load_capmatched_checkpoint(
            checkpoint_path,
            device,
            static_path=static_path,
        )
        members.append(member)
    model = CapMatchedEnsemble(members).to(device).eval()
    return model, "capmatched_output_ensemble"
