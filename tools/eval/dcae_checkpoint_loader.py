"""Direct loader for WeatherDCAE Lightning checkpoints.

The paper checkpoints store the backbone below ``model.``. Loading the
backbone directly avoids importing the archived Lightning trainer and makes
the effective input/output width explicit.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


SUPPORTED_MODEL_TYPES = {"dcae_adaln_residual_linear"}


def _state_and_hparams(
    checkpoint: dict[str, Any],
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    state = checkpoint.get("state_dict", checkpoint)
    hparams = dict(checkpoint.get("hyper_parameters", {}))
    return state, hparams


def _infer_io_channels(
    state: dict[str, torch.Tensor],
    hparams: dict[str, Any],
) -> int:
    output_weight = state.get("model.decoder.conv_out.weight")
    if output_weight is not None:
        return int(output_weight.shape[0])
    n_pl = int(hparams.get("n_pl_channels", 20))
    n_surface = int(hparams.get("n_surface_channels", 4))
    return n_pl + n_surface


def _build_kwargs(
    state: dict[str, torch.Tensor],
    hparams: dict[str, Any],
) -> dict[str, Any]:
    channels = _infer_io_channels(state, hparams)
    block_out = tuple(hparams.get("block_out_channels") or (128, 256, 512))
    layers = tuple(hparams.get("layers_per_block") or (2, 2, 2))
    n_stages = min(len(block_out), len(layers))
    block_type = ("ResBlock",) * (n_stages - 1) + ("EfficientViTBlock",)
    qkv_multiscales = ((),) * (n_stages - 1) + ((5,),)
    return {
        "latent_channels": int(hparams.get("latent_channels", 256)),
        "block_out_channels": block_out,
        "layers_per_block": layers,
        "block_type": block_type,
        "qkv_multiscales": qkv_multiscales,
        "attention_head_dim": int(hparams.get("attention_head_dim", 32)),
        "in_channels": channels,
        "out_channels": channels,
        "n_static_features": int(hparams.get("n_static_features", 0)),
        "lat_crop": int(hparams.get("lat_crop", 0)),
        "residual_scale_init": float(hparams.get("residual_scale_init", 0.1)),
        "residual_scale_learnable": bool(
            hparams.get("residual_scale_learnable", True)
        ),
        "residual_clip": hparams.get("residual_clip"),
        "residual_scale_floor": float(
            hparams.get("residual_scale_floor", 0.0)
        ),
        "direct_prediction": bool(hparams.get("direct_prediction", False)),
        "time_emb_dim": int(hparams.get("time_emb_dim", 256)),
        "time_freq_dim": int(hparams.get("time_freq_dim", 128)),
        "time_base_period": float(hparams.get("time_base_period", 16.0)),
    }


def load_dcae_checkpoint(
    checkpoint_path: str | Path,
    device: torch.device | str,
) -> tuple[torch.nn.Module, str]:
    checkpoint = torch.load(
        str(checkpoint_path),
        map_location=device,
        weights_only=False,
    )
    state, hparams = _state_and_hparams(checkpoint)
    model_type = str(hparams.get("model_type", ""))
    if model_type not in SUPPORTED_MODEL_TYPES:
        raise ValueError(
            f"{checkpoint_path}: unsupported direct WeatherDCAE type "
            f"{model_type!r}"
        )

    from weather_time_interp.model.dcae_adaln_model import (
        WeatherDCAEAdaLNModel,
    )

    kwargs = _build_kwargs(state, hparams)
    model = WeatherDCAEAdaLNModel(**kwargs)
    prefix = "model."
    model_state = {
        key.removeprefix(prefix): value
        for key, value in state.items()
        if key.startswith(prefix)
    }
    if not model_state:
        raise ValueError(f"{checkpoint_path}: no model.* state_dict entries")
    model.load_state_dict(model_state, strict=True)
    model.to(device).eval()
    return model, model_type


def dcae_bare_payload(checkpoint_path: str | Path) -> dict[str, Any]:
    checkpoint = torch.load(
        str(checkpoint_path),
        map_location="cpu",
        weights_only=False,
    )
    state, hparams = _state_and_hparams(checkpoint)
    model_type = str(hparams.get("model_type", ""))
    if model_type not in SUPPORTED_MODEL_TYPES:
        raise ValueError(
            f"{checkpoint_path}: unsupported WeatherDCAE type {model_type!r}"
        )
    prefix = "model."
    model_state = {
        key.removeprefix(prefix): value
        for key, value in state.items()
        if key.startswith(prefix)
    }
    return {
        "arch": "WeatherDCAEAdaLNModel",
        "kwargs": _build_kwargs(state, hparams),
        "state_dict": model_state,
    }
