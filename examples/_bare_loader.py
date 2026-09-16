"""Tiny universal bare-blob loader.

A bare blob is the artefact ``tools/ckpt_normalize.py`` writes alongside
the Lightning-format ckpt: ``{arch, kwargs, state_dict}``. This module
imports the correct backbone class by name and returns a fully loaded
``nn.Module`` in two lines of user code:

    from examples._bare_loader import load_bare
    model = load_bare(
        "weights/weatherbridge_14m_6h_bare.pt"
    )
"""
from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

# Bare-blob ``arch`` field → bare backbone class (no Lightning).
#
# ``WeatherBridgeModel`` is the model the paper calls WeatherBridge, trained
# under the experiment flag ``flow_pp3`` and formerly written as
# "Flow-Spectral". It is the only entry here that carries that name. The
# ``WeatherBridge*`` prefixes on the Mamba and UPR entries below are historical
# labels for rejected exploratory branches; they are not WeatherBridge, and the
# ``arch`` string is kept verbatim only because it is what their saved blobs
# record.
_ARCH_REGISTRY = {
    "WeatherDCAEAdaLNModel":
        ("weather_time_interp.model.dcae_adaln_model", "WeatherDCAEAdaLNModel"),
    "WeatherDCAEAdaLNSkipModel":
        ("weather_time_interp.model.dcae_adaln_skip_model", "WeatherDCAEAdaLNSkipModel"),
    "WeatherFuXiSwinV2Model":
        ("weather_time_interp.model.fuxi_swinv2_model", "WeatherFuXiSwinV2Model"),
    "WeatherSDyffusionDYffusionModel":
        ("weather_time_interp.model.sdyff_dyffusion_model", "WeatherSDyffusionDYffusionModel"),
    "WeatherModAFNOResidualLinearModel":
        ("weather_time_interp.model.modafno_baseline_model", "WeatherModAFNOResidualLinearModel"),
    "WeatherDCAECrossFrameModel":
        ("weather_time_interp.model.weatherbridge_crossframe_model", "WeatherDCAECrossFrameModel"),
    "WeatherBridgeMambaModel":
        ("weather_time_interp.model.weatherbridge_mamba_model", "WeatherBridgeMambaModel"),
    "WeatherBridgeModel":
        ("weather_time_interp.model.weatherbridge_flow_model", "WeatherBridgeModel"),
    # Same class: weatherbridge_flow_model.py keeps
    # ``WeatherBridgeFlowModel = WeatherBridgeModel`` so blobs saved under the
    # older class name still load. Do not drop this entry.
    "WeatherBridgeFlowModel":
        ("weather_time_interp.model.weatherbridge_flow_model", "WeatherBridgeFlowModel"),
    "WeatherBridgeUPRLiteModel":
        ("weather_time_interp.model.weatherbridge_upr_lite_model", "WeatherBridgeUPRLiteModel"),
    "WeatherBridgeUPRSphericalModel":
        ("weather_time_interp.model.weatherbridge_upr_spherical_model", "WeatherBridgeUPRSphericalModel"),
    "WeatherAMTModel":
        ("weather_time_interp.model.weather_amt_model", "WeatherAMTModel"),
    "WeatherAMTResidualModel":
        (
            "weather_time_interp.model.weather_amt_residual_model",
            "WeatherAMTResidualModel",
        ),
    "PixelAttentionVFI":
        ("train_atm_vfi_12h_oddskip", "PixelAttentionVFI"),
}


def load_bare(blob_path: str | Path,
              device: str | torch.device = "cpu") -> nn.Module:
    blob = torch.load(str(blob_path), map_location=device, weights_only=False)

    # ATM-VFI v2 has an asymmetric I/O ckpt (encoder sees [x; static] →
    # 27ch, residual stream stays 24ch) that needs the production
    # loader's monkey-patching to reattach the static prefuse inside
    # ``net.forward``. Defer to it directly — the rest of the bare path
    # cannot reproduce that constructor rewrite.
    if blob["arch"] == "PixelAttentionVFI":
        from tools.eval.batch_eval_memmap import _load_atmvfi_model
        ckpt_dir = Path(blob_path).parent
        ckpt = ckpt_dir / Path(blob_path).name.replace("_bare.pt", ".ckpt")
        if not ckpt.exists():
            ckpt = blob_path
        model = _load_atmvfi_model(
            str(ckpt), blob["state_dict"], blob["kwargs"], device,
        )
        return model.eval()

    mod_path, cls_name = _ARCH_REGISTRY[blob["arch"]]
    import importlib
    cls = getattr(importlib.import_module(mod_path), cls_name)
    model = cls(**blob["kwargs"])
    state_dict = blob["state_dict"]
    if blob["arch"] == "WeatherBridgeModel":
        from tools.eval.capmatched_loader import (
            normalize_weatherbridge_decoder_state_dict,
        )

        state_dict = normalize_weatherbridge_decoder_state_dict(state_dict)
    model.load_state_dict(state_dict)
    return model.eval().to(device)
