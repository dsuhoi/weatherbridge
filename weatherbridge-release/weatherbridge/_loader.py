"""Load a released checkpoint into an ``nn.Module``, with no training stack.

Every released checkpoint is a *bare blob*: ``{arch, kwargs, state_dict}``.
The architecture is rebuilt from its own recorded kwargs, so nothing here
sniffs shapes or guesses hyper-parameters.

Two checkpoints need a documented fix-up, both applied below rather than
silently:

* ``WeatherBridgeModel`` blobs predating the decoder wrapper store keys as
  ``dec.<level>.<module>.<suffix>``; the current layout inserts a block
  index. :func:`normalize_decoder_state_dict` maps one onto the other.
* ``PixelAttentionVFI``'s released 6 h checkpoint is asymmetric: its encoder
  was trained on ``[x; static]`` (27 channels) while the residual it predicts
  covers only the 24 prognostic channels. :func:`_apply_static_prefuse`
  rebuilds the output head at the smaller width and re-attaches the static
  concatenation inside ``forward``.
"""
from __future__ import annotations

import importlib
import json
import re
import types
from pathlib import Path

import torch
from torch import nn

_PACKAGE_ROOT = Path(__file__).resolve().parent
_REPO_ROOT = _PACKAGE_ROOT.parent
WEIGHTS_DIR = _REPO_ROOT / "weights"
DATA_DIR = _REPO_ROOT / "data"

_LEGACY_DECODER_KEY = re.compile(r"^dec\.(\d+)\.(\d+)\.(.+)$")
_CURRENT_DECODER_KEY = re.compile(r"^dec\.\d+\.\d+\.\d+\.")

# ``arch`` recorded in the blob -> module and class inside weatherbridge.models
_ARCH_REGISTRY: dict[str, tuple[str, str]] = {
    # WeatherBridge is exactly this class, trained under the experiment flag
    # ``flow_pp3``. WeatherBridgeFlowModel is an alias kept so that blobs
    # saved under the older class name still load.
    "WeatherBridgeModel": ("weatherbridge_flow_model", "WeatherBridgeModel"),
    "WeatherBridgeFlowModel": ("weatherbridge_flow_model", "WeatherBridgeModel"),
    "WeatherDCAEAdaLNModel": ("dcae_adaln_model", "WeatherDCAEAdaLNModel"),
    "WeatherFuXiSwinV2Model": ("fuxi_swinv2_model", "WeatherFuXiSwinV2Model"),
    "WeatherModAFNOResidualLinearModel": (
        "modafno_baseline_model",
        "WeatherModAFNOResidualLinearModel",
    ),
    "WeatherSDyffusionDYffusionModel": (
        "sdyff_dyffusion_model",
        "WeatherSDyffusionDYffusionModel",
    ),
    "PixelAttentionVFI": ("pixel_attention_vfi", "PixelAttentionVFINet"),
}


def _catalogue() -> dict:
    with (WEIGHTS_DIR / "catalogue.json").open() as handle:
        return json.load(handle)


def list_models() -> list[str]:
    """Names accepted by :func:`load_model`, in paper-table order."""
    return list(_catalogue()["models"])


def model_info(name: str) -> dict:
    models = _catalogue()["models"]
    if name not in models:
        raise KeyError(
            f"unknown model {name!r}; available: {', '.join(models)}"
        )
    return models[name]


def normalize_decoder_state_dict(
    state: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Map pre-wrapper single-block decoder keys onto the current layout."""
    decoder_keys = [key for key in state if key.startswith("dec.")]
    if not decoder_keys:
        return state
    current = [k for k in decoder_keys if _CURRENT_DECODER_KEY.match(k)]
    if len(current) == len(decoder_keys):
        return state
    if current:
        raise ValueError("mixed or unrecognised decoder state layout")

    matches = {k: _LEGACY_DECODER_KEY.match(k) for k in decoder_keys}
    if not all(matches.values()):
        raise ValueError("mixed or unrecognised decoder state layout")

    out: dict[str, torch.Tensor] = {}
    for key, value in state.items():
        match = matches.get(key)
        if match is None:
            out[key] = value
            continue
        level, module, suffix = match.groups()
        out[f"dec.{level}.0.{module}.{suffix}"] = value
    return out


def load_static_features(n_features: int, device="cpu") -> torch.Tensor:
    """Return ``(1, n_features, 360, 720)`` land mask, orography and latitude."""
    path = DATA_DIR / "static_features_0p5.pt"
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} is missing; it ships with the release archive and every "
            "model needs it"
        )
    static = torch.load(str(path), map_location="cpu", weights_only=False).float()
    if static.dim() == 3:
        static = static.unsqueeze(0)
    return static[:, :n_features].to(device)


def _apply_static_prefuse(net: nn.Module, state: dict, device) -> nn.Module:
    """Rebuild PixelAttn-VFI's head and re-attach the static concatenation."""
    first = state.get("frame_encoder.0.0.weight")
    out_w = state.get("out_conv.weight")
    if first is None or out_w is None:
        return net
    n_in, n_out = int(first.shape[1]), int(out_w.shape[0])
    if n_in == n_out:
        return net

    n_static = n_in - n_out
    hidden = net.out_conv.weight.shape[1]
    net.out_conv = nn.Conv2d(hidden, n_out, kernel_size=3, padding=1)
    nn.init.zeros_(net.out_conv.weight)
    nn.init.zeros_(net.out_conv.bias)
    net.scale = nn.Parameter(torch.full((n_out,), 0.1))
    net.register_buffer(
        "eval_static", load_static_features(n_static).cpu(), persistent=False
    )

    def forward(self, x_0, x_T, tau):
        tau_b = tau.view(-1, 1, 1, 1) if tau.dim() <= 2 else tau
        x_linear = (1.0 - tau_b) * x_0 + tau_b * x_T
        batch, _, height, width = x_0.shape
        static = self.eval_static
        if static.shape[-2:] != (height, width):
            static = torch.nn.functional.interpolate(
                static, size=(height, width), mode="bilinear", align_corners=False
            )
        static = static.expand(batch, -1, height, width)
        f0 = self.encode_frame(torch.cat([x_0, static], dim=1))
        fT = self.encode_frame(torch.cat([x_T, static], dim=1))
        h = self.attn(f0[-1], fT[-1])
        h = self.adaln(h, tau.view(-1))
        for i in range(len(self.decoder) // 2):
            h = self.decoder[2 * i](h)
            skip = (f0[-(i + 2)] + fT[-(i + 2)]) / 2
            if h.shape[-2:] != skip.shape[-2:]:
                h = torch.nn.functional.interpolate(
                    h, size=skip.shape[-2:], mode="bilinear", align_corners=False
                )
            h = self.decoder[2 * i + 1](torch.cat([h, skip], dim=1))
        if h.shape[-2:] != x_0.shape[-2:]:
            h = torch.nn.functional.interpolate(
                h, size=x_0.shape[-2:], mode="bilinear", align_corners=False
            )
        delta = self.out_conv(h)
        return x_linear + torch.tanh(self.scale).view(1, -1, 1, 1) * delta

    net.forward = types.MethodType(forward, net)
    return net


def load_model(name: str, device: str | torch.device = "cpu") -> nn.Module:
    """Load a released model by catalogue name, ready for ``eval()`` inference."""
    info = model_info(name)
    blob_path = WEIGHTS_DIR / info["weights"]
    if not blob_path.is_file():
        raise FileNotFoundError(
            f"{blob_path} is missing; unpack the full release archive"
        )
    blob = torch.load(str(blob_path), map_location="cpu", weights_only=False)

    module_name, class_name = _ARCH_REGISTRY[blob["arch"]]
    module = importlib.import_module(f"weatherbridge.models.{module_name}")
    cls = getattr(module, class_name)
    state = blob["state_dict"]

    if blob["arch"] == "PixelAttentionVFI":
        # The blob was saved from the Lightning wrapper: its backbone sits at
        # ``net`` and its kwargs describe the wrapper, not the backbone. This
        # package instantiates the backbone directly, so the geometry comes
        # from the tensors themselves, which are authoritative.
        state = {
            key[len("net."):]: value
            for key, value in state.items()
            if key.startswith("net.")
        }
        first = state["frame_encoder.0.0.weight"]
        net = cls(
            in_ch=int(first.shape[1]),
            hidden=int(first.shape[0]),
            n_levels=sum(
                1 for k in state if re.fullmatch(r"decoder\.\d+\.weight", k)
            ),
            time_dim=int(state["adaln.time_mlp.0.weight"].shape[1]),
        )
        net = _apply_static_prefuse(net, state, device)
    else:
        net = cls(**blob["kwargs"])
        if blob["arch"] in ("WeatherBridgeModel", "WeatherBridgeFlowModel"):
            state = normalize_decoder_state_dict(state)

    net.load_state_dict(state)
    # Tag the module so weatherbridge.predict knows which calling convention
    # this family uses without the caller having to remember.
    net._wb_family = info["family"]
    net._wb_delta_t = float(info["delta_t_hours"])
    net._wb_name = name
    return net.eval().to(device)
