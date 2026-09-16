#!/usr/bin/env python3
"""Fix the ``hyper_parameters`` block inside a legacy Lightning ckpt so
downstream loaders can read it through the standard
``WTIModelModule.load_from_checkpoint`` path with NO
sniffing, hp filtering, or shape patching.

What it does
------------
1. Load the ckpt once via the production ``load_model_safe`` (the only
   place that sniffs).
2. Read the *real* backbone (``lightning_module.model``) and its
   ``__init__`` parameter list.
3. Overwrite ``ckpt["hyper_parameters"]`` so every kwarg the trainer
   would now reject (``freeze_skip_gates``, etc.) is dropped, and the
   architectural ones (``block_out_channels`` / ``layers_per_block`` /
   ``latent_channels`` / ...) match what the backbone really is.
4. Save back as a normal Lightning ckpt: ``{state_dict, hyper_parameters,
   ...rest}``. The state_dict is left untouched.

After this fix, ``WTIModelModule.load_from_checkpoint(
fixed_path)`` Just Works™ — no sniff, no monkey-patching.
"""
from __future__ import annotations

import argparse
import copy
import inspect
import sys
from pathlib import Path

import torch
import torch.nn as nn


def _is_leaf(value) -> bool:
    if value is None:
        return True
    if isinstance(value, (bool, int, float, str)):
        return True
    if isinstance(value, (tuple, list)):
        return all(_is_leaf(v) for v in value)
    if isinstance(value, dict):
        return all(_is_leaf(v) for v in value.values())
    if isinstance(value, torch.Tensor) and value.numel() <= 1:
        return True
    return False


def _coerce(value):
    if isinstance(value, torch.Tensor) and value.numel() == 1:
        return float(value.item())
    if isinstance(value, tuple):
        return tuple(_coerce(v) for v in value)
    if isinstance(value, list):
        return [_coerce(v) for v in value]
    return value


def _find_attr(module: nn.Module, name: str):
    """Search ``name`` on ``module`` and its immediate submodules.

    Many backbones don't store init kwargs as ``self.<name>`` — they get
    handed straight to a sub-block (``self.encoder``, ``self.decoder``)
    that does keep them as attributes. This walks one level deep so the
    introspect step still recovers the truth without arch-specific code.
    """
    if hasattr(module, name):
        v = getattr(module, name)
        if not isinstance(v, nn.Module) and _is_leaf(v):
            return v
    for child_name in ("encoder", "decoder", "net", "modafno", "interpolator"):
        child = getattr(module, child_name, None)
        if child is None or not isinstance(child, nn.Module):
            continue
        if hasattr(child, name):
            v = getattr(child, name)
            if not isinstance(v, nn.Module) and _is_leaf(v):
                return v
    return None


def introspect_backbone_kwargs(backbone: nn.Module) -> dict:
    """Match the backbone's __init__ parameter names against its
    attributes (and immediate submodules) to recover the kwargs that
    built it. Skips nn.Module attrs (those are submodules)."""
    sig = inspect.signature(type(backbone).__init__)
    out: dict = {}
    for name, _ in sig.parameters.items():
        if name == "self":
            continue
        v = _find_attr(backbone, name)
        if v is not None:
            out[name] = _coerce(v)
    return out


def fix(src: str, dst: str) -> dict:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from tools.eval.batch_eval_memmap import load_model_safe

    src_p = Path(src).expanduser().resolve()
    dst_p = Path(dst).expanduser().resolve()

    # Production loader: this is the one place that runs sniff. Once.
    lightning_module, mt = load_model_safe(str(src_p), "cpu", None)

    # Re-read the raw ckpt — we want to preserve every other field
    # (optimizer states, scheduler, epoch, etc.) exactly as saved.
    raw = torch.load(str(src_p), map_location="cpu", weights_only=False)
    hp_in = dict(raw.get("hyper_parameters", {}))

    if mt == "atm_vfi_pixel_attn":
        # ATM-VFI lives in train_atm_vfi_12h_oddskip.py and is loaded
        # by a dedicated path; the production loader's PixelAttentionVFI
        # constructor already accepts only sane kwargs, so we just drop
        # the trainer-internal junk and trust the state_dict.
        backbone = lightning_module
        backbone_kwargs = introspect_backbone_kwargs(backbone)
    else:
        backbone = lightning_module.model
        backbone_kwargs = introspect_backbone_kwargs(backbone)

    # New hp = take only the keys the LightningModule.__init__ actually
    # accepts (drops freeze_skip_gates / freq_cond_film / etc.), then
    # overlay the backbone-side architecture truth on top.
    from trainer import WTIModelModule
    accepted = set(
        inspect.signature(WTIModelModule.__init__).parameters.keys()
    ) - {"self"}
    hp_new = {k: v for k, v in hp_in.items() if k in accepted}

    # Map backbone kwargs into the LightningModule hp slot. Most names
    # round-trip directly (``block_out_channels``, ``layers_per_block``,
    # ``latent_channels``, ``residual_scale_init``, ...); a few have
    # different aliases (FuXi / S-DYff / ModAFNO have model-specific
    # prefixes).
    backbone_to_lm = {
        # DC-AE family + WB-Hybrid
        "block_out_channels": "block_out_channels",
        "layers_per_block":   "layers_per_block",
        "block_type":         "block_type",
        "qkv_multiscales":    "qkv_multiscales",
        "latent_channels":    "latent_channels",
        "hidden_channels":    "hidden_channels",
        "cond_mlp_dim":       "cond_mlp_dim",
        "lat_crop":           "lat_crop",
        "n_static_features":  "n_static_features",
        "residual_scale_init":     "residual_scale_init",
        "residual_scale_learnable":"residual_scale_learnable",
        "residual_clip":           "residual_clip",
        "residual_scale_floor":    "residual_scale_floor",
        "direct_prediction":       "direct_prediction",
        "skip_gate_init":          "skip_gate_init",
        "skip_lateral_rank":       "skip_lateral_rank",
        "tau_conditional_gates":   "tau_conditional_gates",
        # FuXi
        "embed_dim":  None,   # depends on family; handled below
        "depth":      None,
        "num_heads":  "fuxi_num_heads",
        # S-DYff
        "num_layers":        "sdyff_num_layers",
        "n_modes_lat":       "sdyff_n_modes_lat",
        "n_modes_lon":       "sdyff_n_modes_lon",
        "n_inference_steps": "sdyff_inference_steps",
        "n_train_refine_steps":"sdyff_train_refine_steps",
        # ModAFNO
        "num_blocks":  "modafno_num_blocks",
        "drop_rate":   "modafno_drop_rate",
    }
    # Family-specific: depth / embed_dim are reused across FuXi, ModAFNO
    if mt == "fuxi_swinv2_residual_linear":
        backbone_to_lm["depth"] = "fuxi_depth"
        backbone_to_lm["embed_dim"] = "latent_channels"
    elif mt == "modafno_residual_linear":
        backbone_to_lm["depth"] = "modafno_depth"
        backbone_to_lm["embed_dim"] = "latent_channels"
    elif mt == "sdyff_dyffusion_residual_linear":
        backbone_to_lm["embed_dim"] = "latent_channels"

    for bk, lk in backbone_to_lm.items():
        if lk is None or bk not in backbone_kwargs:
            continue
        hp_new[lk] = backbone_kwargs[bk]

    # Preserve the original model_type label.
    if "model_type" in hp_in:
        hp_new["model_type"] = hp_in["model_type"]

    raw["hyper_parameters"] = hp_new

    # Also emit a flat "bare" blob next to it: {arch, kwargs, state_dict}
    # whose state_dict already has the ``model.`` prefix stripped and
    # whose kwargs are exactly what the backbone __init__ wants. That
    # makes downstream loading a strict two-liner with NO Lightning::
    #
    #     ckpt = torch.load(path, weights_only=False)
    #     model = <ClassFromCkpt>(**ckpt["kwargs"]); model.load_state_dict(ckpt["state_dict"])
    #
    bare_kwargs = introspect_backbone_kwargs(backbone)
    bare_sd = dict(backbone.state_dict())
    if mt == "atm_vfi_pixel_attn":
        bare_kwargs.pop("out_channels", None)
        # PixelAttentionVFI doesn't keep ``in_channels`` as an attribute;
        # sniff it from the first encoder conv's input dim instead.
        first_w = bare_sd.get("net.frame_encoder.0.0.weight")
        if first_w is not None and first_w.dim() == 4:
            bare_kwargs["in_channels"] = int(first_w.shape[1])
    elif mt == "sdyff_dyffusion_residual_linear":
        # S-DYff's SHT primitive's nlat / nlon were set by the production
        # env vars (``SDYFF_NLAT`` etc.) but stored as backbone attributes
        # via the source-side ``self.nlat`` we added — overlay the
        # production 0.5° grid values so consumers don't need env state.
        bare_kwargs["nlat"] = 360
        bare_kwargs["nlon"] = 720
        bare_kwargs["lat_crop"] = 0
    bare_blob = {
        "arch": type(backbone).__name__,
        "kwargs": bare_kwargs,
        "state_dict": bare_sd,
    }
    dst_p.parent.mkdir(parents=True, exist_ok=True)
    torch.save(raw, str(dst_p))
    bare_p = dst_p.with_name(dst_p.stem + "_bare.pt")
    torch.save(bare_blob, str(bare_p))

    n_params = sum(v.numel() for v in raw["state_dict"].values()) / 1e6
    print(f"  saved {dst_p}")
    print(f"    model_type        = {mt}")
    print(f"    hp keys (new)     = {sorted(hp_new.keys())}")
    print(f"    block_out_channels= {hp_new.get('block_out_channels')}")
    print(f"    latent_channels   = {hp_new.get('latent_channels')}")
    print(f"    layers_per_block  = {hp_new.get('layers_per_block')}")
    print(f"    params            = {n_params:.2f} M")
    return raw


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("src", help="Legacy Lightning ckpt path")
    ap.add_argument("dst", help="Output (fixed) Lightning ckpt path")
    args = ap.parse_args()
    fix(args.src, args.dst)
