#!/usr/bin/env python3
"""Single-process batch eval: ACC + RMSE for multiple model checkpoints.

Loads test data + climatology + stats ONCE, then loops over checkpoints.
For 4 0.5°-trained models this is ~10× faster than per-model standalone evals.

Usage:
    python tools/eval/batch_eval_memmap.py \
        --memmap-dir /tmp/wb2_0p5_cache \
        --test-year 2020 \
        --climatology .../climatology_1990-2019_0p5_canonical_v2.zarr \
        --models name1:path1.ckpt,name2:path2.ckpt,... \
        --out-acc-dir metrics/acc_6h_2020_paper_leaderboard \
        --out-rmse-dir metrics/eval_6h_2020_paper_leaderboard
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
import xarray as xr
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from weather_time_interp.memmap_dataset import ERA5MemmapDataset


def interpolate_time_with_f_interpolate(
    x0: torch.Tensor,
    x1: torch.Tensor,
    tau: torch.Tensor,
    mode: str,
) -> torch.Tensor:
    """Interpolate two temporal anchors; both legacy modes reduce to lerp."""
    if mode not in {"bilinear", "bicubic"}:
        raise ValueError("mode must be 'bilinear' or 'bicubic'")
    tau_flat = tau.float().view(-1)
    batch = x0.size(0)
    if tau_flat.numel() == 1 and batch > 1:
        tau_flat = tau_flat.expand(batch)
    tau_broadcast = tau_flat.view(batch, 1, 1, 1)
    return (1.0 - tau_broadcast) * x0 + tau_broadcast * x1


def _forward_hermite(model, x0, xT, tau, cond, static, *, atm_vfi: bool = False):
    """Forward dispatcher.

    - WeatherHermite: ``model(x0, xT, tau, cond, static=static)``; may return a tuple.
    - Legacy ATM-VFI: ``model.net(x0, xT, tau)`` over 24ch tensors.
    - Capacity-matched ATM-VFI: use its inference wrapper so recorded static
      fields and conditioning are applied exactly as during evaluation.
    """
    if atm_vfi:
        if not getattr(model, "arch", None):
            return model.net(x0, xT, tau)
    out = model(x0, xT, tau, cond, static=static)
    if isinstance(out, tuple):
        return out[0]
    return out


class _ATMVFICapMatchedNetAdapter(torch.nn.Module):
    """Expose a capacity-matched wrapper through the legacy ATM net API."""

    def __init__(self, wrapped: torch.nn.Module) -> None:
        super().__init__()
        self.wrapped = wrapped

    def forward(self, x0, xT, tau):
        out = self.wrapped(x0, xT, tau)
        return out[0] if isinstance(out, tuple) else out


class _ATMVFICapMatchedAdapter(torch.nn.Module):
    """Preserve both matched and legacy ATM evaluator call signatures."""

    def __init__(self, wrapped: torch.nn.Module) -> None:
        super().__init__()
        self.net = _ATMVFICapMatchedNetAdapter(wrapped)
        self.arch = getattr(wrapped, "arch", "atmvfi")

    def forward(self, x0, xT, tau, cond=None, static=None):
        out = self.net.wrapped(x0, xT, tau, cond, static=static)
        return out


def _load_atmvfi_model(ckpt_path: str, state, hparams, device):
    """Load ``PixelAttentionVFI`` ckpt; supports asymmetric in/out channels.

    The 12h ATM-VFI ckpt (``exp_atmvfi_12h_oddskip``) is symmetric 24-in / 24-out.
    The 6h v2 ckpt (``exp_atm_vfi_24ch_v2_static_3yr_135only_fibo``) is 27-in /
    24-out: the trainer concatenated 3 static features (lat-cos, lsm, orog)
    inside its forward before encoding, but predicted residuals only over the
    24 prognostic channels. The canonical ``PixelAttentionVFI`` ties in_ch
    and out_ch to a single ``in_channels`` kwarg — we sidestep by constructing
    the net with the larger encoder ``in_ch`` and replacing ``out_conv`` /
    ``scale`` with the smaller output width, then wrapping forward so the
    encoder sees [x; static] while the bilinear scaffold + delta stay 24ch.
    """
    if hparams.get("arch"):
        from tools.eval.capmatched_loader import load_capmatched_checkpoint

        model, _ = load_capmatched_checkpoint(
            ckpt_path,
            device,
            static_path=hparams.get("static_path", "data/static_features_0p5.pt"),
        )
        return _ATMVFICapMatchedAdapter(model).to(device).eval()

    import importlib.util
    train_mod_path = (
        Path(__file__).resolve().parents[2]
        / "legacy"
        / "scripts"
        / "train_atm_vfi_12h_oddskip.py"
    )
    spec = importlib.util.spec_from_file_location("train_atm_vfi", train_mod_path)
    atm_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(atm_mod)  # type: ignore[union-attr]
    PixelAttentionVFI = atm_mod.PixelAttentionVFI

    # Sniff in / out widths from state_dict (authoritative).
    first_w = state.get("net.frame_encoder.0.0.weight")
    sniffed_in = (
        int(first_w.shape[1]) if first_w is not None and first_w.dim() == 4
        else int(hparams.get("in_channels", 24))
    )
    out_w = state.get("net.out_conv.weight")
    sniffed_out = (
        int(out_w.shape[0]) if out_w is not None and out_w.dim() == 4
        else sniffed_in
    )

    # Drop hparams the current PixelAttentionVFI.__init__ does not accept; set
    # in_channels=sniffed_in so the encoder's first conv matches.
    import inspect as _inspect
    _accepted = set(_inspect.signature(PixelAttentionVFI.__init__).parameters.keys())
    clean_hp = {k: v for k, v in hparams.items() if k in _accepted}
    clean_hp["in_channels"] = sniffed_in
    if sniffed_in != int(hparams.get("in_channels", 24)):
        print(
            f"    [atm_vfi] overriding in_channels "
            f"{hparams.get('in_channels')} → {sniffed_in} (from state_dict)"
        )

    model = PixelAttentionVFI(**clean_hp)

    # Asymmetric I/O: rebuild out_conv + scale to match ckpt output width and
    # wrap net.forward so the encoder sees [x; static] (27ch) while the
    # bilinear scaffold + delta stay 24ch.
    if sniffed_out != sniffed_in:
        import torch.nn as _nn
        n_static = sniffed_in - sniffed_out
        print(
            f"    [atm_vfi] asymmetric I/O: encoder={sniffed_in}ch, "
            f"output={sniffed_out}ch, n_static={n_static} "
            f"(static prepended inside forward)"
        )
        ch0 = model.net.out_conv.weight.shape[1]
        model.net.out_conv = _nn.Conv2d(
            ch0, sniffed_out, kernel_size=3, padding=1
        )
        _nn.init.zeros_(model.net.out_conv.weight)
        _nn.init.zeros_(model.net.out_conv.bias)
        model.net.scale = _nn.Parameter(torch.full((sniffed_out,), 0.1))

        # Load static features (lat-cos / lsm / orog) the same way the trainer
        # did and attach as a non-persistent buffer on the net.
        static_path = clean_hp.get(
            "static_features_path", "data/static_features_0p5.pt"
        )
        # Resolve relative to project root so eval cwd doesn't matter.
        sp = Path(static_path)
        if not sp.is_absolute():
            sp = Path(__file__).resolve().parents[2] / sp
        try:
            static_full = torch.load(str(sp), weights_only=False).float()
            # Match the trainer's choice: use first n_static channels of the
            # tensor (lat-cos / lsm / orog by convention).
            static_buf = static_full[:n_static].unsqueeze(0).contiguous()
        except Exception as _e:
            print(
                f"    [atm_vfi] static load failed ({type(_e).__name__}: {_e}); "
                f"falling back to zeros"
            )
            static_buf = torch.zeros(1, n_static, 1, 1)
        model.net.register_buffer("eval_static", static_buf, persistent=False)

        # Monkey-patch forward to concat static before encoding.
        orig_net = model.net
        n_out = sniffed_out

        def _atmvfi_forward_with_static(x_0, x_T, tau):
            tau_b = tau.view(-1, 1, 1, 1) if tau.dim() <= 2 else tau
            x_bilinear = (1.0 - tau_b) * x_0 + tau_b * x_T  # (B, n_out, H, W)
            B, _, H, W = x_0.shape
            stc = orig_net.eval_static
            if stc.shape[-2:] != (H, W):
                stc = torch.nn.functional.interpolate(
                    stc, size=(H, W), mode="bilinear", align_corners=False
                )
            stc = stc.expand(B, -1, H, W)
            x_0_aug = torch.cat([x_0, stc], dim=1)
            x_T_aug = torch.cat([x_T, stc], dim=1)
            f0 = orig_net.encode_frame(x_0_aug)
            fT = orig_net.encode_frame(x_T_aug)
            h = orig_net.attn(f0[-1], fT[-1])
            h = orig_net.adaln(h, tau.view(-1))
            n_levels = len(orig_net.decoder) // 2
            for i in range(n_levels):
                up = orig_net.decoder[2 * i]
                blk = orig_net.decoder[2 * i + 1]
                h = up(h)
                skip = (f0[-(i + 2)] + fT[-(i + 2)]) / 2
                if h.shape[-2:] != skip.shape[-2:]:
                    h = torch.nn.functional.interpolate(
                        h, size=skip.shape[-2:], mode="bilinear",
                        align_corners=False,
                    )
                h = blk(torch.cat([h, skip], dim=1))
            if h.shape[-2:] != x_0.shape[-2:]:
                h = torch.nn.functional.interpolate(
                    h, size=x_0.shape[-2:], mode="bilinear", align_corners=False
                )
            delta = orig_net.out_conv(h)
            s = torch.tanh(orig_net.scale).view(1, -1, 1, 1)
            return x_bilinear + s * delta

        import types as _types
        orig_net.forward = _types.MethodType(
            lambda self, x_0, x_T, tau: _atmvfi_forward_with_static(x_0, x_T, tau),
            orig_net,
        )
        # Mark effective input width (for keep_n inference).
        orig_net._effective_in_channels = n_out

    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"{ckpt_path}: incompatible ATM-VFI state "
            f"(missing={missing[:5]}, unexpected={unexpected[:5]})"
        )
    model.to(device).eval()
    return model


def load_model_safe(ckpt_path: str, device, channel_groups):
    """Load checkpoint — dispatches between WeatherHermite and ATM-VFI loaders.

    ATM-VFI checkpoints (``PixelAttentionVFI``) come from
    ``train_atm_vfi_12h_oddskip.py`` and use a different call signature
    (``model.net(x0, xT, tau)`` over 24ch tensors). Detection mirrors
    ``tools/eval/region_season_12h_eval.py``: substring match on ckpt path,
    ``state_dict`` ``net.``-prefix fraction, or the presence of the
    ``in_channels`` hparam (used only by ``PixelAttentionVFI``).

    Bare ``*_bare.pt`` blobs (``{arch, kwargs, state_dict}``) skip sniffing
    entirely — kwargs are read straight from the blob.
    """
    if ckpt_path.endswith("_bare.pt"):
        import sys as _sys
        _sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
        from examples._bare_loader import load_bare
        model = load_bare(ckpt_path, device)
        return model.to(device).eval(), "dcae_adaln_residual_linear"
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    hparams = dict(ckpt.get("hyper_parameters", {}))
    state = ckpt.get("state_dict", ckpt)
    mt = hparams.get("model_type", "")

    if hparams.get("arch"):
        from tools.eval.capmatched_loader import load_capmatched_checkpoint

        return load_capmatched_checkpoint(
            ckpt_path,
            device,
            static_path=hparams.get("static_path", "data/static_features_0p5.pt"),
        )

    # --- ATM-VFI detection + load --------------------------------------- #
    state_keys = list(state.keys())
    net_pref_frac = (
        sum(1 for k in state_keys if k.startswith("net.")) / max(1, len(state_keys))
    )
    is_atmvfi = (
        mt.startswith("atm_vfi")
        or "atm_vfi" in ckpt_path.lower()
        or "atmvfi" in ckpt_path.lower()
        or net_pref_frac > 0.95
        or "in_channels" in hparams  # PixelAttentionVFI uses this kwarg
    )
    if is_atmvfi:
        model = _load_atmvfi_model(ckpt_path, state, hparams, device)
        return model, "atm_vfi_pixel_attn"

    if "dcae" in mt:
        boc = tuple(hparams.get("block_out_channels", ()))
        hparams["block_out_channels"] = boc if len(boc) >= 3 else (128, 256, 512)
        lpb = tuple(hparams.get("layers_per_block", (3, 3, 3)))
        hparams["layers_per_block"] = lpb[:3] if len(lpb) > 3 else (lpb or (3, 3, 3))
        # Legacy compat: several DC-AE ckpts were trained by trainer code that
        # ignored block_out_channels / layers_per_block hparams and built the
        # model with class defaults (e.g. (128, 256, 512) + (2, 2, 2)). Their
        # stored hparams may claim (128, 128, 256, 256) + (2, 2, 2, 2) but the
        # actual weights are 3-stage. Detect the true per-stage channel widths
        # by sniffing the encoder block weight shapes from state_dict.
        try:
            import re as _re
            # 1) Per-index channel widths from ResBlocks (conv1.weight first dim)
            # and EfficientViT blocks (qkv_multiscale proj_in.weight first dim / 3).
            idx_chan: dict[int, int] = {}
            for k, v in state.items():
                m = _re.match(r"model\.encoder\.down_blocks\.(\d+)\.conv1\.weight$", k)
                if m and v.dim() == 4:
                    idx_chan[int(m.group(1))] = int(v.shape[0])
                m2 = _re.match(
                    r"model\.encoder\.down_blocks\.(\d+)\.attn\.to_qkv_multiscale\.0\.proj_in\.weight$",
                    k,
                )
                if m2 and v.dim() == 4:
                    # qkv stacks 3 → channel = first_dim / 3
                    idx_chan[int(m2.group(1))] = int(v.shape[0]) // 3
            if idx_chan:
                widths = []
                for idx in sorted(idx_chan):
                    w = idx_chan[idx]
                    if not widths or widths[-1] != w:
                        widths.append(w)
                if (
                    len(widths) >= 3
                    and tuple(widths) != tuple(hparams["block_out_channels"])
                ):
                    print(
                        f"    [legacy-compat] overriding hparam block_out_channels"
                        f" {hparams['block_out_channels']} → {tuple(widths)} based on ckpt state_dict"
                    )
                    hparams["block_out_channels"] = tuple(widths)
                    # Match layers_per_block length to widths; assume 2 (model default).
                    hparams["layers_per_block"] = (2,) * len(widths)
        except Exception as _e:
            print(f"    [legacy-compat] state_dict sniff failed: {_e}")
    # Drop kwargs that __init__ doesn't accept. Some ckpts were saved with
    # legacy / training-only hparams (e.g. `freeze_skip_gates`, `freq_cond_film`,
    # `pyramid_levels`, `lambda_pyramid`) that the current trainer signature
    # no longer recognizes — filter dynamically to keep the loader forward-
    # compatible across trainer revisions.
    import inspect as _inspect
    from trainer_weather_hermite import WeatherHermiteLightningModule

    _accepted = set(_inspect.signature(WeatherHermiteLightningModule.__init__).parameters.keys())
    for k in list(hparams.keys()):
        if k not in _accepted:
            hparams.pop(k, None)
    hparams["channel_groups"] = channel_groups
    model = WeatherHermiteLightningModule(**hparams)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"    state load: missing={len(missing)}, unexpected={len(unexpected)}")
    model.to(device).eval()
    return model, mt


def evaluate_one(model, model_type, loader, wrapped_ds, base_ds, device, mu, sigma, clim, acc_indices, channel_names, n_pl, keep_n_channels=None):
    is_atmvfi = model_type == "atm_vfi_pixel_attn"
    """Run eval for one model. Returns (acc_results, rmse_results).

    Wrapped dataset returns grouped per-hour batches:
      x0, xT:    (B, C, H, W)
      tau:       (B, nH, 1)  — interior hours per group
      tau_hour:  (B, nH, 1)
      target:    (B, nH, C, H, W)

    When ``keep_n_channels`` is set, ``x0``/``xT``/``target`` are sliced to the
    first N channels before model forward + ACC/RMSE accumulation — mirrors the
    behaviour of ``batch_eval_12h_memmap.py`` and is required for ckpts trained
    with ``KEEP_24CH=1`` (i.e. 24 prognostic channels, sst/tcc/tcwv dropped).
    """
    # Filter accumulator channel_names + mu/sigma to the sliced view when needed.
    if keep_n_channels is not None:
        channel_names = list(channel_names)[:keep_n_channels]
        mu = mu[:, :keep_n_channels]
        sigma = sigma[:, :keep_n_channels]
    # Per-hour-channel RMSE accumulators
    rmse_sum_sq = {h: {} for h in range(7)}
    bil_rmse_sum_sq = {h: {} for h in range(7)}
    bic_rmse_sum_sq = {h: {} for h in range(7)}

    # ACC sums (sxy, sxx, syy) per hour for model + bilinear baseline
    sxy = {h: torch.zeros(len(acc_indices), dtype=torch.float64, device=device) for h in range(7)}
    sxx = {h: torch.zeros_like(sxy[0]) for h in range(7)}
    syy = {h: torch.zeros_like(sxy[0]) for h in range(7)}
    bil_sxy = {h: torch.zeros(len(acc_indices), dtype=torch.float64, device=device) for h in range(7)}
    bil_sxx = {h: torch.zeros_like(bil_sxy[0]) for h in range(7)}
    bil_syy = {h: torch.zeros_like(bil_sxy[0]) for h in range(7)}

    n_per_hour = {h: 0 for h in range(7)}

    # Lat weights (1, 1, H, 1)
    H = list(base_ds.memmaps.values())[0].shape[-2]
    lat_grid = np.linspace(89.75, -89.75, H, dtype=np.float32) if H == 360 else np.linspace(90.0, -90.0, H, dtype=np.float32)
    w_lat = np.cos(np.deg2rad(lat_grid))
    w_lat = w_lat / w_lat.sum()
    w_lat = torch.from_numpy(w_lat).to(device).view(1, 1, -1, 1)

    acc_idx_t = torch.tensor(acc_indices, device=device, dtype=torch.long)
    grouped_idx = getattr(wrapped_ds, "_grouped_indices", None)

    def _yr_t0(wrapped_index):
        if grouped_idx is not None:
            base_i = grouped_idx[wrapped_index][0]
        else:
            base_i = wrapped_index
        entry = base_ds.index[base_i]
        return int(entry[0]), int(entry[1])

    with torch.no_grad():
        n_batches = len(loader)
        for batch_idx, batch in enumerate(loader):
            x0 = batch["x0"].to(device, non_blocking=True)
            xT = batch["xT"].to(device, non_blocking=True)
            tau_all = batch["tau"].to(device, non_blocking=True)          # (B, nH, 1)
            tau_hour_all = batch["tau_hour"].long()                       # (B, nH, 1)
            target_all = batch["target"].to(device, non_blocking=True)    # (B, nH, C, H, W)
            static = batch.get("static")
            if static is not None:
                static = static.to(device, non_blocking=True)

            # If model was trained with keep_24ch=true, slice tensors before forward.
            if keep_n_channels is not None and x0.size(1) > keep_n_channels:
                n = keep_n_channels
                x0 = x0[:, :n].contiguous()
                xT = xT[:, :n].contiguous()
                target_all = target_all[:, :, :n].contiguous()

            B = x0.size(0)
            nH = tau_all.size(1)
            cond = torch.full((B,), 6.0, device=device, dtype=torch.float32)

            # Loop over interior hours; predict once per hour for the whole batch
            for h_idx in range(nH):
                tau_h = tau_all[:, h_idx, 0]              # (B,)
                target_h = target_all[:, h_idx]            # (B, C, H, W)
                hours_per_sample = tau_hour_all[:, h_idx, 0].tolist()

                pred_bil = interpolate_time_with_f_interpolate(x0, xT, tau_h, mode="bilinear")
                pred_bic = interpolate_time_with_f_interpolate(x0, xT, tau_h, mode="bicubic")
                pred_model = _forward_hermite(
                    model, x0, xT, tau_h, cond, static, atm_vfi=is_atmvfi
                )

                # Denormalize for ACC
                tgt_phys_a = (target_h * sigma + mu).index_select(1, acc_idx_t)
                mdl_phys_a = (pred_model * sigma + mu).index_select(1, acc_idx_t)
                bil_phys_a = (pred_bil * sigma + mu).index_select(1, acc_idx_t)

                # RMSE in normalized units (matches paper convention)
                err_m_lat = ((pred_model - target_h) ** 2 * w_lat).sum(dim=(-2, -1))  # (B, C)
                err_b_lat = ((pred_bil  - target_h) ** 2 * w_lat).sum(dim=(-2, -1))
                err_c_lat = ((pred_bic  - target_h) ** 2 * w_lat).sum(dim=(-2, -1))

                for i in range(B):
                    h = int(hours_per_sample[i])
                    if h < 0 or h > 6:
                        continue
                    wrapped_index = batch_idx * loader.batch_size + i
                    if wrapped_index >= len(wrapped_ds):
                        break
                    if clim is not None:
                        year, t0 = _yr_t0(wrapped_index)
                        ts = base_ds.time_starts[year] + timedelta(hours=int(t0 + h))
                        doy = ts.timetuple().tm_yday
                        hf = ts.hour + ts.minute / 60.0
                        clim_at_t = clim.lookup(
                            doy,
                            hf,
                            ts.year,
                        ).unsqueeze(0).to(device)

                        t_an = tgt_phys_a[i:i+1] - clim_at_t
                        m_an = mdl_phys_a[i:i+1] - clim_at_t
                        b_an = bil_phys_a[i:i+1] - clim_at_t

                        sxy[h] += (w_lat * m_an * t_an).sum(dim=(0, 2, 3)).double()
                        sxx[h] += (w_lat * m_an * m_an).sum(dim=(0, 2, 3)).double()
                        syy[h] += (w_lat * t_an * t_an).sum(dim=(0, 2, 3)).double()
                        bil_sxy[h] += (w_lat * b_an * t_an).sum(dim=(0, 2, 3)).double()
                        bil_sxx[h] += (w_lat * b_an * b_an).sum(dim=(0, 2, 3)).double()
                        bil_syy[h] += (w_lat * t_an * t_an).sum(dim=(0, 2, 3)).double()
                    n_per_hour[h] += 1

                    for ch_i, name in enumerate(channel_names):
                        rmse_sum_sq[h].setdefault(name, 0.0)
                        rmse_sum_sq[h][name] += float(err_m_lat[i, ch_i].item())
                        bil_rmse_sum_sq[h].setdefault(name, 0.0)
                        bil_rmse_sum_sq[h][name] += float(err_b_lat[i, ch_i].item())
                        bic_rmse_sum_sq[h].setdefault(name, 0.0)
                        bic_rmse_sum_sq[h][name] += float(err_c_lat[i, ch_i].item())

            if batch_idx % 25 == 0:
                print(f"      batch {batch_idx}/{n_batches}")

    # Finalize ACC: mean over channels per hour
    acc_per_hour = {}
    for h in range(7):
        n = n_per_hour[h]
        if n == 0:
            continue
        m_acc = sxy[h] / torch.sqrt(sxx[h] * syy[h] + 1e-12)
        b_acc = bil_sxy[h] / torch.sqrt(bil_sxx[h] * bil_syy[h] + 1e-12)
        acc_per_hour[str(h)] = {
            "model": {
                "acc_mean": float(m_acc.mean().item()),
                **{f"acc_{c}": float(m_acc[i].item()) for i, c in enumerate([channel_names[j] for j in acc_indices])},
            },
            "bilinear": {
                "acc_mean": float(b_acc.mean().item()),
                **{f"acc_{c}": float(b_acc[i].item()) for i, c in enumerate([channel_names[j] for j in acc_indices])},
            },
        }

    # Finalize RMSE: sqrt(sum_sq / n) per channel per hour
    rmse_per_hour = {}
    for h in range(7):
        n = n_per_hour[h]
        if n == 0:
            continue
        rmse_per_hour[str(h)] = {"model": {}, "bilinear": {}, "bicubic": {}}
        for name in channel_names:
            if name in rmse_sum_sq[h]:
                rmse_per_hour[str(h)]["model"][f"rmse_{name}"] = float(np.sqrt(rmse_sum_sq[h][name] / n))
                rmse_per_hour[str(h)]["bilinear"][f"rmse_{name}"] = float(np.sqrt(bil_rmse_sum_sq[h][name] / n))
                rmse_per_hour[str(h)]["bicubic"][f"rmse_{name}"] = float(np.sqrt(bic_rmse_sum_sq[h][name] / n))

    return acc_per_hour, rmse_per_hour, n_per_hour


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--memmap-dir", default="/tmp/wb2_0p5_cache")
    ap.add_argument("--test-year", type=int, default=2020)
    ap.add_argument("--climatology", default="/workspace-SR006.nfs2/weather_data/time_interpolation_0p5/climatology_1990-2019_0p5_canonical_v2.zarr")
    ap.add_argument("--stats-path", default="data/json_stats_0p5.nc")
    ap.add_argument("--surface-stats-path", default="data/surface_stats_0p5.json")
    ap.add_argument("--static-path", default="data/static_features_0p5.pt")
    ap.add_argument("--models", required=True,
                    help="Comma-separated list NAME:CKPT[:ENVS], e.g. modafno:path.ckpt:MODAFNO_INP_H=360")
    ap.add_argument("--out-acc-dir", default="metrics/acc_6h_2020_paper_leaderboard")
    ap.add_argument("--out-rmse-dir", default="metrics/eval_6h_2020_paper_leaderboard")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--samples-per-date", type=int, default=4)
    ap.add_argument("--eval-days-per-month", type=int, default=4)
    ap.add_argument(
        "--keep-n-channels",
        type=int,
        default=None,
        help=(
            "Slice x0/xT/target to the first N channels before model forward. "
            "Use 24 for keep_24ch=true trained ckpts (drops sst/tcc/tcwv). "
            "Mirrors the same flag in batch_eval_12h_memmap.py."
        ),
    )
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    t_global = time.time()
    print(f"loading test dataset (memmap year={args.test_year})...")
    ds_base = ERA5MemmapDataset(
        memmap_dir=args.memmap_dir,
        years=[args.test_year],
        max_tau_hours=6,
        samples_per_date=args.samples_per_date,
        train=False,
        eval_hours=list(range(7)),
        static_path=args.static_path,
        stats_path=args.stats_path,
        surface_stats_path=args.surface_stats_path,
    )
    # Economy days filter
    if args.eval_days_per_month is not None:
        K = max(1, int(args.eval_days_per_month))
        day_picks = {3: [1, 11, 21], 4: [1, 8, 15, 22], 5: [1, 7, 14, 21, 28]}.get(
            K, sorted({1 + i * (30 // K) for i in range(K)})
        )
        allowed = set(day_picks)
        import datetime as _dt
        filt = []
        for entry in ds_base.index:
            y, t0, _, _ = entry
            doy = t0 // 24
            try:
                d = _dt.date(int(y), 1, 1) + _dt.timedelta(days=int(doy))
            except Exception:
                continue
            if d.day in allowed:
                filt.append(entry)
        print(f"  economy filter: {len(filt)}/{len(ds_base.index)} index entries")
        ds_base.index = filt

    channel_groups = ds_base.channel_groups
    full_channel_names = list(ds_base.channel_names) + list(ds_base.surface_variables)
    print(f"  all model channels ({len(full_channel_names)}): {full_channel_names}")

    # Pre-probe climatology (absent -> RMSE-only mode).
    channel_names = full_channel_names  # full list for RMSE
    n_pl = len(ds_base.channel_names)
    if not Path(args.climatology).exists():
        print(f"  [no-acc] climatology {args.climatology} not found - RMSE only.")
        clim = None
        acc_indices = []
        acc_channel_names = []
    else:
        from tools.eval.compute_acc import ClimatologyLookup, CLIM_PL_NAMES

        print(f"loading climatology {args.climatology}...")
        _probe = xr.open_zarr(str(args.climatology), consolidated=True)
        _clim_vars = set(_probe.data_vars)
        _probe.close()
        skip = {"tisr"}
        for c in full_channel_names:
            is_pl = len(c) > 1 and c[0] in CLIM_PL_NAMES and c[1:].isdigit()
            if is_pl:
                if CLIM_PL_NAMES[c[0]] not in _clim_vars:
                    skip.add(c)
            else:
                if c not in _clim_vars:
                    skip.add(c)
        print(f"  ACC skip: {sorted(skip)}")
        acc_indices = [i for i, c in enumerate(full_channel_names) if c not in skip]
        if args.keep_n_channels is not None:
            acc_indices = [i for i in acc_indices if i < args.keep_n_channels]
        acc_channel_names = [full_channel_names[i] for i in acc_indices]
        clim = ClimatologyLookup(Path(args.climatology), acc_channel_names, device)
        print(f"  climatology shape: {tuple(clim.clim.shape)}")

    # Denorm stats
    mu_all = torch.cat([ds_base.mu, ds_base.surface_mu], dim=0).to(device).view(1, -1, 1, 1)
    sigma_all = torch.cat([ds_base.sigma, ds_base.surface_sigma], dim=0).to(device).view(1, -1, 1, 1)

    # Loader (shared across all models)
    from trainer_weather_hermite import ERA5WeatherHermiteDataset

    test_wrapped = ERA5WeatherHermiteDataset(ds_base, delta_t_hours=6.0)
    print(f"  test dataset: {len(test_wrapped)} windows")
    loader = DataLoader(test_wrapped, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True,
                        persistent_workers=args.num_workers > 0)
    print(f"setup done in {time.time()-t_global:.1f}s")

    # Models list
    models = []
    for entry in args.models.split(","):
        parts = entry.split(":")
        if len(parts) >= 2:
            name, ckpt = parts[0], parts[1]
            envs = ":".join(parts[2:]) if len(parts) > 2 else ""
            models.append((name, ckpt, envs))

    out_acc = Path(args.out_acc_dir); out_acc.mkdir(parents=True, exist_ok=True)
    out_rmse = Path(args.out_rmse_dir); out_rmse.mkdir(parents=True, exist_ok=True)

    for i, (name, ckpt, envs) in enumerate(models, 1):
        print(f"\n=== [{i}/{len(models)}] {name} ===")
        if not Path(ckpt).exists():
            print(f"  [MISS] {ckpt}")
            continue
        # Set env vars
        for kv in envs.split():
            if "=" in kv:
                k, v = kv.split("=", 1)
                os.environ[k] = v
                print(f"  env {k}={v}")
        t_m = time.time()
        model, mt = load_model_safe(ckpt, device, channel_groups)
        print(f"  model: {sum(p.numel() for p in model.parameters())/1e6:.1f}M params, type={mt}")
        # Per-model keep_n override: ATM-VFI ckpts may declare 24 or 27 input
        # channels independent of the CLI flag (6h v2 uses 27, 12h uses 24).
        # Sniff the actual width from the loaded module to avoid passing the
        # wrong number of channels to a model the CLI flag mis-classified.
        model_keep_n = args.keep_n_channels
        if mt == "atm_vfi_pixel_attn":
            try:
                # Effective input width = encoder in_ch unless asymmetric I/O
                # is used (then the prognostic input is smaller; static is
                # appended inside forward).
                eff = getattr(model.net, "_effective_in_channels", None)
                in_w = int(eff) if eff is not None else int(
                    model.net.frame_encoder[0][0].weight.shape[1]
                )
                if model_keep_n != in_w:
                    print(
                        f"  [atm_vfi] overriding keep_n_channels "
                        f"{model_keep_n} → {in_w} (effective prog input width)"
                    )
                    model_keep_n = in_w
            except (AttributeError, IndexError):
                pass
        acc_res, rmse_res, n_per_hour = evaluate_one(
            model, mt, loader, test_wrapped, ds_base, device, mu_all, sigma_all,
            clim, acc_indices, full_channel_names, n_pl,
            keep_n_channels=model_keep_n,
        )
        rmse_channel_names = (
            channel_names[:model_keep_n]
            if model_keep_n is not None
            else channel_names
        )
        payload_acc = {
            "checkpoint": ckpt, "model_type": mt,
            "num_samples": sum(n_per_hour.values()),
            "years": [args.test_year],
            "channel_names": acc_channel_names,
            "per_hour": acc_res,
        }
        payload_rmse = {
            "checkpoint": ckpt, "model_type": mt,
            "num_samples": sum(n_per_hour.values()),
            "years": [args.test_year],
            "channel_names": rmse_channel_names,
            "per_hour": rmse_res,
        }
        with open(out_acc / f"{name}.json", "w") as f:
            json.dump(payload_acc, f, indent=2)
        with open(out_rmse / f"{name}.json", "w") as f:
            json.dump(payload_rmse, f, indent=2)
        print(f"  saved {name} in {(time.time()-t_m)/60:.1f} min")
        del model
        torch.cuda.empty_cache()

    print(f"\n=== ALL DONE in {(time.time()-t_global)/60:.1f} min ===")


if __name__ == "__main__":
    main()
