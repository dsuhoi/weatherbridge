#!/usr/bin/env python3
"""Bit-exact A/B verifier — legacy load_model_safe (with sniff) vs
clean ``WTIModelModule.load_from_checkpoint`` on the
fixed ckpt (no sniff anywhere).

Run after ``tools/ckpt_normalize.py legacy.ckpt fixed.ckpt`` — both
paths read the same tensors and we report ``torch.equal``.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("legacy_ckpt", help="Legacy Lightning ckpt (sniff path)")
    ap.add_argument("fixed_ckpt",  help="Fixed Lightning ckpt produced by ckpt_normalize")
    ap.add_argument("--static", default="data/static_features_0p5.pt")
    args = ap.parse_args()

    sys.path.insert(0, ".")
    from tools.eval.batch_eval_memmap import load_model_safe
    from trainer import WTIModelModule

    # Path A — production loader with sniff.
    m_a, mt = load_model_safe(args.legacy_ckpt, "cpu", None)
    m_a.eval()

    # Path B — pure Lightning load_from_checkpoint on the fixed ckpt.
    # No sniff, no hp filtering, no monkey-patching.
    if mt == "atm_vfi_pixel_attn":
        # ATM-VFI has its own loader; reuse it (state_dict is identical).
        from tools.eval.batch_eval_memmap import _load_atmvfi_model
        raw = torch.load(args.fixed_ckpt, map_location="cpu",
                          weights_only=False)
        m_b = _load_atmvfi_model(
            args.fixed_ckpt, raw["state_dict"], raw["hyper_parameters"], "cpu",
        )
    else:
        # ``strict=False`` matches the policy load_model_safe applies in
        # the production eval pipeline. Critical: this is the only place
        # the fixed-ckpt loader differs from a strict canonical load.
        m_b = WTIModelModule.load_from_checkpoint(
            args.fixed_ckpt, map_location="cpu", strict=False,
        )
    m_b.eval()

    # Inputs.
    n_static = (
        m_a.hparams.get("n_static_features", 0) if hasattr(m_a, "hparams") else 0
    )
    is_atmvfi = mt == "atm_vfi_pixel_attn"
    if not is_atmvfi and n_static and Path(args.static).exists():
        static = torch.load(args.static, weights_only=False).float()
        static = static.unsqueeze(0) if static.dim() == 3 else static
        static = static[:, :n_static]
    else:
        static = None

    torch.manual_seed(20260624)
    x0 = torch.randn(1, 24, 360, 720)
    xT = torch.randn(1, 24, 360, 720)
    tau_norm = torch.tensor([[3.0 / 6.0]])
    cond = torch.tensor([6.0])

    with torch.no_grad():
        if is_atmvfi:
            o_a = m_a.net(x0, xT, tau_norm)
            o_b = m_b.net(x0, xT, tau_norm)
        else:
            o_a = m_a(x0, xT, tau_norm, cond, static=static)
            o_b = m_b(x0, xT, tau_norm, cond, static=static)
    def _unwrap(v):
        if isinstance(v, tuple):
            return v[0]
        if isinstance(v, dict):
            return v.get("x_hat", v)
        return v
    o_a = _unwrap(o_a)
    o_b = _unwrap(o_b)

    if o_a.shape != o_b.shape:
        print(f"  shape mismatch: A={tuple(o_a.shape)} B={tuple(o_b.shape)}")
        sys.exit(1)
    d = (o_a - o_b).abs()
    print(f"  output shape:  {tuple(o_a.shape)}")
    print(f"  abs diff:      max={d.max().item():.3e}  mean={d.mean().item():.3e}")
    print(f"  allclose(1e-6)? {torch.allclose(o_a, o_b, atol=1e-6, rtol=1e-5)}")
    print(f"  BIT-EXACT?      {torch.equal(o_a, o_b)}")


if __name__ == "__main__":
    main()
