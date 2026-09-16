#!/usr/bin/env python3
"""One-step synthetic smoke test for universal WeatherBridge variants."""
from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import torch

from tools.train.train_capacity_matched_6h import CapMatchedLit


ARCHES = ("flow_universal_latent", "flow_universal_fm")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--height", type=int, default=360)
    parser.add_argument("--width", type=int, default=720)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--arches", nargs="+", default=list(ARCHES))
    parser.add_argument("--anchor-swap-probability", type=float, default=0.0)
    args = parser.parse_args()
    if args.height % 8 or args.width % 8:
        raise SystemExit("height and width must be divisible by 8")

    device = torch.device(args.device)
    with tempfile.TemporaryDirectory(prefix="weatherbridge-universal-smoke-") as tmp:
        static_path = Path(tmp) / "static.pt"
        torch.save(torch.zeros(3, args.height, args.width), static_path)
        for arch in args.arches:
            module = CapMatchedLit(
                arch=arch,
                static_path=str(static_path),
                lr=1e-4,
                delta_t=6.0,
                eval_taus=(1, 2, 3, 4, 5),
                total_steps=8,
                anchor_swap_probability=args.anchor_swap_probability,
            ).to(device)
            module.log = lambda *unused_args, **unused_kwargs: None
            batch_size = args.batch_size
            tau_hours = torch.arange(
                batch_size,
                device=device,
            ).remainder(5).add(1)
            batch = {
                "x0": torch.randn(
                    batch_size,
                    24,
                    args.height,
                    args.width,
                    device=device,
                ),
                "x1": torch.randn(
                    batch_size,
                    24,
                    args.height,
                    args.width,
                    device=device,
                ),
                "target": torch.randn(
                    batch_size,
                    24,
                    args.height,
                    args.width,
                    device=device,
                ),
                "tau": tau_hours / 6.0,
                "tau_hour": tau_hours,
            }
            module.train()
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                loss = module._step(batch, "train")
            loss.backward()
            grad_norm = torch.sqrt(
                sum(
                    parameter.grad.abs().float().square().sum()
                    for parameter in module.net.parameters()
                    if parameter.grad is not None
                )
            )
            if not torch.isfinite(loss) or not torch.isfinite(grad_norm):
                raise RuntimeError(f"{arch}: non-finite training smoke")
            module.eval()
            with torch.no_grad(), torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                validation_loss = module._step(batch, "val")
            if not torch.isfinite(validation_loss):
                raise RuntimeError(f"{arch}: non-finite validation smoke")
            print(
                f"{arch}: loss={loss.item():.6f} "
                f"grad_norm={grad_norm.item():.6f} "
                f"val={validation_loss.item():.6f}",
                flush=True,
            )
            del module, batch, loss, validation_loss
            if device.type == "cuda":
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
