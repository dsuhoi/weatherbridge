#!/usr/bin/env python3
"""Export a WeatherDCAE Lightning checkpoint as a strict bare blob."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from tools.eval.dcae_checkpoint_loader import dcae_bare_payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    payload = dcae_bare_payload(args.ckpt)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out_path)
    params = sum(value.numel() for value in payload["state_dict"].values())
    print(
        f"[bare] WeatherDCAE -> {out_path} "
        f"({params / 1e6:.2f}M params, "
        f"{payload['kwargs']['in_channels']} input channels)"
    )


if __name__ == "__main__":
    main()
