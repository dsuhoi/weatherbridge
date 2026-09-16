"""Minimal WeatherDCAE-14M (6h) bare-blob inference."""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import torch

from examples._bare_loader import load_bare


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_bare(
        REPO_ROOT / "weights" / "weatherdcae_14m_6h_bare.pt",
        device,
    )
    print(
        "loaded WeatherDCAE-14M 6h: "
        f"{sum(p.numel() for p in model.parameters()) / 1e6:.2f} M params"
    )

    input_channels = int(model.in_channels)
    x0 = torch.randn(1, input_channels, 360, 720, device=device)
    xT = torch.randn(1, input_channels, 360, 720, device=device)
    static = torch.load(
        REPO_ROOT / "data" / "static_features_0p5.pt",
        weights_only=False,
    ).float()
    if static.dim() == 3:
        static = static.unsqueeze(0)
    static = static[:, :model.n_static_features].to(device)

    with torch.no_grad():
        out = model(
            x0,
            xT,
            torch.tensor([[3.0 / 6.0]], device=device),
            torch.tensor([6.0], device=device),
            static=static,
        )
    if isinstance(out, tuple):
        out = out[0]
    print(
        f"input_channels: {input_channels}  output: {tuple(out.shape)}  "
        f"mean={out.mean().item():+.4f}"
    )


if __name__ == "__main__":
    main()
