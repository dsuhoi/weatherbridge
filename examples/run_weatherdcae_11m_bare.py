"""Legacy WeatherDCAE-mid 11M (6h, ATM-VFI budget) bare-blob load."""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import torch
from examples._bare_loader import load_bare


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_bare(REPO_ROOT / "weights" / "weatherdcae_11m_6h_bare.pt", device)
    print(f"loaded WeatherDCAE-mid: "
          f"{sum(p.numel() for p in model.parameters())/1e6:.2f} M params")

    x0 = torch.randn(1, 24, 360, 720, device=device)
    xT = torch.randn(1, 24, 360, 720, device=device)
    static = torch.load(str(REPO_ROOT / "data" / "static_features_0p5.pt"),
                       weights_only=False).float()
    if static.dim() == 3:
        static = static.unsqueeze(0)
    static = static[:, :model.n_static_features].to(device)

    with torch.no_grad():
        out = model(x0, xT,
                    torch.tensor([[3.0 / 6.0]], device=device),
                    torch.tensor([6.0], device=device),
                    static=static)
    if isinstance(out, tuple):
        out = out[0]
    print(f"output: {tuple(out.shape)}  mean={out.mean().item():+.4f}")


if __name__ == "__main__":
    main()
