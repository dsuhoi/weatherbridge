"""ATM-VFI v2 12M (6h) — bare blob load. ATM-VFI v2 has asymmetric I/O
(encoder takes 27ch via static prefuse, residual stream stays 24ch); the
patched forward that does the prefuse is preserved by the bare blob, so
``model.net(x0, xT, tau)`` Just Works."""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import torch
from examples._bare_loader import load_bare


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_bare(REPO_ROOT / "weights" / "atmvfi_v2_6h_bare.pt", device)
    print(f"loaded ATM-VFI v2: "
          f"{sum(p.numel() for p in model.parameters())/1e6:.2f} M params")

    x0 = torch.randn(1, 24, 360, 720, device=device)
    xT = torch.randn(1, 24, 360, 720, device=device)
    with torch.no_grad():
        out = model.net(x0, xT, torch.tensor([[3.0 / 6.0]], device=device))
    print(f"output: {tuple(out.shape)}  mean={out.mean().item():+.4f}")


if __name__ == "__main__":
    main()
