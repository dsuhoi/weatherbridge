"""One calling convention for models whose forward signatures differ.

The transport and autoencoder families take the anchor spacing as a separate
conditioning argument and the static fields as a keyword:

    model(x0, xT, tau_norm, delta_t, static=static)  -> tensor or (tensor, aux)

PixelAttn-VFI descends from a video-interpolation codebase and takes only

    model(x0, xT, tau_norm)

with the static fields fused inside its own forward. :func:`predict` hides
that difference so a caller works in hours and never has to remember which
family it is holding.
"""
from __future__ import annotations

import torch

from ._loader import load_static_features, model_info

# Families whose forward takes (x0, xT, tau, delta_t, static=...)
_CONDITIONED = {"weatherbridge", "weatherdcae", "fuxi", "modafno", "sdyff"}


def linear_interpolation(
    x0: torch.Tensor, xT: torch.Tensor, tau_hours: float, delta_t_hours: float
) -> torch.Tensor:
    """The baseline every model is measured against: the chord between anchors."""
    weight = float(tau_hours) / float(delta_t_hours)
    return (1.0 - weight) * x0 + weight * xT


@torch.no_grad()
def predict(
    model: torch.nn.Module,
    x0: torch.Tensor,
    xT: torch.Tensor,
    tau_hours: float,
    *,
    name: str | None = None,
    delta_t_hours: float | None = None,
) -> torch.Tensor:
    """Reconstruct the state ``tau_hours`` after ``x0``.

    ``x0`` and ``xT`` are normalised ``(B, 24, 360, 720)`` tensors. Pass the
    catalogue ``name`` the model came from, or rely on the attributes
    :func:`weatherbridge.load_model` attaches to it.
    """
    family = getattr(model, "_wb_family", None)
    if name is not None:
        info = model_info(name)
        family = info["family"]
        delta_t_hours = delta_t_hours or info["delta_t_hours"]
    delta_t_hours = delta_t_hours or getattr(model, "_wb_delta_t", None)
    if family is None or delta_t_hours is None:
        raise ValueError(
            "cannot tell which family this model belongs to; pass name=..., "
            "or load it through weatherbridge.load_model"
        )

    if x0.dim() == 3:
        x0 = x0.unsqueeze(0)
    if xT.dim() == 3:
        xT = xT.unsqueeze(0)
    device = next(model.parameters()).device
    x0 = x0.to(device=device, dtype=torch.float32)
    xT = xT.to(device=device, dtype=torch.float32)

    batch = x0.shape[0]
    tau_norm = torch.full(
        (batch, 1), float(tau_hours) / float(delta_t_hours), device=device
    )

    if family in _CONDITIONED:
        n_static = int(getattr(model, "n_static_features", 0) or 0)
        static = (
            load_static_features(n_static, device).expand(batch, -1, -1, -1)
            if n_static
            else None
        )
        out = model(
            x0,
            xT,
            tau_norm,
            torch.full((batch,), float(delta_t_hours), device=device),
            static=static,
        )
    else:
        out = model(x0, xT, tau_norm)

    return out[0] if isinstance(out, tuple) else out
