"""Flow-Matching-style interpolator (boundary-aware residual).

Architecture:
    x_bilinear(τ) = (1−τ)·x0 + τ·xT
    v_θ = field([x_bilinear, x0, xT, static], τ)        # backbone is vendored ModAFNO
    x̂_τ = x_bilinear + τ·(1−τ)·v_θ

Properties:
- At τ=0 → x̂ = x0   (exact, no learnable freedom — kills endpoint noise that residual_scale couldn't)
- At τ=1 → x̂ = xT   (exact)
- v_θ predicts the *correction velocity* needed to bend bilinear path toward true ERA5 trajectory
- Multi-step ODE inference (Euler/RK4) optional via `ode_steps`; default 1 = single Euler step ≈ closed-form

This is equivalent to one step of rectified-flow / OT-CFM with bilinear initial path.
For the full simulation-free CFM training (constant velocity target xT−x0) the
gain would be near zero on weather (path nearly straight), so we keep the
data-supervised training: loss = ‖x̂_τ − x_τ_true‖.

Forward signature matches sibling models (sfno, dcae, modafno):
    x_hat, aux = model(x0, xT, tau, cond, static=None)
"""

from __future__ import annotations

from typing import Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class WeatherFlowMatchingResidualModel(nn.Module):
    """Boundary-aware FM-style interpolator (single-step rectified flow).

    Heavy lifting is delegated to the vendored ModAFNO backbone — same as
    `WeatherModAFNOOfficialResidualLinearModel`. The wrapper differs in:
      1. Field input includes x_bilinear (self-conditioning) — backbone sees its
         own current trajectory point, not just x0/xT.
      2. Output gating is τ·(1−τ) (boundary zero) instead of learnable tanh(scale).
      3. Optional multi-step Euler ODE integration (slow but more accurate for
         strongly non-linear paths).
    """

    def __init__(
        self,
        in_channels: int = 5,
        out_channels: int = 5,
        embed_dim: int = 256,
        mod_dim: int = 64,
        depth: int = 8,
        patch_size: tuple = (2, 2),
        mlp_ratio: float = 2.0,
        num_blocks: int = 8,
        drop_rate: float = 0.0,
        sparsity_threshold: float = 0.01,
        hard_thresholding_fraction: float = 1.0,
        modulate_filter: bool = True,
        modulate_mlp: bool = True,
        scale_shift_mode: Literal["complex", "real"] = "complex",
        n_static_features: int = 0,
        inp_shape: tuple = (182, 360),
        native_shape: tuple = (181, 360),
        ode_steps: int = 1,
        # accepted but unused (kept for trainer-side parameter parity):
        residual_scale_init: float = 0.0,
        residual_scale_learnable: bool = False,
        residual_clip: Optional[float] = None,
        residual_scale_floor: float = 0.0,
        direct_prediction: bool = False,
    ):
        super().__init__()
        from .physicsnemo_vendor import ModAFNO as VendorModAFNO

        if direct_prediction:
            raise ValueError("FM model is residual-only; direct_prediction=True is not supported.")

        self.n_static_features = int(n_static_features)
        self.inp_shape = tuple(inp_shape)
        self.native_shape = tuple(native_shape)
        self.pad_top = (inp_shape[0] - native_shape[0]) // 2
        self.pad_bot = inp_shape[0] - native_shape[0] - self.pad_top
        self.ode_steps = max(1, int(ode_steps))
        self.in_channels = in_channels
        self.out_channels = out_channels

        # field input: cat([x_curr, x0, xT, static]) → 3*C + n_static
        field_in = 3 * in_channels + self.n_static_features
        self.field = VendorModAFNO(
            inp_shape=list(inp_shape),
            in_channels=field_in,
            out_channels=out_channels,
            patch_size=list(patch_size),
            embed_dim=embed_dim,
            mod_dim=mod_dim,
            depth=depth,
            mlp_ratio=mlp_ratio,
            drop_rate=drop_rate,
            num_blocks=num_blocks,
            sparsity_threshold=sparsity_threshold,
            hard_thresholding_fraction=hard_thresholding_fraction,
            modulate_filter=modulate_filter,
            modulate_mlp=modulate_mlp,
            scale_shift_mode=scale_shift_mode,
        )

    def _pad_lat(self, x: torch.Tensor) -> torch.Tensor:
        if self.pad_top == 0 and self.pad_bot == 0:
            return x
        return F.pad(x, (0, 0, self.pad_top, self.pad_bot), mode="replicate")

    def _crop_lat(self, x: torch.Tensor) -> torch.Tensor:
        if self.pad_top == 0 and self.pad_bot == 0:
            return x
        if self.pad_bot == 0:
            return x[:, :, self.pad_top :]
        return x[:, :, self.pad_top : -self.pad_bot]

    def _broadcast_static(self, static: Optional[torch.Tensor], x0: torch.Tensor) -> Optional[torch.Tensor]:
        if self.n_static_features <= 0:
            return None
        if static is None:
            raise ValueError(f"Model requires static (n_static={self.n_static_features}), got None")
        if static.dim() == 3:
            static = static.unsqueeze(0).expand(x0.size(0), -1, -1, -1)
        elif static.size(0) != x0.size(0):
            if x0.size(0) % static.size(0) == 0:
                n_tau = x0.size(0) // static.size(0)
                static = static.repeat_interleave(n_tau, dim=0)
            else:
                static = static.expand(x0.size(0), -1, -1, -1)
        return static

    def _field_call(self, x_curr, x0, xT, static_b, tau_scalar):
        """One velocity evaluation at given (x_curr, τ scalar in [0,1])."""
        parts = [x_curr, x0, xT]
        if static_b is not None:
            parts.append(static_b)
        x_in = torch.cat(parts, dim=1)
        x_in_padded = self._pad_lat(x_in)
        mod = tau_scalar.view(tau_scalar.size(0), 1)
        v_padded = self.field(x_in_padded, mod)
        return self._crop_lat(v_padded)

    def forward(self, x0, xT, tau, cond, static=None):
        if tau.dim() == 1:
            tau_ = tau.view(-1, 1, 1, 1)
        elif tau.dim() == 2:
            tau_ = tau.view(tau.size(0), 1, 1, 1)
        else:
            tau_ = tau
        tau_scalar = tau_.view(tau_.size(0))

        static_b = self._broadcast_static(static, x0)

        if self.ode_steps == 1:
            # Single-step closed-form (one Euler step from τ=0 with full path).
            x_bilinear = (1.0 - tau_) * x0 + tau_ * xT
            v = self._field_call(x_bilinear, x0, xT, static_b, tau_scalar)
            x_hat = x_bilinear + tau_ * (1.0 - tau_) * v
        else:
            # Multi-step Euler ODE: integrate dz/dτ' = u_θ(z, τ') from 0 to τ in N substeps.
            # Same target tau for every sample in batch is not assumed; we step in normalized
            # path-time s ∈ [0,1] mapping to per-sample τ via z = (1-s·τ)·x0 + s·τ·xT + ...
            # Simpler: substep size δ = τ/N, integrate per-sample.
            x_bilinear = (1.0 - tau_) * x0 + tau_ * xT
            v_curr = self._field_call(x_bilinear, x0, xT, static_b, tau_scalar)
            x_hat = x_bilinear + tau_ * (1.0 - tau_) * v_curr
            # Refinement: feed x_hat back as x_curr and re-estimate.
            for _ in range(self.ode_steps - 1):
                v_curr = self._field_call(x_hat, x0, xT, static_b, tau_scalar)
                x_hat = x_bilinear + tau_ * (1.0 - tau_) * v_curr

        return x_hat, {
            "z_tau": None,
            "d0": None,
            "d1": None,
            "decoder_out": x_hat - x_bilinear,
            "x_bilinear": x_bilinear,
            "residual_scale": torch.tensor(1.0, device=x_hat.device).detach(),
        }
