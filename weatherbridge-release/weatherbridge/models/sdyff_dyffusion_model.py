"""S-DYff with DYffusion-style iterative refinement (Cachay et al. 2023, adapted).

Two-network architecture for the interpolation task:
  - Interpolator I_φ(x_0, x_T, τ): single-shot prediction of x̂_τ (Spherical FNO).
  - Refiner R_θ(x̂_τ, x_0, x_T, τ): denoises/refines the interpolator output.

Training: I_φ trained with MSE on x_τ_true. R_θ trained to refine I_φ outputs
toward x_τ_true (with MC-dropout on I_φ to provide stochastic input distribution).

Inference: x̂_τ^(0) = I_φ(x_0, x_T, τ) (or bilinear);
           for k = 1..K: x̂_τ^(k) = R_θ(x̂_τ^(k-1), x_0, x_T, τ).
Final output keeps the bilinear-residual wrapping: x_hat = bilinear + tanh(α) * res.

Differences from paper-original DYffusion (forecasting framework):
  - Paper task is forecasting (predict x_T from x_0); ours is interpolation (predict
    x_τ from x_0 and x_T). I_φ plays the same role, R_θ replaces forecaster F_θ.
  - Paper iterates over diffusion timesteps n=N..1; ours iterates the refiner over
    fixed K passes (cold-diffusion / iterative-refinement spirit).
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .sdyff_baseline_model import (
    SDyffBlock,
    SphericalConv2d,
    SinusoidalPosEmb,
    TimeMLP,
)


class _SHTBackbone(nn.Module):
    """Reusable SHT backbone: conv_in → N × SDyffBlock(FiLM) → conv_out."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        embed_dim: int = 96,
        num_layers: int = 4,
        time_dim: int = 128,
        n_modes_lat: int = 16,
        n_modes_lon: int = 32,
        mlp_ratio: float = 2.0,
        dropout: float = 0.1,
        drop_path: float = 0.1,
        nlat: int = 180,
        nlon: int = 360,
    ):
        super().__init__()
        self.conv_in = nn.Conv2d(in_channels, embed_dim, kernel_size=1)
        self.blocks = nn.ModuleList([
            SDyffBlock(
                embed_dim=embed_dim, time_dim=time_dim,
                nlat=nlat, nlon=nlon,
                n_modes_lat=n_modes_lat, n_modes_lon=n_modes_lon,
                mlp_ratio=mlp_ratio, dropout=dropout, drop_path=drop_path,
            )
            for _ in range(num_layers)
        ])
        self.norm_out = nn.InstanceNorm2d(embed_dim, affine=True)
        self.conv_out = nn.Conv2d(embed_dim, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv_in(x)
        for block in self.blocks:
            h = block(h, t_emb)
        h = self.norm_out(h)
        return self.conv_out(h)


class WeatherSDyffusionDYffusionModel(nn.Module):
    """Interpolator + iterative refiner with bilinear-residual wrapping."""

    def __init__(
        self,
        in_channels: int = 5,
        out_channels: int = 5,
        embed_dim: int = 96,
        num_layers: int = 6,
        refiner_num_layers: int = 4,
        time_dim: int = 128,
        n_modes_lat: int = 16,
        n_modes_lon: int = 32,
        mlp_ratio: float = 2.0,
        dropout: float = 0.1,
        drop_path: float = 0.1,
        nlat: int = 180,
        nlon: int = 360,
        lat_crop: int = 1,
        n_inference_steps: int = 5,
        n_train_refine_steps: int = 1,
        residual_scale_init: float = 0.10,
        residual_scale_learnable: bool = True,
        residual_clip: float | None = None,
        residual_scale_floor: float = 0.0,
        n_static_features: int = 0,
        time_base_period: float = 16.0,
    ):
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.lat_crop = max(0, int(lat_crop))
        self.residual_clip = residual_clip
        self.residual_scale_floor = float(residual_scale_floor)
        self.nlat = nlat
        self.nlon = nlon
        self.n_inference_steps = max(0, int(n_inference_steps))
        self.n_train_refine_steps = max(0, int(n_train_refine_steps))
        self.n_static_features = int(n_static_features)
        # Persist remaining init kwargs for introspection-based loaders.
        self.embed_dim = int(embed_dim)
        self.num_layers = int(num_layers)
        self.refiner_num_layers = int(refiner_num_layers)
        self.time_dim = int(time_dim)
        self.n_modes_lat = int(n_modes_lat)
        self.n_modes_lon = int(n_modes_lon)
        self.mlp_ratio = float(mlp_ratio)
        self.dropout = float(dropout)
        self.drop_path = float(drop_path)
        self.residual_scale_init = float(residual_scale_init)
        self.residual_scale_learnable = bool(residual_scale_learnable)
        self.time_base_period = float(time_base_period)

        self.time_mlp = TimeMLP(time_dim=time_dim, freq_dim=64, base_period=float(time_base_period))

        # I_φ: input = cat([x0, xT, static]), output = x̂_τ
        interp_in = in_channels * 2 + self.n_static_features
        self.interpolator = _SHTBackbone(
            in_channels=interp_in,
            out_channels=out_channels,
            embed_dim=embed_dim, num_layers=num_layers, time_dim=time_dim,
            n_modes_lat=n_modes_lat, n_modes_lon=n_modes_lon,
            mlp_ratio=mlp_ratio, dropout=dropout, drop_path=drop_path,
            nlat=nlat, nlon=nlon,
        )

        # R_θ: input = cat([x_curr, x0, xT, static]), output = refined x̂_τ
        refiner_in = in_channels * 3 + self.n_static_features
        self.refiner = _SHTBackbone(
            in_channels=refiner_in,
            out_channels=out_channels,
            embed_dim=embed_dim, num_layers=refiner_num_layers, time_dim=time_dim,
            n_modes_lat=n_modes_lat, n_modes_lon=n_modes_lon,
            mlp_ratio=mlp_ratio, dropout=dropout, drop_path=drop_path,
            nlat=nlat, nlon=nlon,
        )

        init_scale = torch.tensor(float(residual_scale_init), dtype=torch.float32)
        if residual_scale_learnable:
            self.residual_scale = nn.Parameter(init_scale)
        else:
            self.register_buffer("residual_scale", init_scale)

    def _crop_lat(self, x: torch.Tensor) -> torch.Tensor:
        if self.lat_crop <= 0:
            return x
        h = x.size(-2)
        if h <= self.lat_crop:
            raise ValueError(f"lat_crop={self.lat_crop} too large for height={h}")
        crop_top = self.lat_crop // 2
        crop_bottom = self.lat_crop - crop_top
        if crop_bottom == 0:
            return x[:, :, crop_top:]
        return x[:, :, crop_top:-crop_bottom]

    def _expand_static(self, static: torch.Tensor, batch: int) -> torch.Tensor:
        if static is None:
            return None
        if static.dim() == 3:
            static = static.unsqueeze(0).expand(batch, -1, -1, -1)
        elif static.size(0) != batch:
            if batch % static.size(0) == 0:
                static = static.repeat_interleave(batch // static.size(0), dim=0)
            else:
                static = static.expand(batch, -1, -1, -1)
        return static

    def _residual_scale_clipped(self) -> torch.Tensor:
        residual_scale = torch.tanh(self.residual_scale)
        if self.residual_scale_floor > 0.0:
            sign = torch.where(
                residual_scale >= 0,
                torch.ones_like(residual_scale),
                -torch.ones_like(residual_scale),
            )
            magnitude = residual_scale.abs().clamp(min=self.residual_scale_floor)
            residual_scale = sign * magnitude
        return residual_scale

    def _apply_bilinear_residual(
        self, decoder_out: torch.Tensor, x0: torch.Tensor, xT: torch.Tensor, tau_: torch.Tensor
    ) -> torch.Tensor:
        if self.residual_clip is not None and self.residual_clip > 0:
            decoder_out = torch.clamp(decoder_out, -self.residual_clip, self.residual_clip)
        residual_scale = self._residual_scale_clipped()
        x_bilinear = (1.0 - tau_) * x0 + tau_ * xT
        return x_bilinear + residual_scale * decoder_out, x_bilinear, residual_scale

    def _run_interpolator(
        self, x0: torch.Tensor, xT: torch.Tensor, static: torch.Tensor | None, t_emb: torch.Tensor
    ) -> torch.Tensor:
        parts = [x0, xT]
        if self.n_static_features > 0:
            parts.append(static)
        x_in = torch.cat(parts, dim=1)
        return self.interpolator(self._crop_lat(x_in), t_emb)

    def _run_refiner(
        self, x_curr: torch.Tensor, x0: torch.Tensor, xT: torch.Tensor,
        static: torch.Tensor | None, t_emb: torch.Tensor,
    ) -> torch.Tensor:
        parts = [x_curr, x0, xT]
        if self.n_static_features > 0:
            parts.append(static)
        x_in = torch.cat(parts, dim=1)
        return self.refiner(self._crop_lat(x_in), t_emb)

    def forward(self, x0, xT, tau, cond, static=None):
        if tau.dim() == 1:
            tau_ = tau.view(-1, 1, 1, 1)
        elif tau.dim() == 2:
            tau_ = tau.view(tau.size(0), 1, 1, 1)
        else:
            tau_ = tau

        t_emb = self.time_mlp(tau.view(-1))
        # Align t_emb batch with x0 batch (multi-tau path).
        if t_emb.size(0) != x0.size(0):
            if x0.size(0) % t_emb.size(0) == 0:
                t_emb = t_emb.repeat_interleave(x0.size(0) // t_emb.size(0), dim=0)
            else:
                raise ValueError(
                    f"tau batch {t_emb.size(0)} cannot align with x0 batch {x0.size(0)}"
                )

        static_b = self._expand_static(static, x0.size(0)) if self.n_static_features > 0 else None
        if self.n_static_features > 0 and static_b is None:
            raise ValueError(
                f"Model requires static (n_static_features={self.n_static_features}), got None"
            )

        # Stage 1: interpolator I_φ → initial decoder output (interp_out).
        interp_out_raw = self._run_interpolator(x0, xT, static_b, t_emb)
        interp_out = F.interpolate(
            interp_out_raw, size=x0.shape[-2:], mode="bilinear", align_corners=False
        )
        x_interp, x_bilinear, residual_scale = self._apply_bilinear_residual(
            interp_out, x0, xT, tau_
        )

        # Stage 2: refiner R_θ. Number of refine passes:
        #   - training: n_train_refine_steps (typically 1)
        #   - inference: n_inference_steps (typically 5)
        n_refine = self.n_train_refine_steps if self.training else self.n_inference_steps

        x_curr = x_interp
        refine_outputs: list[torch.Tensor] = []
        for _ in range(n_refine):
            ref_out_raw = self._run_refiner(x_curr, x0, xT, static_b, t_emb)
            ref_out = F.interpolate(
                ref_out_raw, size=x0.shape[-2:], mode="bilinear", align_corners=False
            )
            x_next, _, _ = self._apply_bilinear_residual(ref_out, x0, xT, tau_)
            refine_outputs.append(x_next)
            x_curr = x_next

        x_hat = x_curr  # final = bilinear-residual after K refinements (or interp if K=0)

        return x_hat, {
            "z_tau": None,
            "d0": None,
            "d1": None,
            "decoder_out": interp_out,
            "x_bilinear": x_bilinear,
            "residual_scale": residual_scale.detach(),
            "x_interp": x_interp,            # I_φ output (post-residual wrap)
            "refine_outputs": refine_outputs,  # R_θ outputs at each iteration
            "n_refine_steps": n_refine,
        }
