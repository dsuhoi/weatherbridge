"""Modulated Adaptive Fourier Neural Operator (ModAFNO) baseline.

Standalone re-implementation of NVIDIA PhysicsNeMo ModAFNO
(`physicsnemo/models/afno/modafno.py`, Apache-2.0). Removes the physicsnemo
runtime dependency (Module, ModelMetaData, fft helpers) so the model is usable
directly inside our pipeline without pulling the whole framework.

Paper: Leinonen et al., "Modulated Adaptive Fourier Neural Operators for
Temporal Interpolation of Weather Forecasts", arXiv:2410.18904 (2024).

Architecture summary:
- Patch embed (Conv2d stride=patch_size) → (B, N, D) tokens
- Add learnable positional embedding
- Add additive bias from time embedding (sinusoidal → MLP)
- Reshape to (B, h, w, D) and pass through `depth` ModAFNO blocks
- Each block: LN → ModAFNO2DLayer (FFT spectral conv with FiLM in freq domain)
  → +residual → LN → ModAFNOMlp (FiLM-modulated MLP) → +residual
- Linear head → unfold patches → (B, C_out, H, W)

Adapter `WeatherModAFNOResidualLinearModel` wraps this in our standard
residual-to-bilinear pipeline and matches the signature used by
WeatherSFNOResidualLinearModel / WeatherDCAEResidualLinearModel.
"""
from __future__ import annotations

import math
from functools import partial
from typing import List, Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ──────────────────────────── Time embedding ────────────────────────────


class SinusoidalTimestepEmbedding(nn.Module):
    """Sinusoidal timestep embedding identical to DDPM/ADM-style projections."""

    def __init__(self, dim: int, max_period: float = 10000.0):
        super().__init__()
        if dim % 2 != 0:
            raise ValueError(f"SinusoidalTimestepEmbedding dim must be even, got {dim}")
        self.dim = dim
        self.max_period = float(max_period)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        if t.dim() > 1:
            t = t.view(t.size(0))
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(self.max_period)
            * torch.arange(half, dtype=torch.float32, device=t.device)
            / half
        )
        args = t.float().unsqueeze(-1) * freqs.unsqueeze(0)
        return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


class ModEmbedNet(nn.Module):
    """Time → modulation embedding (sinusoidal + MLP, dim=mod_dim)."""

    def __init__(
        self,
        max_time: float = 1.0,
        dim: int = 64,
        depth: int = 1,
        activation_cls=nn.GELU,
    ):
        super().__init__()
        self.max_time = float(max_time)
        self.embed = SinusoidalTimestepEmbedding(dim)
        layers: list[nn.Module] = []
        for _ in range(depth):
            layers.extend([nn.Linear(dim, dim), activation_cls()])
        self.mlp = nn.Sequential(*layers) if layers else nn.Identity()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t = t / self.max_time
        return self.mlp(self.embed(t))


# ──────────────────────────── Patch embed ────────────────────────────


class AFNOPatchEmbed(nn.Module):
    """Conv2d-based patch embed (stride = patch size). Outputs (B, N, D)."""

    def __init__(
        self,
        inp_shape: List[int],
        in_channels: int,
        patch_size: List[int] = (2, 2),
        embed_dim: int = 256,
    ):
        super().__init__()
        if len(inp_shape) != 2 or len(patch_size) != 2:
            raise ValueError("inp_shape and patch_size must each be length-2")
        if inp_shape[0] % patch_size[0] != 0 or inp_shape[1] % patch_size[1] != 0:
            raise ValueError(
                f"inp_shape {inp_shape} must be divisible by patch_size {patch_size}"
            )
        self.inp_shape = list(inp_shape)
        self.patch_size = list(patch_size)
        self.num_patches = (inp_shape[0] // patch_size[0]) * (inp_shape[1] // patch_size[1])
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x).flatten(2).transpose(1, 2)


# ──────────────────────────── Scale-shift helper ────────────────────────────


class ScaleShiftMlp(nn.Module):
    """Two-layer MLP that produces (1+scale, shift) from a modulation vector."""

    def __init__(self, in_features: int, out_features: int, hidden_features: Optional[int] = None):
        super().__init__()
        hidden = hidden_features if hidden_features is not None else out_features * 2
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.GELU(),
            nn.Linear(hidden, out_features * 2),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        scale, shift = torch.chunk(self.net(x), 2, dim=-1)
        return 1.0 + scale, shift


# ──────────────────────────── MLP blocks ────────────────────────────


class AFNOMlp(nn.Module):
    """Plain 2-layer GELU MLP used inside non-modulated AFNO blocks."""

    def __init__(self, in_features: int, latent_features: int, out_features: int, drop: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(in_features, latent_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(latent_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class ModAFNOMlp(AFNOMlp):
    """AFNOMlp + per-block FiLM applied between fc1 and the activation."""

    def __init__(
        self,
        in_features: int,
        latent_features: int,
        out_features: int,
        mod_features: int,
        drop: float = 0.0,
    ):
        super().__init__(in_features, latent_features, out_features, drop=drop)
        self.scale_shift = ScaleShiftMlp(mod_features, latent_features)

    def forward(self, x: torch.Tensor, mod_embed: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        scale, shift = self.scale_shift(mod_embed)
        # x is (B, h, w, D_in); reshape scale/shift to (B, 1, 1, latent)
        scale = scale.view(scale.size(0), *(1,) * (x.dim() - 2), scale.size(-1))
        shift = shift.view(shift.size(0), *(1,) * (x.dim() - 2), shift.size(-1))
        x = self.fc1(x)
        x = x * scale + shift
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


# ──────────────────────────── 2D spectral layers ────────────────────────────


class AFNO2DLayer(nn.Module):
    """Block-diagonal complex weight matmul in 2D Fourier domain (no modulation)."""

    def __init__(
        self,
        hidden_size: int,
        num_blocks: int = 8,
        sparsity_threshold: float = 0.01,
        hard_thresholding_fraction: float = 1.0,
        hidden_size_factor: int = 1,
    ):
        super().__init__()
        if hidden_size % num_blocks != 0:
            raise ValueError(
                f"hidden_size {hidden_size} must be divisible by num_blocks {num_blocks}"
            )
        self.hidden_size = hidden_size
        self.sparsity_threshold = sparsity_threshold
        self.num_blocks = num_blocks
        self.block_size = hidden_size // num_blocks
        self.hard_thresholding_fraction = hard_thresholding_fraction
        self.hidden_size_factor = hidden_size_factor

        scale = 0.02
        self.w1 = nn.Parameter(scale * torch.randn(2, num_blocks, self.block_size, self.block_size * hidden_size_factor))
        self.b1 = nn.Parameter(scale * torch.randn(2, num_blocks, self.block_size * hidden_size_factor))
        self.w2 = nn.Parameter(scale * torch.randn(2, num_blocks, self.block_size * hidden_size_factor, self.block_size))
        self.b2 = nn.Parameter(scale * torch.randn(2, num_blocks, self.block_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bias = x
        dtype = x.dtype
        x = x.float()
        B, H, W, C = x.shape

        x_hat = torch.fft.rfft2(x, dim=(1, 2), norm="ortho")
        x_real = x_hat.real.reshape(B, H, W // 2 + 1, self.num_blocks, self.block_size)
        x_imag = x_hat.imag.reshape(B, H, W // 2 + 1, self.num_blocks, self.block_size)

        out_shape = (B, H, W // 2 + 1, self.num_blocks, self.block_size * self.hidden_size_factor)
        o1_real = torch.zeros(out_shape, device=x.device)
        o1_imag = torch.zeros(out_shape, device=x.device)
        o2 = torch.zeros(x_real.shape + (2,), device=x.device)

        total_modes = H // 2 + 1
        kept = int(total_modes * self.hard_thresholding_fraction)
        sl_y = slice(total_modes - kept, total_modes + kept)
        sl_x = slice(None, kept)

        o1_real[:, sl_y, sl_x] = F.relu(
            torch.einsum("nyxbi,bio->nyxbo", x_real[:, sl_y, sl_x], self.w1[0])
            - torch.einsum("nyxbi,bio->nyxbo", x_imag[:, sl_y, sl_x], self.w1[1])
            + self.b1[0]
        )
        o1_imag[:, sl_y, sl_x] = F.relu(
            torch.einsum("nyxbi,bio->nyxbo", x_imag[:, sl_y, sl_x], self.w1[0])
            + torch.einsum("nyxbi,bio->nyxbo", x_real[:, sl_y, sl_x], self.w1[1])
            + self.b1[1]
        )

        o2[:, sl_y, sl_x, ..., 0] = (
            torch.einsum("nyxbi,bio->nyxbo", o1_real[:, sl_y, sl_x], self.w2[0])
            - torch.einsum("nyxbi,bio->nyxbo", o1_imag[:, sl_y, sl_x], self.w2[1])
            + self.b2[0]
        )
        o2[:, sl_y, sl_x, ..., 1] = (
            torch.einsum("nyxbi,bio->nyxbo", o1_imag[:, sl_y, sl_x], self.w2[0])
            + torch.einsum("nyxbi,bio->nyxbo", o1_real[:, sl_y, sl_x], self.w2[1])
            + self.b2[1]
        )

        x_thresh = F.softshrink(o2, lambd=self.sparsity_threshold)
        x_complex = torch.complex(x_thresh[..., 0], x_thresh[..., 1])
        x_complex = x_complex.reshape(B, H, W // 2 + 1, C)
        x = torch.fft.irfft2(x_complex, s=(H, W), dim=(1, 2), norm="ortho")
        return x.to(dtype) + bias


class ModAFNO2DLayer(AFNO2DLayer):
    """AFNO2DLayer + FiLM applied after w1 in the spectral domain."""

    def __init__(
        self,
        hidden_size: int,
        mod_features: int,
        num_blocks: int = 8,
        sparsity_threshold: float = 0.01,
        hard_thresholding_fraction: float = 1.0,
        hidden_size_factor: int = 1,
        scale_shift_mode: Literal["complex", "real"] = "complex",
    ):
        super().__init__(
            hidden_size=hidden_size,
            num_blocks=num_blocks,
            sparsity_threshold=sparsity_threshold,
            hard_thresholding_fraction=hard_thresholding_fraction,
            hidden_size_factor=hidden_size_factor,
        )
        if scale_shift_mode not in ("complex", "real"):
            raise ValueError("scale_shift_mode must be 'real' or 'complex'")
        self.scale_shift_mode = scale_shift_mode
        channel_mul = 1 if scale_shift_mode == "real" else 2
        self.channel_mul = channel_mul
        self.scale_shift = ScaleShiftMlp(
            mod_features,
            num_blocks * self.block_size * hidden_size_factor * channel_mul,
        )

    def forward(self, x: torch.Tensor, mod_embed: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        bias = x
        dtype = x.dtype
        x = x.float()
        B, H, W, C = x.shape

        x_hat = torch.fft.rfft2(x, dim=(1, 2), norm="ortho")
        x_real = x_hat.real.reshape(B, H, W // 2 + 1, self.num_blocks, self.block_size)
        x_imag = x_hat.imag.reshape(B, H, W // 2 + 1, self.num_blocks, self.block_size)

        latent = self.block_size * self.hidden_size_factor
        out_shape = (B, H, W // 2 + 1, self.num_blocks, latent)
        scale_shift_shape = (B, self.channel_mul, 1, self.num_blocks, latent)

        o1_real = torch.zeros(out_shape, device=x.device)
        o1_imag = torch.zeros(out_shape, device=x.device)
        o2 = torch.zeros(x_real.shape + (2,), device=x.device)

        total_modes = min(H, W) // 2 + 1
        kept = int(total_modes * self.hard_thresholding_fraction)
        sl_y = slice(total_modes - kept, total_modes + kept)
        sl_x = slice(None, kept)

        o1_re = (
            torch.einsum("nyxbi,bio->nyxbo", x_real[:, sl_y, sl_x], self.w1[0])
            - torch.einsum("nyxbi,bio->nyxbo", x_imag[:, sl_y, sl_x], self.w1[1])
            + self.b1[0]
        )
        o1_im = (
            torch.einsum("nyxbi,bio->nyxbo", x_imag[:, sl_y, sl_x], self.w1[0])
            + torch.einsum("nyxbi,bio->nyxbo", x_real[:, sl_y, sl_x], self.w1[1])
            + self.b1[1]
        )

        scale, shift = self.scale_shift(mod_embed)
        scale = scale.view(*scale_shift_shape)
        shift = shift.view(*scale_shift_shape)
        if self.scale_shift_mode == "real":
            o1_re = o1_re * scale + shift
            o1_im = o1_im * scale + shift
        else:  # "complex"
            scale_re, scale_im = torch.chunk(scale, 2, dim=1)
            shift_re, shift_im = torch.chunk(shift, 2, dim=1)
            # Broadcast: o1_* is (B, H_kept, W_kept, num_blocks, latent);
            # scale_*/shift_* are (B, 1, 1, num_blocks, latent) after chunk.
            new_re = o1_re * scale_re - o1_im * scale_im + shift_re
            new_im = o1_im * scale_re + o1_re * scale_im + shift_im
            o1_re, o1_im = new_re, new_im

        o1_real[:, sl_y, sl_x] = F.relu(o1_re)
        o1_imag[:, sl_y, sl_x] = F.relu(o1_im)

        o2[:, sl_y, sl_x, ..., 0] = (
            torch.einsum("nyxbi,bio->nyxbo", o1_real[:, sl_y, sl_x], self.w2[0])
            - torch.einsum("nyxbi,bio->nyxbo", o1_imag[:, sl_y, sl_x], self.w2[1])
            + self.b2[0]
        )
        o2[:, sl_y, sl_x, ..., 1] = (
            torch.einsum("nyxbi,bio->nyxbo", o1_imag[:, sl_y, sl_x], self.w2[0])
            + torch.einsum("nyxbi,bio->nyxbo", o1_real[:, sl_y, sl_x], self.w2[1])
            + self.b2[1]
        )

        x_thresh = F.softshrink(o2, lambd=self.sparsity_threshold)
        x_complex = torch.complex(x_thresh[..., 0], x_thresh[..., 1])
        x_complex = x_complex.reshape(B, H, W // 2 + 1, C)
        x = torch.fft.irfft2(x_complex, s=(H, W), dim=(1, 2), norm="ortho")
        return x.to(dtype) + bias


# ──────────────────────────── ModAFNO block & core ────────────────────────────


class ModAFNOBlock(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        mod_dim: int,
        num_blocks: int = 8,
        mlp_ratio: float = 2.0,
        drop: float = 0.0,
        norm_layer=None,
        sparsity_threshold: float = 0.01,
        hard_thresholding_fraction: float = 1.0,
        modulate_filter: bool = True,
        modulate_mlp: bool = True,
        scale_shift_mode: Literal["complex", "real"] = "complex",
    ):
        super().__init__()
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        self.norm1 = norm_layer(embed_dim)
        self.modulate_filter = modulate_filter
        self.modulate_mlp = modulate_mlp

        if modulate_filter:
            self.filter = ModAFNO2DLayer(
                embed_dim, mod_dim, num_blocks,
                sparsity_threshold, hard_thresholding_fraction,
                scale_shift_mode=scale_shift_mode,
            )
        else:
            self.filter = AFNO2DLayer(embed_dim, num_blocks, sparsity_threshold, hard_thresholding_fraction)

        self.norm2 = norm_layer(embed_dim)
        latent = int(embed_dim * mlp_ratio)
        if modulate_mlp:
            self.mlp = ModAFNOMlp(embed_dim, latent, embed_dim, mod_features=mod_dim, drop=drop)
        else:
            self.mlp = AFNOMlp(embed_dim, latent, embed_dim, drop=drop)

    def forward(self, x: torch.Tensor, mod_embed: torch.Tensor) -> torch.Tensor:
        residual = x
        h = self.norm1(x)
        h = self.filter(h, mod_embed) if self.modulate_filter else self.filter(h)
        x = h + residual
        residual = x
        h = self.norm2(x)
        h = self.mlp(h, mod_embed) if self.modulate_mlp else self.mlp(h)
        return h + residual


class ModAFNO(nn.Module):
    """Full ModAFNO core (without our residual-to-bilinear wrapper)."""

    def __init__(
        self,
        inp_shape: List[int],
        in_channels: int,
        out_channels: int,
        patch_size: List[int] = (2, 2),
        embed_dim: int = 256,
        mod_dim: int = 64,
        depth: int = 8,
        mlp_ratio: float = 2.0,
        drop_rate: float = 0.0,
        num_blocks: int = 8,
        sparsity_threshold: float = 0.01,
        hard_thresholding_fraction: float = 1.0,
        modulate_filter: bool = True,
        modulate_mlp: bool = True,
        scale_shift_mode: Literal["complex", "real"] = "complex",
        max_time: float = 1.0,
        embed_depth: int = 1,
    ):
        super().__init__()
        self.in_chans = in_channels
        self.out_chans = out_channels
        self.inp_shape = list(inp_shape)
        self.patch_size = list(patch_size)
        self.embed_dim = embed_dim
        self.h = inp_shape[0] // patch_size[0]
        self.w = inp_shape[1] // patch_size[1]

        self.patch_embed = AFNOPatchEmbed(inp_shape, in_channels, patch_size, embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.num_patches, embed_dim))
        self.pos_drop = nn.Dropout(drop_rate)

        self.mod_embed_net = ModEmbedNet(max_time=max_time, dim=mod_dim, depth=embed_depth)
        self.mod_additive_proj = nn.Linear(mod_dim, embed_dim)

        self.blocks = nn.ModuleList(
            [
                ModAFNOBlock(
                    embed_dim=embed_dim,
                    mod_dim=mod_dim,
                    num_blocks=num_blocks,
                    mlp_ratio=mlp_ratio,
                    drop=drop_rate,
                    sparsity_threshold=sparsity_threshold,
                    hard_thresholding_fraction=hard_thresholding_fraction,
                    modulate_filter=modulate_filter,
                    modulate_mlp=modulate_mlp,
                    scale_shift_mode=scale_shift_mode,
                )
                for _ in range(depth)
            ]
        )

        self.head = nn.Linear(embed_dim, out_channels * patch_size[0] * patch_size[1], bias=False)

        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m: nn.Module):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x: torch.Tensor, mod: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        if mod.dim() == 1:
            mod = mod.view(B, 1)

        x = self.patch_embed(x)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        mod_embed = self.mod_embed_net(mod)
        x = x + self.mod_additive_proj(mod_embed).unsqueeze(1)

        x = x.reshape(B, self.h, self.w, self.embed_dim)
        for blk in self.blocks:
            x = blk(x, mod_embed=mod_embed)

        # Head + unfold patches → (B, C_out, H, W)
        x = self.head(x)  # (B, h, w, C_out * pH * pW)
        out = x.view(B, self.h, self.w, self.patch_size[0], self.patch_size[1], self.out_chans)
        out = out.permute(0, 5, 1, 3, 2, 4).contiguous()
        out = out.reshape(B, self.out_chans, self.inp_shape[0], self.inp_shape[1])
        return out


# ──────────────────────────── Adapter (matches our pipeline) ────────────────────────────


class WeatherModAFNOResidualLinearModel(nn.Module):
    """ModAFNO core + residual-to-bilinear pipeline.

    Input pad: (181, 360) → (182, 360) so it divides by patch_size=(2, 2).
    Output crop: (182, 360) → (181, 360).

    Forward signature matches sibling baselines (sfno, dcae, true_unet):
        x_hat, aux = model(x0, xT, tau, cond, static=None)
    where `cond` is unused (modulation comes from `tau`).
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
        residual_scale_init: float = 0.10,
        residual_scale_learnable: bool = True,
        residual_clip: Optional[float] = None,
        residual_scale_floor: float = 0.0,
        direct_prediction: bool = False,
        n_static_features: int = 0,
        # input shape after padding:
        inp_shape: tuple = (182, 360),
        native_shape: tuple = (181, 360),
    ):
        super().__init__()
        self.residual_clip = residual_clip
        self.residual_scale_floor = float(residual_scale_floor)
        self.direct_prediction = bool(direct_prediction)
        self.n_static_features = int(n_static_features)
        self.inp_shape = tuple(inp_shape)
        self.native_shape = tuple(native_shape)
        self.pad_top = (inp_shape[0] - native_shape[0]) // 2
        self.pad_bot = inp_shape[0] - native_shape[0] - self.pad_top
        # Persist remaining init kwargs for introspection-based loaders.
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.embed_dim = int(embed_dim)
        self.mod_dim = int(mod_dim)
        self.depth = int(depth)
        self.patch_size = tuple(patch_size)
        self.mlp_ratio = float(mlp_ratio)
        self.num_blocks = int(num_blocks)
        self.drop_rate = float(drop_rate)
        self.sparsity_threshold = float(sparsity_threshold)
        self.hard_thresholding_fraction = float(hard_thresholding_fraction)
        self.modulate_filter = bool(modulate_filter)
        self.modulate_mlp = bool(modulate_mlp)
        self.scale_shift_mode = str(scale_shift_mode)
        self.residual_scale_init = float(residual_scale_init)
        self.residual_scale_learnable = bool(residual_scale_learnable)

        afno_in = in_channels * 2 + self.n_static_features  # cat(x0, xT, [static])
        self.modafno = ModAFNO(
            inp_shape=list(inp_shape),
            in_channels=afno_in,
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
            max_time=1.0,
        )

        init_scale = torch.tensor(float(residual_scale_init), dtype=torch.float32)
        if residual_scale_learnable and not self.direct_prediction:
            self.residual_scale = nn.Parameter(init_scale)
        else:
            self.register_buffer("residual_scale", init_scale)

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

    def forward(self, x0, xT, tau, cond, static=None):
        if tau.dim() == 1:
            tau_ = tau.view(-1, 1, 1, 1)
        elif tau.dim() == 2:
            tau_ = tau.view(tau.size(0), 1, 1, 1)
        else:
            tau_ = tau

        parts = [x0, xT]
        if self.n_static_features > 0:
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
            parts.append(static)
        x_in = torch.cat(parts, dim=1)
        x_in_padded = self._pad_lat(x_in)

        mod = tau_.view(tau_.size(0), 1)  # (B, 1) ∈ [0, 1]
        decoder_out_padded = self.modafno(x_in_padded, mod)
        decoder_out = self._crop_lat(decoder_out_padded)

        if self.residual_clip is not None and self.residual_clip > 0:
            decoder_out = torch.clamp(decoder_out, -self.residual_clip, self.residual_clip)

        residual_scale = torch.tanh(self.residual_scale)
        if self.residual_scale_floor > 0.0:
            sign = torch.where(
                residual_scale >= 0,
                torch.ones_like(residual_scale),
                -torch.ones_like(residual_scale),
            )
            magnitude = residual_scale.abs().clamp(min=self.residual_scale_floor)
            residual_scale = sign * magnitude

        x_bilinear = (1.0 - tau_) * x0 + tau_ * xT
        if self.direct_prediction:
            x_hat = decoder_out
        else:
            x_hat = x_bilinear + residual_scale * decoder_out

        return x_hat, {
            "z_tau": None,
            "d0": None,
            "d1": None,
            "decoder_out": decoder_out,
            "x_bilinear": x_bilinear,
            "residual_scale": residual_scale.detach(),
        }


# ──────────────────────────── Official-vendored adapter ────────────────────────────


class WeatherModAFNOOfficialResidualLinearModel(nn.Module):
    """Same wrapper as `WeatherModAFNOResidualLinearModel`, but the inner ModAFNO
    is the **bit-exact vendored copy** from NVIDIA/physicsnemo (Apache-2.0).

    Use this for new runs to guarantee the architecture matches the paper
    (Leinonen et al., arXiv:2410.18904). The old wrapper above keeps a
    standalone reproduction; checkpoints from one are NOT loadable into the
    other (state_dict keys differ in the scale-shift MLP nesting).

    Forward signature is identical:
        x_hat, aux = model(x0, xT, tau, cond, static=None)
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
        residual_scale_init: float = 0.10,
        residual_scale_learnable: bool = True,
        residual_clip: Optional[float] = None,
        residual_scale_floor: float = 0.0,
        direct_prediction: bool = False,
        n_static_features: int = 0,
        inp_shape: tuple = (182, 360),
        native_shape: tuple = (181, 360),
    ):
        super().__init__()
        from .physicsnemo_vendor import ModAFNO as VendorModAFNO

        self.residual_clip = residual_clip
        self.residual_scale_floor = float(residual_scale_floor)
        self.direct_prediction = bool(direct_prediction)
        self.n_static_features = int(n_static_features)
        self.inp_shape = tuple(inp_shape)
        self.native_shape = tuple(native_shape)
        self.pad_top = (inp_shape[0] - native_shape[0]) // 2
        self.pad_bot = inp_shape[0] - native_shape[0] - self.pad_top

        afno_in = in_channels * 2 + self.n_static_features
        self.modafno = VendorModAFNO(
            inp_shape=list(inp_shape),
            in_channels=afno_in,
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

        init_scale = torch.tensor(float(residual_scale_init), dtype=torch.float32)
        if residual_scale_learnable and not self.direct_prediction:
            self.residual_scale = nn.Parameter(init_scale)
        else:
            self.register_buffer("residual_scale", init_scale)

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

    def forward(self, x0, xT, tau, cond, static=None):
        if tau.dim() == 1:
            tau_ = tau.view(-1, 1, 1, 1)
        elif tau.dim() == 2:
            tau_ = tau.view(tau.size(0), 1, 1, 1)
        else:
            tau_ = tau

        parts = [x0, xT]
        if self.n_static_features > 0:
            if static is None:
                raise ValueError(
                    f"Model requires static (n_static={self.n_static_features}), got None"
                )
            if static.dim() == 3:
                static = static.unsqueeze(0).expand(x0.size(0), -1, -1, -1)
            elif static.size(0) != x0.size(0):
                if x0.size(0) % static.size(0) == 0:
                    n_tau = x0.size(0) // static.size(0)
                    static = static.repeat_interleave(n_tau, dim=0)
                else:
                    static = static.expand(x0.size(0), -1, -1, -1)
            parts.append(static)
        x_in = torch.cat(parts, dim=1)
        x_in_padded = self._pad_lat(x_in)

        # Vendored ModAFNO accepts mod as (B, 1) in [0, max_time=1.0]
        mod = tau_.view(tau_.size(0), 1)
        decoder_out_padded = self.modafno(x_in_padded, mod)
        decoder_out = self._crop_lat(decoder_out_padded)

        if self.residual_clip is not None and self.residual_clip > 0:
            decoder_out = torch.clamp(decoder_out, -self.residual_clip, self.residual_clip)

        residual_scale = torch.tanh(self.residual_scale)
        if self.residual_scale_floor > 0.0:
            sign = torch.where(
                residual_scale >= 0,
                torch.ones_like(residual_scale),
                -torch.ones_like(residual_scale),
            )
            magnitude = residual_scale.abs().clamp(min=self.residual_scale_floor)
            residual_scale = sign * magnitude

        x_bilinear = (1.0 - tau_) * x0 + tau_ * xT
        if self.direct_prediction:
            x_hat = decoder_out
        else:
            x_hat = x_bilinear + residual_scale * decoder_out

        return x_hat, {
            "z_tau": None,
            "d0": None,
            "d1": None,
            "decoder_out": decoder_out,
            "x_bilinear": x_bilinear,
            "residual_scale": residual_scale.detach(),
        }
