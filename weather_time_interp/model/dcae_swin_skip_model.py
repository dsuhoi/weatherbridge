"""DC-AE Skip + SwinV2 at deepest bottleneck stage (replacing EfficientViTBlock).

Motivation:
  DC-AE deepest stage uses EfficientViTBlock with Sana-style linear attention
  (lossy O(N) approximation). At bottleneck spatial 44×90, full attention is
  affordable. SwinV2 with W-MSA + SW-MSA gives exact local attention with
  shifted windows — better for capturing fine bottleneck structure than linear
  attention's low-rank approximation.

  Hybrid: ResBlock at shallow stages (CNN locality) + SwinV2 at deep stage
  (precise self-attention) + U-Net skip connections + FiLM time conditioning.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.swin_transformer import SwinTransformerBlockV2

from .dcae_adaln_skip_model import WeatherDCAEAdaLNSkipModel


class SwinV2BottleneckBlock(nn.Module):
    """SwinV2 W-MSA/SW-MSA block with DC-AE-compatible (NCHW, temb) interface."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        window_size: tuple = (4, 9),
        shift: bool = False,
        temb_channels: int | None = None,
        mlp_ratio: float = 4.0,
        attention_dropout: float = 0.0,
        stochastic_depth_prob: float = 0.0,
    ):
        super().__init__()
        shift_size = (window_size[0] // 2, window_size[1] // 2) if shift else (0, 0)
        self.swin = SwinTransformerBlockV2(
            dim=embed_dim,
            num_heads=num_heads,
            window_size=list(window_size),
            shift_size=list(shift_size),
            mlp_ratio=mlp_ratio,
            attention_dropout=attention_dropout,
            stochastic_depth_prob=stochastic_depth_prob,
        )
        if temb_channels is not None:
            self.film = nn.Linear(temb_channels, 2 * embed_dim)
            # zero-init → block starts as identity wrt time, learns gradually
            nn.init.zeros_(self.film.weight)
            nn.init.zeros_(self.film.bias)
        else:
            self.film = None

    def forward(self, x: torch.Tensor, temb: torch.Tensor | None = None) -> torch.Tensor:
        # NCHW → NHWC for SwinV2
        x_nhwc = x.permute(0, 2, 3, 1).contiguous()
        x_nhwc = self.swin(x_nhwc)
        if self.film is not None and temb is not None:
            scale_shift = self.film(temb)
            scale, shift = scale_shift.chunk(2, dim=-1)
            x_nhwc = x_nhwc * (1.0 + scale[:, None, None, :]) + shift[:, None, None, :]
        return x_nhwc.permute(0, 3, 1, 2).contiguous()


class WeatherDCAESwinSkipModel(WeatherDCAEAdaLNSkipModel):
    """DC-AE Skip + SwinV2 bottleneck. Identical to parent except deepest stage
    blocks are replaced with SwinV2BottleneckBlock (W-MSA alternating SW-MSA)."""

    def __init__(
        self,
        *args,
        bottleneck_window_size: tuple = (4, 9),
        bottleneck_num_heads: int = 8,
        bottleneck_drop_path: float = 0.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        bottleneck_dim = self.block_out_channels[-1]
        # layers_per_block default from parent: (2, 2, 2); deepest = 2 blocks
        n_bottleneck = 2  # match default layers_per_block[-1]

        # Encoder: last n_bottleneck entries in down_blocks are EfficientViT to swap
        enc_blocks = self.encoder.down_blocks
        for j in range(n_bottleneck):
            idx = len(enc_blocks) - n_bottleneck + j
            shift = (j % 2 == 1)
            enc_blocks[idx] = SwinV2BottleneckBlock(
                embed_dim=bottleneck_dim,
                num_heads=bottleneck_num_heads,
                window_size=bottleneck_window_size,
                shift=shift,
                temb_channels=self.time_emb_dim,
                stochastic_depth_prob=bottleneck_drop_path * j / max(n_bottleneck - 1, 1),
            )

        # Decoder: first n_bottleneck entries in up_blocks are EfficientViT
        dec_blocks = self.decoder.up_blocks
        for j in range(n_bottleneck):
            shift = (j % 2 == 1)
            dec_blocks[j] = SwinV2BottleneckBlock(
                embed_dim=bottleneck_dim,
                num_heads=bottleneck_num_heads,
                window_size=bottleneck_window_size,
                shift=shift,
                temb_channels=self.time_emb_dim,
                stochastic_depth_prob=bottleneck_drop_path * j / max(n_bottleneck - 1, 1),
            )
