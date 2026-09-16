from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .sphere_conv import SphereConv2d


class DCDownBlock2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        downsample: bool = False,
        shortcut: bool = True,
    ) -> None:
        super().__init__()

        self.downsample = downsample
        self.factor = 2
        self.stride = 1 if downsample else 2
        self.group_size = in_channels * self.factor**2 // out_channels
        self.shortcut = shortcut

        out_ratio = self.factor**2
        if downsample:
            assert out_channels % out_ratio == 0
            out_channels = out_channels // out_ratio

        self.conv = SphereConv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=self.stride,
            padding=1,
            padding_mode="circular",
        )

    def forward(self, hidden_states: torch.Tensor, temb=None) -> torch.Tensor:
        x = self.conv(hidden_states)
        if self.downsample:
            x = F.pixel_unshuffle(x, self.factor)

        if self.shortcut:
            y = F.pixel_unshuffle(hidden_states, self.factor)
            y = y.unflatten(1, (-1, self.group_size))
            y = y.mean(dim=2)
            hidden_states = x + y
        else:
            hidden_states = x

        return hidden_states


class DCUpBlock2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        interpolate: bool = False,
        shortcut: bool = True,
        interpolation_mode: str = "nearest",
    ) -> None:
        super().__init__()

        self.interpolate = interpolate
        self.interpolation_mode = interpolation_mode
        self.shortcut = shortcut
        self.factor = 2
        self.repeats = out_channels * self.factor**2 // in_channels

        out_ratio = self.factor**2

        if not interpolate:
            out_channels = out_channels * out_ratio

        self.conv = SphereConv2d(
            in_channels, out_channels, 3, 1, 1, padding_mode="circular"
        )

    def forward(self, hidden_states: torch.Tensor, temb=None) -> torch.Tensor:
        if self.interpolate:
            x = F.interpolate(
                hidden_states, scale_factor=self.factor, mode=self.interpolation_mode
            )
            x = self.conv(x)
        else:
            x = self.conv(hidden_states)
            x = F.pixel_shuffle(x, self.factor)

        if self.shortcut:
            y = hidden_states.repeat_interleave(self.repeats, dim=1)
            y = F.pixel_shuffle(y, self.factor)
            hidden_states = x + y
        else:
            hidden_states = x

        return hidden_states


class ResBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        norm_type: str = "batch_norm",
        act_fn: str = "relu6",
        temb_channels: Optional[int] = None,
    ) -> None:
        super().__init__()

        self.norm_type = norm_type
        self.nonlinearity = nn.SiLU()
        # self.nonlinearity = (
        #     get_activation(act_fn) if act_fn is not None else nn.Identity()
        # )
        self.conv1 = SphereConv2d(
            in_channels, in_channels, 3, 1, 1, padding_mode="circular"
        )
        self.conv2 = SphereConv2d(
            in_channels, out_channels, 3, 1, 1, bias=False, padding_mode="circular"
        )
        self.norm = nn.RMSNorm(out_channels)
        # get_normalization(norm_type, out_channels)

        if temb_channels is not None:
            self.time_emb_porj = nn.Linear(temb_channels, 2 * out_channels)
        else:
            self.time_emb_porj = None

    def forward(
        self, hidden_states: torch.Tensor, temb: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.conv1(hidden_states)
        hidden_states = self.nonlinearity(hidden_states)

        if self.time_emb_porj is not None:
            temb = self.nonlinearity(temb)
            temb = self.time_emb_porj(temb)[:, :, None, None]
            time_scale, time_shift = torch.chunk(temb, 2, dim=1)
            hidden_states = hidden_states * time_scale + time_shift

        hidden_states = self.conv2(hidden_states)

        if self.norm_type == "rms_norm":
            # move channel to the last dimension so we apply RMSnorm across channel dimension
            hidden_states = self.norm(hidden_states.movedim(1, -1)).movedim(-1, 1)
        else:
            hidden_states = self.norm(hidden_states)

        return hidden_states + residual
