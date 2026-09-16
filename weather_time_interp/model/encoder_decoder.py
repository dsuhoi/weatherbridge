import torch
import torch.nn as nn

from .sphere_conv import SphereConv2d
from .components import ResBlock, DCDownBlock2d, DCUpBlock2d


class WeatherEncoder(nn.Module):
    """ResNet-style encoder for weather fields."""

    def __init__(
        self,
        in_channels: int = 5,
        latent_channels: int = 64,
        block_out_channels: tuple = (128, 128, 256, 256),
        layers_per_block: tuple = (2, 2, 2, 2),
        downsample_block_type: str = "pixel_unshuffle",
    ):
        super().__init__()

        assert len(block_out_channels) == len(layers_per_block), (
            "block_out_channels and layers_per_block must have the same length"
        )
        num_blocks = len(block_out_channels)

        self.conv_in = SphereConv2d(
            in_channels,
            block_out_channels[0],
            kernel_size=3,
            stride=1,
            padding=1,
            padding_mode="circular",
        )

        self.down_blocks = nn.ModuleList()
        for i, (out_ch, num_layers) in enumerate(zip(block_out_channels, layers_per_block)):
            for _ in range(num_layers):
                self.down_blocks.append(
                    ResBlock(
                        in_channels=out_ch,
                        out_channels=out_ch,
                        norm_type="rms_norm",
                        act_fn="silu",
                        temb_channels=None,
                    )
                )

            if i < num_blocks - 1:
                self.down_blocks.append(
                    DCDownBlock2d(
                        in_channels=out_ch,
                        out_channels=block_out_channels[i + 1],
                        downsample=(downsample_block_type == "pixel_unshuffle"),
                        shortcut=True,
                    )
                )

        self.conv_out = SphereConv2d(
            block_out_channels[-1],
            latent_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            padding_mode="circular",
        )

    @staticmethod
    def _trim_to_even_spatial(x: torch.Tensor) -> torch.Tensor:
        if x.size(-2) % 2 != 0:
            x = x[:, :, :-1, :]
        if x.size(-1) % 2 != 0:
            x = x[:, :, :, :-1]
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden_states = self._trim_to_even_spatial(self.conv_in(x))
        for block in self.down_blocks:
            if isinstance(block, DCDownBlock2d) and block.downsample:
                hidden_states = self._trim_to_even_spatial(hidden_states)
            hidden_states = block(hidden_states, temb=None)
            hidden_states = self._trim_to_even_spatial(hidden_states)
        return self.conv_out(self._trim_to_even_spatial(hidden_states))


class WeatherDecoder(nn.Module):
    """ResNet-style decoder for weather fields."""

    def __init__(
        self,
        out_channels: int = 5,
        latent_channels: int = 64,
        block_out_channels: tuple = (128, 128, 256, 256),
        layers_per_block: tuple = (2, 2, 2, 2),
        upsample_block_type: str = "pixel_shuffle",
    ):
        super().__init__()

        assert len(block_out_channels) == len(layers_per_block), (
            "block_out_channels and layers_per_block must have the same length"
        )
        num_blocks = len(block_out_channels)

        self.conv_in = SphereConv2d(
            latent_channels,
            block_out_channels[-1],
            kernel_size=3,
            stride=1,
            padding=1,
            padding_mode="circular",
        )

        self.up_blocks = nn.ModuleList()
        for i in reversed(range(num_blocks)):
            out_ch = block_out_channels[i]

            if i < num_blocks - 1:
                self.up_blocks.append(
                    DCUpBlock2d(
                        in_channels=block_out_channels[i + 1],
                        out_channels=out_ch,
                        interpolate=(upsample_block_type == "interpolate"),
                        shortcut=True,
                    )
                )

            for _ in range(layers_per_block[i]):
                self.up_blocks.append(
                    ResBlock(
                        in_channels=out_ch,
                        out_channels=out_ch,
                        norm_type="rms_norm",
                        act_fn="silu",
                        temb_channels=None,
                    )
                )

        self.conv_out = SphereConv2d(
            block_out_channels[0],
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            padding_mode="circular",
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        hidden_states = self.conv_in(z)
        for block in self.up_blocks:
            hidden_states = block(hidden_states, temb=None)
        return self.conv_out(hidden_states)
