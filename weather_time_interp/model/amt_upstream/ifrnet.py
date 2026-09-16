import torch
import torch.nn as nn

from .flow_utils import (
    SphereConv2d,
    SphereConvTranspose2d,
    spherical_resize,
    warp,
)


def resize(x, scale_factor):
    return spherical_resize(x, scale_factor)


def match_spatial_size(x, output_size):
    if output_size is None:
        return x
    target_h, target_w = (int(value) for value in output_size)
    if x.size(-2) < target_h or x.size(-1) < target_w:
        raise ValueError(
            "decoder output is smaller than the requested skip grid"
        )
    return x[..., :target_h, :target_w]


def convrelu(
    in_channels,
    out_channels,
    kernel_size=3,
    stride=1,
    padding=1,
    dilation=1,
    groups=1,
    bias=True,
    pole_parity=None,
):
    return nn.Sequential(
        SphereConv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias=bias,
            pole_parity=pole_parity,
        ),
        nn.PReLU(out_channels)
    )

class ResBlock(nn.Module):
    def __init__(self, in_channels, side_channels, bias=True):
        super(ResBlock, self).__init__()
        self.side_channels = side_channels
        self.conv1 = nn.Sequential(
            SphereConv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=bias),
            nn.PReLU(in_channels)
        )
        self.conv2 = nn.Sequential(
            SphereConv2d(side_channels, side_channels, kernel_size=3, stride=1, padding=1, bias=bias),
            nn.PReLU(side_channels)
        )
        self.conv3 = nn.Sequential(
            SphereConv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=bias),
            nn.PReLU(in_channels)
        )
        self.conv4 = nn.Sequential(
            SphereConv2d(side_channels, side_channels, kernel_size=3, stride=1, padding=1, bias=bias),
            nn.PReLU(side_channels)
        )
        self.conv5 = SphereConv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=bias)
        self.prelu = nn.PReLU(in_channels)

    def forward(self, x):
        out = self.conv1(x)

        res_feat = out[:, :-self.side_channels, ...]
        side_feat = out[:, -self.side_channels:, :, :]
        side_feat = self.conv2(side_feat)
        out = self.conv3(torch.cat([res_feat, side_feat], 1))

        res_feat = out[:, :-self.side_channels, ...]
        side_feat = out[:, -self.side_channels:, :, :]
        side_feat = self.conv4(side_feat)
        out = self.conv5(torch.cat([res_feat, side_feat], 1))

        out = self.prelu(x + out)
        return out
    
class Encoder(nn.Module):
    def __init__(
        self,
        channels,
        large=False,
        in_channels=3,
        input_pole_parity=None,
    ):
        super(Encoder, self).__init__()
        self.channels = channels        
        prev_ch = in_channels
        for idx, ch in enumerate(channels, 1):
            k = 7 if large and idx == 1 else 3
            p = 3 if k ==7 else 1
            self.register_module(f'pyramid{idx}', 
            nn.Sequential(
                convrelu(
                    prev_ch,
                    ch,
                    k,
                    2,
                    p,
                    pole_parity=(
                        input_pole_parity if idx == 1 else None
                    ),
                ),
                convrelu(ch, ch, 3, 1, 1)
            ))
            prev_ch = ch
                
    def forward(self, in_x):
        fs = []
        for idx in range(len(self.channels)):
            out_x = getattr(self, f'pyramid{idx+1}')(in_x)
            fs.append(out_x)
            in_x = out_x
        return fs
    
class InitDecoder(nn.Module):
    def __init__(self, in_ch, out_ch, skip_ch) -> None:
        super().__init__()
        self.convblock = nn.Sequential(
            convrelu(in_ch*2+1, in_ch*2), 
            ResBlock(in_ch*2, skip_ch), 
            SphereConvTranspose2d(
                in_ch*2, out_ch+4, 4, 2, 1, bias=True
            )
        )
    def forward(self, f0, f1, embt, output_size=None):
        h, w = f0.shape[2:]
        embt = embt.repeat(1, 1, h, w)
        out = self.convblock(torch.cat([f0, f1, embt], 1))
        out = match_spatial_size(out, output_size)
        flow0, flow1 = torch.chunk(out[:, :4, ...], 2, 1)
        ft_ = out[:, 4:, ...]
        return flow0, flow1, ft_
    
class IntermediateDecoder(nn.Module):
    def __init__(self, in_ch, out_ch, skip_ch) -> None:
        super().__init__()
        self.convblock = nn.Sequential(
            convrelu(in_ch*3+4, in_ch*3), 
            ResBlock(in_ch*3, skip_ch), 
            SphereConvTranspose2d(
                in_ch*3, out_ch+4, 4, 2, 1, bias=True
            )
        )
    def forward(
        self,
        ft_,
        f0,
        f1,
        flow0_in,
        flow1_in,
        output_size=None,
    ):
        f0_warp = warp(f0, flow0_in)
        f1_warp = warp(f1, flow1_in)
        f_in = torch.cat([ft_, f0_warp, f1_warp, flow0_in, flow1_in], 1)
        out = self.convblock(f_in)
        out = match_spatial_size(out, output_size)
        flow0, flow1 = torch.chunk(out[:, :4, ...], 2, 1)
        ft_ = out[:, 4:, ...]
        flow0 = flow0 + 2.0 * resize(flow0_in, scale_factor=2.0)
        flow1 = flow1 + 2.0 * resize(flow1_in, scale_factor=2.0)
        return flow0, flow1, ft_
