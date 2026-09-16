"""WeatherAMT: AMT-L adapted to multivariate weather interpolation.

The transport core follows AMT-L (Li et al., CVPR 2023): bidirectional
all-pairs correlation, coarse-to-fine flow refinement, and multiple final
flow hypotheses. The weather adaptation changes only the field interface and
geometry-facing synthesis:

* 24 normalized prognostic fields replace RGB;
* feature extraction, refinement, warps, and lookup use spherical geometry;
* field warping additionally applies vector-component pole parity;
* static geography conditions synthesis without contaminating motion
  correlation;
* each field has its own residual while flow hypotheses remain shared;
* a smooth endpoint envelope anchors the prediction to x0 and x1.

The AMT-derived components live in ``amt_upstream`` with their upstream
CC BY-NC 4.0 license and provenance.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .amt_upstream.feat_enc import BasicEncoder
from .amt_upstream.flow_utils import SphereConv2d, spherical_pad
from .amt_upstream.ifrnet import (
    Encoder,
    InitDecoder,
    IntermediateDecoder,
    resize,
)
from .amt_upstream.multi_flow import MultiFlowDecoder, multi_flow_combine
from .amt_upstream.raft import BasicUpdateBlock, BidirCorrBlock, coords_grid
from .weatherbridge_flow_model import warp as spherical_field_warp


class WeatherAMTModel(nn.Module):
    """Capacity-matched AMT-L weather interpolator.

    ``forward(x0, x1, tau, cond=None, static=None)`` accepts normalized tau in
    ``[0, 1]`` and returns a 24-field tensor. Static geography conditions the
    synthesis pyramid, while the all-pairs correlation encoder sees only the
    prognostic anchors.
    """

    def __init__(
        self,
        in_channels: int = 24,
        out_channels: int = 24,
        n_static_features: int = 3,
        corr_radius: int = 3,
        corr_levels: int = 4,
        num_flows: int = 5,
        channels: tuple[int, int, int, int] = (48, 64, 72, 110),
        skip_channels: int = 48,
        endpoint_envelope: bool = True,
        max_field_displacement: float = 16.0,
    ):
        super().__init__()
        if in_channels != out_channels:
            raise ValueError("WeatherAMT requires matching input/output fields")
        if len(channels) != 4:
            raise ValueError("WeatherAMT requires four pyramid levels")
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.n_static_features = int(n_static_features)
        if self.n_static_features < 0:
            raise ValueError("n_static_features must be non-negative")
        self.radius = int(corr_radius)
        self.corr_levels = int(corr_levels)
        self.num_flows = int(num_flows)
        self.endpoint_envelope = bool(endpoint_envelope)
        self.max_field_displacement = float(max_field_displacement)

        field_pole_parity = torch.ones(out_channels)
        for index in (*range(4, 12), 21, 22):
            if index < out_channels:
                field_pole_parity[index] = -1.0
        self.register_buffer(
            "field_pole_parity",
            field_pole_parity,
            persistent=False,
        )
        static_pole_parity = torch.ones(self.n_static_features)
        self.register_buffer(
            "static_pole_parity",
            static_pole_parity,
            persistent=False,
        )
        encoder_pole_parity = torch.cat(
            (field_pole_parity, static_pole_parity)
        )

        self.feat_encoder = BasicEncoder(
            output_dim=128,
            norm_fn="instance",
            dropout=0.0,
            in_channels=in_channels,
            input_pole_parity=self.field_pole_parity,
        )
        self.encoder = Encoder(
            list(channels),
            large=True,
            in_channels=in_channels + self.n_static_features,
            input_pole_parity=encoder_pole_parity,
        )
        self.decoder4 = InitDecoder(channels[3], channels[2], skip_channels)
        self.decoder3 = IntermediateDecoder(
            channels[2], channels[1], skip_channels
        )
        self.decoder2 = IntermediateDecoder(
            channels[1], channels[0], skip_channels
        )
        self.decoder1 = MultiFlowDecoder(
            channels[0],
            skip_channels,
            num_flows=num_flows,
            field_channels=out_channels,
        )
        self.update4 = self._make_update_block(channels[2])
        self.update3 = self._make_update_block(channels[1], 2.0)
        self.update2 = self._make_update_block(channels[0], 4.0)

        combine_channels = out_channels * num_flows
        self.comb_block = nn.Sequential(
            SphereConv2d(
                combine_channels,
                2 * combine_channels,
                kernel_size=7,
                padding=3,
            ),
            nn.PReLU(2 * combine_channels),
            SphereConv2d(
                2 * combine_channels,
                out_channels,
                kernel_size=7,
                padding=3,
            ),
        )
    def _make_update_block(
        self,
        feature_channels: int,
        scale_factor: float | None = None,
    ) -> BasicUpdateBlock:
        return BasicUpdateBlock(
            cdim=feature_channels,
            hidden_dim=128,
            flow_dim=48,
            corr_dim=256,
            corr_dim2=160,
            fc_dim=124,
            scale_factor=scale_factor,
            corr_levels=self.corr_levels,
            radius=self.radius,
        )

    def _pad_to_multiple(
        self,
        x: torch.Tensor,
        multiple: int = 8,
        pole_parity: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, tuple[int, int, int]]:
        height, width = x.shape[-2:]
        pad_height = (-height) % multiple
        pad_width = (-width) % multiple
        top = pad_height // 2
        bottom = pad_height - top
        if pad_width or pad_height:
            x = spherical_pad(
                x,
                (0, pad_width, top, bottom),
                pole_parity=(
                    self.field_pole_parity
                    if pole_parity is None
                    else pole_parity
                ),
            )
        return x, (top, height, width)

    def _prepare_static(
        self,
        static: torch.Tensor | None,
        anchors: torch.Tensor,
    ) -> torch.Tensor:
        batch, _, height, width = anchors.shape
        if self.n_static_features == 0:
            return anchors.new_empty(batch, 0, height, width)
        if static is None:
            return anchors.new_zeros(
                batch,
                self.n_static_features,
                height,
                width,
            )
        if static.ndim == 3:
            static = static.unsqueeze(0)
        if static.ndim != 4:
            raise ValueError("static fields must be CHW or BCHW")
        if static.shape[1:] != (
            self.n_static_features,
            height,
            width,
        ):
            raise ValueError(
                "static fields must match configured channels and grid"
            )
        if static.size(0) == 1 and batch != 1:
            static = static.expand(batch, -1, -1, -1)
        elif static.size(0) != batch:
            raise ValueError("static batch size must match the anchors")
        return static.to(device=anchors.device, dtype=anchors.dtype)

    @staticmethod
    def _crop(
        x: torch.Tensor,
        crop: tuple[int, int, int],
    ) -> torch.Tensor:
        top, height, width = crop
        return x[..., top : top + height, :width]

    def _corr_lookup(
        self,
        corr_fn: BidirCorrBlock,
        coordinates: torch.Tensor,
        flow0: torch.Tensor,
        flow1: torch.Tensor,
        tau_safe: torch.Tensor,
        downsample: int = 1,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if downsample != 1:
            inverse = 1.0 / downsample
            flow0 = inverse * resize(flow0, scale_factor=inverse)
            flow1 = inverse * resize(flow1, scale_factor=inverse)
        corr0, corr1 = corr_fn(
            coordinates + flow1 / tau_safe,
            coordinates + flow0 / (1.0 - tau_safe),
        )
        return torch.cat((corr0, corr1), dim=1), torch.cat(
            (flow0, flow1), dim=1
        )

    def _raw_field_warp(
        self,
        fields: torch.Tensor,
        flow: torch.Tensor,
    ) -> torch.Tensor:
        limit = self.max_field_displacement
        if limit > 0:
            flow = limit * torch.tanh(flow / limit)
        return spherical_field_warp(
            fields,
            flow,
            periodic_longitude=True,
            pole_parity=self.field_pole_parity,
        )

    def _transport(
        self,
        x0: torch.Tensor,
        x1: torch.Tensor,
        tau: torch.Tensor,
        static: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        batch, _, height, width = x0.shape
        tau_safe = tau.clamp(1.0e-4, 1.0 - 1.0e-4)
        coordinates = coords_grid(
            batch,
            height // 8,
            width // 8,
            x0.device,
        )

        fmap0, fmap1 = self.feat_encoder([x0, x1])
        corr_fn = BidirCorrBlock(
            fmap0,
            fmap1,
            radius=self.radius,
            num_levels=self.corr_levels,
        )
        pyramid0 = torch.cat((x0, static), dim=1)
        pyramid1 = torch.cat((x1, static), dim=1)
        f0_1, f0_2, f0_3, f0_4 = self.encoder(pyramid0)
        f1_1, f1_2, f1_3, f1_4 = self.encoder(pyramid1)

        flow0_4, flow1_4, ft_3 = self.decoder4(
            f0_4,
            f1_4,
            tau,
            output_size=f0_3.shape[-2:],
        )
        corr4, pair4 = self._corr_lookup(
            corr_fn, coordinates, flow0_4, flow1_4, tau_safe
        )
        delta_ft, delta_pair = self.update4(ft_3, pair4, corr4)
        delta0, delta1 = delta_pair.chunk(2, dim=1)
        flow0_4 = flow0_4 + delta0
        flow1_4 = flow1_4 + delta1
        ft_3 = ft_3 + delta_ft

        flow0_3, flow1_3, ft_2 = self.decoder3(
            ft_3,
            f0_3,
            f1_3,
            flow0_4,
            flow1_4,
            output_size=f0_2.shape[-2:],
        )
        corr3, pair3 = self._corr_lookup(
            corr_fn,
            coordinates,
            flow0_3,
            flow1_3,
            tau_safe,
            downsample=2,
        )
        delta_ft, delta_pair = self.update3(ft_2, pair3, corr3)
        delta0, delta1 = delta_pair.chunk(2, dim=1)
        flow0_3 = flow0_3 + delta0
        flow1_3 = flow1_3 + delta1
        ft_2 = ft_2 + delta_ft

        flow0_2, flow1_2, ft_1 = self.decoder2(
            ft_2,
            f0_2,
            f1_2,
            flow0_3,
            flow1_3,
            output_size=f0_1.shape[-2:],
        )
        corr2, pair2 = self._corr_lookup(
            corr_fn,
            coordinates,
            flow0_2,
            flow1_2,
            tau_safe,
            downsample=4,
        )
        delta_ft, delta_pair = self.update2(ft_1, pair2, corr2)
        delta0, delta1 = delta_pair.chunk(2, dim=1)
        flow0_2 = flow0_2 + delta0
        flow1_2 = flow1_2 + delta1
        ft_1 = ft_1 + delta_ft

        flow0, flow1, mask, field_residual = self.decoder1(
            ft_1,
            f0_1,
            f1_1,
            flow0_2,
            flow1_2,
            output_size=x0.shape[-2:],
        )
        prediction = multi_flow_combine(
            self.comb_block,
            x0,
            x1,
            flow0,
            flow1,
            mask,
            field_residual,
            warp_fn=self._raw_field_warp,
        )
        return prediction, {
            "flow0": flow0.reshape(
                batch, self.num_flows, 2, height, width
            ),
            "flow1": flow1.reshape(
                batch, self.num_flows, 2, height, width
            ),
            "blend": mask,
        }

    def forward(
        self,
        x0: torch.Tensor,
        x1: torch.Tensor,
        tau: torch.Tensor,
        cond: torch.Tensor | None = None,
        static: torch.Tensor | None = None,
        *,
        return_aux: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        del cond
        if x0.shape != x1.shape:
            raise ValueError("WeatherAMT anchor shapes must match")
        if x0.size(1) != self.in_channels:
            raise ValueError(
                f"expected {self.in_channels} fields, got {x0.size(1)}"
            )
        tau = tau.reshape(-1, 1, 1, 1).to(
            device=x0.device, dtype=x0.dtype
        )
        if tau.size(0) != x0.size(0):
            raise ValueError("tau batch size must match the anchors")
        if bool(torch.all(tau <= 0.0)):
            return (x0, {}) if return_aux else x0
        if bool(torch.all(tau >= 1.0)):
            return (x1, {}) if return_aux else x1

        x0_pad, crop = self._pad_to_multiple(x0)
        x1_pad, crop1 = self._pad_to_multiple(x1)
        if crop != crop1:
            raise RuntimeError("anchor padding metadata diverged")
        static_input = self._prepare_static(static, x0)
        static_pad, static_crop = self._pad_to_multiple(
            static_input,
            pole_parity=self.static_pole_parity,
        )
        if static_crop != crop:
            raise RuntimeError("static and anchor padding metadata diverged")

        field_mean = 0.5 * (
            x0_pad.mean(dim=(-2, -1), keepdim=True)
            + x1_pad.mean(dim=(-2, -1), keepdim=True)
        )
        centered0 = x0_pad - field_mean
        centered1 = x1_pad - field_mean
        transported, aux = self._transport(
            centered0,
            centered1,
            tau,
            static_pad,
        )
        transported = transported + field_mean

        linear = (1.0 - tau) * x0_pad + tau * x1_pad
        if self.endpoint_envelope:
            envelope = 4.0 * tau * (1.0 - tau)
            prediction = linear + envelope * (transported - linear)
        else:
            prediction = transported
        prediction = torch.where(tau <= 0.0, x0_pad, prediction)
        prediction = torch.where(tau >= 1.0, x1_pad, prediction)
        prediction = self._crop(prediction, crop)

        if return_aux:
            aux = {
                key: self._crop(value, crop)
                for key, value in aux.items()
            }
            return prediction, aux
        return prediction
