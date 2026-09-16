"""Topology-aware UPR with state-compatible antipodal sphere operators."""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .weatherbridge_upr_lite_model import (
    SphereConv2d as LegacySphereConv2d,
)
from .weatherbridge_upr_lite_model import (
    PyramidRefiner,
    WeatherBridgeUPRLiteModel,
    _evaluate_trajectory,
)


def _antipodal_shift(x: torch.Tensor) -> torch.Tensor:
    width = x.size(-1)
    half = width // 2
    shifted = torch.roll(x, half, dims=-1)
    if width % 2:
        shifted = 0.5 * (
            shifted + torch.roll(x, half + 1, dims=-1)
        )
    return shifted


def _pole_pad_latitude(
    x: torch.Tensor,
    pad: int,
    pole_parity: torch.Tensor | None = None,
) -> torch.Tensor:
    if pad == 0:
        return x
    if pad < 0:
        raise ValueError("pole padding must be non-negative")
    height = x.size(-2)
    if pad <= height:
        top = _antipodal_shift(x[..., :pad, :].flip(-2))
        bottom = _antipodal_shift(x[..., -pad:, :].flip(-2))
        if pole_parity is not None:
            parity = torch.as_tensor(
                pole_parity,
                device=x.device,
                dtype=x.dtype,
            )
            if parity.numel() != x.size(-3):
                raise ValueError(
                    "pole parity must have one value per input channel"
                )
            parity = parity.reshape(1, -1, 1, 1)
            top = top * parity
            bottom = bottom * parity
        return torch.cat((top, x, bottom), dim=-2)

    coordinates = torch.arange(
        -pad,
        height + pad,
        device=x.device,
    )
    phase = coordinates.remainder(2 * height)
    crosses_pole = phase >= height
    source_rows = torch.where(
        crosses_pole,
        2 * height - 1 - phase,
        phase,
    )
    extended = x.index_select(-2, source_rows)
    shifted = _antipodal_shift(extended)
    crossing_mask = crosses_pole.reshape(1, 1, -1, 1)
    extended = torch.where(crossing_mask, shifted, extended)
    if pole_parity is not None:
        parity = torch.as_tensor(
            pole_parity,
            device=x.device,
            dtype=x.dtype,
        )
        if parity.numel() != x.size(-3):
            raise ValueError(
                "pole parity must have one value per input channel"
            )
        parity = parity.reshape(1, -1, 1, 1)
        extended = torch.where(
            crossing_mask,
            extended * parity,
            extended,
        )
    return extended


def _sphere_pad(
    x: torch.Tensor,
    pad: int,
    pole_parity: torch.Tensor | None = None,
) -> torch.Tensor:
    x = _pole_pad_latitude(x, pad, pole_parity)
    if pad:
        indices = torch.arange(
            -pad,
            x.size(-1) + pad,
            device=x.device,
        ).remainder(x.size(-1))
        x = x.index_select(-1, indices)
    return x


def spherical_resize(
    x: torch.Tensor,
    size: tuple[int, int] | torch.Size,
    pole_parity: torch.Tensor | None = None,
) -> torch.Tensor:
    """Align-corners-false bilinear resize on the lat-lon sphere."""
    out_h, out_w = (int(value) for value in size)
    in_h, in_w = x.shape[-2:]
    if (in_h, in_w) == (out_h, out_w):
        return x
    if min(in_h, in_w, out_h, out_w) < 1:
        raise ValueError("spherical resize requires non-empty grids")

    yy, xx = torch.meshgrid(
        torch.arange(out_h, device=x.device, dtype=torch.float32),
        torch.arange(out_w, device=x.device, dtype=torch.float32),
        indexing="ij",
    )
    source_y = (yy + 0.5) * (in_h / out_h) - 0.5
    source_x = (xx + 0.5) * (in_w / out_w) - 0.5
    sample = _pole_pad_latitude(x, 1, pole_parity)
    sample = torch.cat((sample, sample[..., :1]), dim=-1)
    grid = torch.stack(
        (
            torch.remainder(source_x, in_w) / in_w * 2.0 - 1.0,
            (source_y + 1.0)
            / max(sample.size(-2) - 1, 1)
            * 2.0
            - 1.0,
        ),
        dim=-1,
    )
    grid = grid.unsqueeze(0).expand(x.size(0), -1, -1, -1)
    with torch.autocast(device_type=x.device.type, enabled=False):
        output = F.grid_sample(
            sample.float(),
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )
    return output.to(x.dtype)


def spherical_warp(
    x: torch.Tensor,
    flow: torch.Tensor,
    pole_parity: torch.Tensor | None = None,
    pole_pad: int | None = None,
) -> torch.Tensor:
    """Backward warp with periodic longitude and antipodal pole crossing."""
    batch, channels, height, width = x.shape
    per_channel = flow.dim() == 5
    if per_channel:
        if flow.shape != (batch, channels, 2, height, width):
            raise ValueError("per-channel flow shape does not match fields")
        flow_flat = flow.reshape(batch * channels, 2, height, width)
    else:
        if flow.shape != (batch, 2, height, width):
            raise ValueError("shared flow shape does not match fields")
        flow_flat = flow

    if pole_pad is None:
        max_vertical_displacement = float(
            flow_flat[:, 1].detach().abs().amax().item()
        )
        if not math.isfinite(max_vertical_displacement):
            raise ValueError(
                "flow contains a non-finite vertical displacement"
            )
        pole_pad = max(1, math.ceil(max_vertical_displacement) + 1)
    elif pole_pad < 1:
        raise ValueError("pole_pad must be positive")
    sample = _pole_pad_latitude(x, pole_pad, pole_parity)
    if per_channel:
        sample = sample.reshape(
            batch * channels,
            1,
            sample.size(-2),
            width,
        )
    sample = torch.cat((sample, sample[..., :1]), dim=-1)
    yy, xx = torch.meshgrid(
        torch.arange(height, device=x.device, dtype=torch.float32),
        torch.arange(width, device=x.device, dtype=torch.float32),
        indexing="ij",
    )
    flow_float = flow_flat.float()
    source_x = torch.remainder(xx[None] + flow_float[:, 0], width)
    source_y = yy[None] + flow_float[:, 1] + pole_pad
    grid = torch.stack(
        (
            source_x / width * 2.0 - 1.0,
            source_y
            / max(sample.size(-2) - 1, 1)
            * 2.0
            - 1.0,
        ),
        dim=-1,
    )
    with torch.autocast(device_type=x.device.type, enabled=False):
        warped = F.grid_sample(
            sample.float(),
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )
    warped = warped.to(x.dtype)
    if per_channel:
        return warped.reshape(batch, channels, height, width)
    return warped


def endpoint_blend(
    logits: torch.Tensor,
    tau: torch.Tensor,
) -> torch.Tensor:
    tau_float = tau.float()
    prior = (1.0 - tau_float).clamp(1.0e-6, 1.0 - 1.0e-6)
    alpha = torch.sigmoid(torch.logit(prior) + 2.0 * logits.float())
    alpha = torch.where(tau_float <= 0.0, torch.ones_like(alpha), alpha)
    alpha = torch.where(tau_float >= 1.0, torch.zeros_like(alpha), alpha)
    return alpha.to(logits.dtype)


class AntipodalSphereConv2d(nn.Conv2d):
    """State-compatible convolution over a lat-lon sphere."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        *,
        stride: tuple[int, int],
        dilation: tuple[int, int],
        groups: int,
        bias: bool,
        pole_parity: torch.Tensor | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=0,
            dilation=dilation,
            groups=groups,
            bias=bias,
            device=device,
            dtype=dtype,
        )
        self.sphere_padding = kernel_size // 2
        self.register_buffer(
            "pole_parity",
            (
                None
                if pole_parity is None
                else torch.as_tensor(pole_parity, dtype=torch.float32)
            ),
            persistent=False,
        )

    @classmethod
    def from_legacy(
        cls,
        module: LegacySphereConv2d,
        pole_parity: torch.Tensor | None = None,
    ) -> "AntipodalSphereConv2d":
        converted = cls(
            module.in_channels,
            module.out_channels,
            module.kernel_size[0],
            stride=module.stride,
            dilation=module.dilation,
            groups=module.groups,
            bias=module.bias is not None,
            pole_parity=pole_parity,
            device=module.weight.device,
            dtype=module.weight.dtype,
        )
        converted.load_state_dict(module.state_dict())
        converted.train(module.training)
        return converted

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(
            _sphere_pad(x, self.sphere_padding, self.pole_parity)
        )


class SphericalPyramidRefiner(PyramidRefiner):
    """State-compatible refiner with antipodal feature warping."""

    @classmethod
    def from_legacy(
        cls,
        module: PyramidRefiner,
    ) -> "SphericalPyramidRefiner":
        hidden = module.input_proj.out_channels
        n_flow_modes = (
            module.flow_head.out_channels // module.motion_components
        )
        local_matching = getattr(module, "local_matching", False)
        static_width = (
            module.input_proj.in_channels
            - hidden * (5 if local_matching else 4)
            - n_flow_modes * module.motion_components
            - 1
        )
        converted = cls(
            hidden=hidden,
            static_width=static_width,
            n_flow_modes=n_flow_modes,
            n_blocks=len(module.blocks),
            motion_components=module.motion_components,
            local_matching=local_matching,
        )
        converted.load_state_dict(module.state_dict())
        converted.train(module.training)
        return converted

    def forward(
        self,
        feature0: torch.Tensor,
        feature1: torch.Tensor,
        state: torch.Tensor,
        flow_modes: torch.Tensor,
        static: torch.Tensor,
        tau: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, _, height, width = feature0.shape
        forward, backward = _evaluate_trajectory(
            flow_modes[:, :1],
            tau,
        )
        warped0 = spherical_warp(
            feature0,
            forward[:, 0],
            pole_pad=8,
        )
        warped1 = spherical_warp(
            feature1,
            backward[:, 0],
            pole_pad=8,
        )
        tau_map = tau.expand(batch, 1, height, width)
        inputs = [warped0, warped1, warped1 - warped0]
        if self.local_matching:
            inputs.append(self._matching_features(warped0, warped1))
        inputs.extend(
            (state, flow_modes.flatten(1, 2), static, tau_map)
        )
        h = torch.cat(inputs, dim=1)
        state = self.blocks(self.input_proj(h))
        delta = torch.tanh(self.flow_head(state))
        delta = delta.view(
            batch,
            flow_modes.size(1),
            self.motion_components,
            height,
            width,
        )
        return state, flow_modes + delta


def _replace_legacy_convolutions(
    module: nn.Module,
    *,
    path: str = "",
    input_pole_parity: torch.Tensor,
) -> None:
    for name, child in tuple(module.named_children()):
        child_path = f"{path}.{name}" if path else name
        if isinstance(child, LegacySphereConv2d):
            parity = (
                input_pole_parity
                if child_path == "frame_stem.proj"
                else None
            )
            setattr(
                module,
                name,
                AntipodalSphereConv2d.from_legacy(child, parity),
            )
        else:
            _replace_legacy_convolutions(
                child,
                path=child_path,
                input_pole_parity=input_pole_parity,
            )


class WeatherBridgeUPRSphericalModel(WeatherBridgeUPRLiteModel):
    """Implicit-global UPR with true spherical field topology."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        field_pole_parity = torch.ones(self.out_channels)
        for index in (*range(4, 12), 21, 22):
            if index < self.out_channels:
                field_pole_parity[index] = -1.0
        self.register_buffer(
            "field_pole_parity",
            field_pole_parity,
            persistent=False,
        )
        self.register_buffer(
            "group_pole_parity",
            torch.tensor((1.0, -1.0, -1.0, 1.0, 1.0)),
            persistent=False,
        )
        self.refiner = SphericalPyramidRefiner.from_legacy(self.refiner)
        _replace_legacy_convolutions(
            self,
            input_pole_parity=self.field_pole_parity,
        )

    @staticmethod
    def _resize_flow(
        flow: torch.Tensor,
        size: tuple[int, int],
    ) -> torch.Tensor:
        old_h, old_w = flow.shape[-2:]
        new_h, new_w = size
        batch, modes, components = flow.shape[:3]
        flattened = flow.reshape(
            batch,
            modes * components,
            old_h,
            old_w,
        )
        parity = flattened.new_full((modes * components,), -1.0)
        resized = spherical_resize(
            flattened,
            size,
            pole_parity=parity,
        ).view(batch, modes, components, new_h, new_w)
        resized[:, :, 0::2] *= new_w / old_w
        resized[:, :, 1::2] *= new_h / old_h
        return resized

    def _prepare_static(
        self,
        static: torch.Tensor | None,
        batch: int,
        size: tuple[int, int],
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if static is None:
            return torch.zeros(
                batch,
                self.n_static_features,
                *size,
                device=device,
                dtype=dtype,
            )
        if static.dim() == 3:
            static = static.unsqueeze(0)
        if static.size(0) == 1:
            static = static.expand(batch, -1, -1, -1)
        if static.size(0) != batch:
            raise ValueError(
                f"static batch {static.size(0)} != input batch {batch}"
            )
        static = static[:, : self.n_static_features].to(
            device=device,
            dtype=dtype,
        )
        return spherical_resize(static, size)

    def _warp_fields(
        self,
        fields: torch.Tensor,
        flow: torch.Tensor,
        pole_pad: int,
    ) -> torch.Tensor:
        if flow.dim() == 4:
            return spherical_warp(
                fields,
                flow,
                pole_parity=self.field_pole_parity,
                pole_pad=pole_pad,
            )
        batch, channels, height, width = fields.shape
        if flow.shape != (batch, 5, 2, height, width):
            raise ValueError("group flow shape does not match fields")
        ordered = fields.index_select(1, self.grouped_order)
        grouped = torch.cat(
            (
                ordered,
                fields.new_zeros(batch, 1, height, width),
            ),
            dim=1,
        ).reshape(batch * 5, 5, height, width)
        warped = spherical_warp(
            grouped,
            flow.reshape(batch * 5, 2, height, width),
            pole_parity=self.group_pole_parity,
            pole_pad=pole_pad,
        )
        warped_ordered = warped.reshape(
            batch,
            25,
            height,
            width,
        )[:, :channels]
        return warped_ordered.index_select(1, self.inverse_grouped_order)

    def _smooth(self, x: torch.Tensor) -> torch.Tensor:
        return F.avg_pool2d(
            _sphere_pad(x, 1, self.field_pole_parity),
            kernel_size=3,
            stride=1,
        )

    def forward(
        self,
        x0: torch.Tensor,
        xT: torch.Tensor,
        tau: torch.Tensor,
        cond: torch.Tensor | None = None,
        static: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        del cond
        batch, _, height, width = x0.shape
        tau = tau.reshape(batch, 1, 1, 1).to(
            device=x0.device,
            dtype=x0.dtype,
        )
        motion_tau = (
            tau if self.query_conditioned else torch.full_like(tau, 0.5)
        )
        finest_divisor = self.pyramid_divisors[-1]
        work_size = (
            max(1, height // finest_divisor),
            max(1, width // finest_divisor),
        )
        x0_work = F.interpolate(x0, size=work_size, mode="area")
        xT_work = F.interpolate(xT, size=work_size, mode="area")
        feature0_work = self.frame_stem(x0_work)
        featureT_work = self.frame_stem(xT_work)
        static_work = self._prepare_static(
            static,
            batch,
            work_size,
            x0.device,
            x0.dtype,
        )
        static_feature_work = self.static_stem(static_work)

        pyramid_sizes = [
            (
                max(1, height // divisor),
                max(1, width // divisor),
            )
            for divisor in self.pyramid_divisors
        ]
        state = None
        flow_modes = None
        for level, size in enumerate(pyramid_sizes):
            feature0 = F.adaptive_avg_pool2d(feature0_work, size)
            featureT = F.adaptive_avg_pool2d(featureT_work, size)
            static_feature = F.adaptive_avg_pool2d(
                static_feature_work,
                size,
            )
            if state is None:
                state = torch.zeros_like(feature0)
                flow_modes = feature0.new_zeros(
                    batch,
                    self.n_flow_modes,
                    self.motion_components,
                    *size,
                )
            else:
                state = spherical_resize(state, size)
                flow_modes = self._resize_flow(flow_modes, size)
            state, flow_modes = self.refiner(
                feature0,
                featureT,
                state,
                flow_modes,
                static_feature,
                motion_tau,
            )
            if level == 0 and self.global_mixer is not None:
                state = self.global_mixer(state)

        flow_modes = self._resize_flow(flow_modes, (height, width))
        flow_modes = flow_modes * self.flow_scale
        if self.n_flow_modes == 1:
            forward_flow, backward_flow = _evaluate_trajectory(
                flow_modes,
                tau,
            )
            forward_flow = forward_flow[:, 0]
            backward_flow = backward_flow[:, 0]
        else:
            group_trajectory = self._group_flows(flow_modes)
            forward_flow, backward_flow = _evaluate_trajectory(
                group_trajectory,
                tau,
            )

        synthesis_state = (
            self.query_state_modulator(state, tau)
            if self.query_state_modulator is not None
            else state
        )
        blend_logits = spherical_resize(
            self.blend_head(synthesis_state),
            (height, width),
        )
        alpha = endpoint_blend(blend_logits, tau)
        max_vertical_displacement = float(
            torch.stack(
                (
                    forward_flow[..., 1, :, :].detach().abs().amax(),
                    backward_flow[..., 1, :, :].detach().abs().amax(),
                )
            ).amax().item()
        )
        if not math.isfinite(max_vertical_displacement):
            raise ValueError(
                "flow contains a non-finite vertical displacement"
            )
        pole_pad = max(1, math.ceil(max_vertical_displacement) + 1)
        warped0 = self._warp_fields(x0, forward_flow, pole_pad)
        warpedT = self._warp_fields(xT, backward_flow, pole_pad)
        scaffold = alpha * warped0 + (1.0 - alpha) * warpedT
        if self.detail_head is not None:
            detail = (
                alpha * (warped0 - self._smooth(warped0))
                + (1.0 - alpha) * (warpedT - self._smooth(warpedT))
            )
            detail_logits = spherical_resize(
                self.detail_head(synthesis_state),
                (height, width),
            )
            detail_gain = (
                1.0
                + 2.0
                * tau
                * (1.0 - tau)
                * torch.tanh(detail_logits)
            )
            scaffold = scaffold + (detail_gain - 1.0) * detail

        residual = spherical_resize(
            self.residual_head(synthesis_state),
            (height, width),
            pole_parity=self.field_pole_parity,
        )
        scale = torch.tanh(self.residual_scale).view(1, -1, 1, 1)
        delta = tau * (1.0 - tau) * scale * residual
        prediction = scaffold + delta
        hydrostatic_delta = prediction.new_zeros(
            batch,
            5,
            height,
            width,
        )
        if self.hydrostatic_head is not None:
            column = torch.cat(
                (prediction[:, 0:4], prediction[:, 12:16]),
                dim=1,
            )
            hydrostatic_delta = (
                tau
                * (1.0 - tau)
                * self.hydrostatic_head(column).to(prediction.dtype)
            )
            prediction = prediction.index_add(
                1,
                self.hydrostatic_indices,
                hydrostatic_delta,
            )
        prediction = torch.where(tau <= 0.0, x0, prediction)
        prediction = torch.where(tau >= 1.0, xT, prediction)
        return prediction, {
            "flow_modes": flow_modes,
            "alpha": alpha,
            "delta": delta,
            "hydrostatic_delta": hydrostatic_delta,
        }
