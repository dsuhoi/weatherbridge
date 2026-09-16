"""Flow-matching expansion of the checkpoint-compatible WeatherDCAE model."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .dcae_adaln_model import TimeMLP, WeatherDCAEAdaLNModel


class WeatherDCAEFlowMatchingModel(WeatherDCAEAdaLNModel):
    """WeatherDCAE with state-conditioned rectified-flow refinement.

    The anchor encoder and decoder are unchanged. A zero-initialized adapter
    injects the current bridge residual into the existing encoder input, while
    a zero-initialized time branch conditions the shared AdaLN embedding. Thus
    a WeatherDCAE checkpoint remains an exact interior-time warm start.
    """

    def __init__(
        self,
        *args,
        flow_matching: bool = True,
        flow_matching_steps: int = 4,
        endpoint_preserving: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        if flow_matching_steps < 1:
            raise ValueError("flow_matching_steps must be positive")
        if self.in_channels != self.out_channels:
            raise ValueError("flow matching requires equal input/output channels")

        self.flow_matching = bool(flow_matching)
        self.flow_matching_steps = int(flow_matching_steps)
        self.endpoint_preserving = bool(endpoint_preserving)

        encoder_channels = 2 * self.in_channels + self.n_static_features
        self.state_adapter = nn.Conv2d(
            self.out_channels,
            encoder_channels,
            kernel_size=1,
        )
        nn.init.zeros_(self.state_adapter.weight)
        nn.init.zeros_(self.state_adapter.bias)

        self.flow_time_mlp = TimeMLP(
            time_dim=self.time_emb_dim,
            freq_dim=self.time_freq_dim,
            base_period=self.time_base_period,
        )
        nn.init.zeros_(self.flow_time_mlp.mlp[-1].weight)
        nn.init.zeros_(self.flow_time_mlp.mlp[-1].bias)

    @staticmethod
    def _align_vector(value: torch.Tensor, batch_size: int, name: str) -> torch.Tensor:
        value = value.reshape(-1)
        if value.size(0) == batch_size:
            return value
        if batch_size % value.size(0) == 0:
            return value.repeat_interleave(batch_size // value.size(0), dim=0)
        raise ValueError(
            f"{name} batch {value.size(0)} cannot align with input batch {batch_size}"
        )

    def _prepare_static(
        self,
        static: torch.Tensor | None,
        batch_size: int,
    ) -> torch.Tensor | None:
        if self.n_static_features == 0:
            return None
        if static is None:
            raise ValueError(
                f"Model requires static (n_static_features={self.n_static_features})"
            )
        if static.dim() == 3:
            static = static.unsqueeze(0)
        if static.size(0) == batch_size:
            return static
        if batch_size % static.size(0) == 0:
            return static.repeat_interleave(batch_size // static.size(0), dim=0)
        return static.expand(batch_size, -1, -1, -1)

    def _scaled_decoder_residual(self, decoder_out: torch.Tensor) -> torch.Tensor:
        if self.residual_clip is not None and self.residual_clip > 0:
            decoder_out = torch.clamp(
                decoder_out,
                -self.residual_clip,
                self.residual_clip,
            )
        residual_scale = torch.tanh(self.residual_scale)
        if self.residual_scale_floor > 0.0:
            sign = torch.where(
                residual_scale >= 0,
                torch.ones_like(residual_scale),
                -torch.ones_like(residual_scale),
            )
            residual_scale = sign * residual_scale.abs().clamp(
                min=self.residual_scale_floor
            )
        return residual_scale * decoder_out

    def _proposal(
        self,
        x0: torch.Tensor,
        xT: torch.Tensor,
        tau: torch.Tensor,
        bridge_state: torch.Tensor,
        flow_time: torch.Tensor,
        static: torch.Tensor | None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        batch_size = x0.size(0)
        tau = self._align_vector(tau, batch_size, "tau")
        flow_time = self._align_vector(flow_time, batch_size, "flow_time")
        tau_b = tau.view(-1, 1, 1, 1)
        x_linear = (1.0 - tau_b) * x0 + tau_b * xT
        if bridge_state.shape != x0.shape:
            raise ValueError("bridge_state must match the anchor shape")

        time_embedding = self.time_mlp(tau) + self.flow_time_mlp(flow_time)
        parts = [x0, xT]
        prepared_static = self._prepare_static(static, batch_size)
        if prepared_static is not None:
            parts.append(prepared_static)
        encoder_input = torch.cat(parts, dim=1)
        encoder_input = encoder_input + self.state_adapter(
            bridge_state - x_linear
        )

        latent = self.encoder(self._crop_lat(encoder_input), time_embedding)
        decoder_raw = self.decoder(latent, time_embedding)
        decoder_out = F.interpolate(
            decoder_raw,
            size=x0.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        residual = self._scaled_decoder_residual(decoder_out)
        proposal = decoder_out if self.direct_prediction else x_linear + residual
        return proposal, {
            "z_tau": latent,
            "decoder_out": decoder_out,
            "x_bilinear": x_linear,
            "flow_time": flow_time,
            "flow_matching_state": bridge_state,
            "t_emb": time_embedding,
        }

    def flow_matching_velocity(
        self,
        x0: torch.Tensor,
        xT: torch.Tensor,
        tau: torch.Tensor,
        bridge_state: torch.Tensor,
        flow_time: torch.Tensor,
        cond: torch.Tensor | None = None,
        static: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        del cond
        if not self.flow_matching:
            raise RuntimeError("flow_matching_velocity requires flow_matching=True")
        proposal, auxiliary = self._proposal(
            x0,
            xT,
            tau,
            bridge_state,
            flow_time,
            static,
        )
        tau = self._align_vector(tau, x0.size(0), "tau")
        tau_b = tau.view(-1, 1, 1, 1)
        x_linear = (1.0 - tau_b) * x0 + tau_b * xT
        return proposal - x_linear, auxiliary

    def forward(
        self,
        x0: torch.Tensor,
        xT: torch.Tensor,
        tau: torch.Tensor,
        cond: torch.Tensor | None = None,
        static: torch.Tensor | None = None,
        *,
        bridge_state: torch.Tensor | None = None,
        flow_time: torch.Tensor | None = None,
        integrate_flow: bool = True,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        del cond
        batch_size = x0.size(0)
        tau = self._align_vector(tau, batch_size, "tau")
        tau_b = tau.view(-1, 1, 1, 1)
        x_linear = (1.0 - tau_b) * x0 + tau_b * xT

        if self.flow_matching and bridge_state is None:
            if not integrate_flow:
                raise ValueError("bridge_state is required for velocity evaluation")
            state = x_linear
            auxiliary: dict[str, torch.Tensor] = {}
            step_size = 1.0 / self.flow_matching_steps
            for step in range(self.flow_matching_steps):
                integration_time = x0.new_full(
                    (batch_size,),
                    (step + 0.5) * step_size,
                )
                velocity, auxiliary = self.flow_matching_velocity(
                    x0,
                    xT,
                    tau,
                    state,
                    integration_time,
                    static=static,
                )
                state = state + step_size * velocity
            output = state
            auxiliary = {
                **auxiliary,
                "flow_matching_state": state,
                "flow_matching_steps": x0.new_tensor(self.flow_matching_steps),
            }
        else:
            if bridge_state is not None and not self.flow_matching:
                raise ValueError("bridge_state requires flow_matching=True")
            state = x_linear if bridge_state is None else bridge_state
            integration_time = (
                x0.new_zeros(batch_size) if flow_time is None else flow_time
            )
            output, auxiliary = self._proposal(
                x0,
                xT,
                tau,
                state,
                integration_time,
                static,
            )

        if self.endpoint_preserving:
            output = torch.where(tau_b == 0, x0, output)
            output = torch.where(tau_b == 1, xT, output)
        return output, auxiliary
