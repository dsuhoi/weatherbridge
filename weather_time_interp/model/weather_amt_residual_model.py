"""Residualized WeatherAMT with an exact linear initial prediction."""
from __future__ import annotations

import torch
import torch.nn as nn

from .weather_amt_model import WeatherAMTModel


class WeatherAMTResidualModel(WeatherAMTModel):
    """Gate AMT transport around the linear interpolation scaffold.

    The original AMT synthesis heads are randomly initialized, which gives a
    poor initial weather field. A per-field zero-initialized gate makes the
    initial prediction exactly linear. The gate receives gradients
    immediately; the transport core starts receiving gradients after the
    first optimizer update.
    """

    def __init__(self, *args, **kwargs) -> None:
        kwargs["endpoint_envelope"] = False
        super().__init__(*args, **kwargs)
        self.transport_gain = nn.Parameter(torch.zeros(self.out_channels))

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
        transported = super().forward(
            x0,
            x1,
            tau,
            cond=cond,
            static=static,
            return_aux=return_aux,
        )
        if return_aux:
            transported, aux = transported
        else:
            aux = {}

        tau = tau.reshape(-1, 1, 1, 1).to(
            device=x0.device,
            dtype=x0.dtype,
        )
        linear = (1.0 - tau) * x0 + tau * x1
        envelope = 4.0 * tau * (1.0 - tau)
        gain = torch.tanh(self.transport_gain).reshape(1, -1, 1, 1)
        prediction = linear + envelope * gain * (transported - linear)
        prediction = torch.where(tau <= 0.0, x0, prediction)
        prediction = torch.where(tau >= 1.0, x1, prediction)

        if return_aux:
            aux = dict(aux)
            aux["transport_gain"] = gain.expand(
                x0.size(0),
                -1,
                1,
                1,
            )
            return prediction, aux
        return prediction
