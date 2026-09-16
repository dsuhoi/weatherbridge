"""Inference-only routing between interpolation experts by target time."""
from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import nn


class TemporalExpertRouter(nn.Module):
    """Dispatch each sample to one frozen expert using only its target hour."""

    def __init__(
        self,
        experts: Mapping[str, nn.Module],
        route_by_tau: Mapping[int, str],
        *,
        delta_t: float,
        tau_tolerance: float = 1e-4,
    ) -> None:
        super().__init__()
        if not experts:
            raise ValueError("temporal router requires at least one expert")
        if delta_t <= 0 or not float(delta_t).is_integer():
            raise ValueError("delta_t must be a positive integer number of hours")
        if tau_tolerance <= 0:
            raise ValueError("tau_tolerance must be positive")
        expert_names = set(experts)
        if any(not name.isidentifier() for name in expert_names):
            raise ValueError("expert names must be valid identifiers")
        if not route_by_tau:
            raise ValueError("temporal router requires a non-empty route")
        normalized_route = {
            int(hour): str(expert)
            for hour, expert in route_by_tau.items()
        }
        unknown = set(normalized_route.values()) - expert_names
        if unknown:
            raise ValueError(f"route references unknown experts: {unknown}")
        if any(hour <= 0 or hour >= delta_t for hour in normalized_route):
            raise ValueError("routed hours must lie strictly between anchors")

        self.experts = nn.ModuleDict(dict(experts))
        self._expert_names = tuple(self.experts)
        self.route_by_tau = normalized_route
        self.delta_t = float(delta_t)
        self.tau_tolerance = float(tau_tolerance)
        expert_indices = {
            name: index
            for index, name in enumerate(self._expert_names)
        }
        route_lookup = torch.full(
            (int(self.delta_t) + 1,),
            -1,
            dtype=torch.int64,
        )
        for hour, name in normalized_route.items():
            route_lookup[hour] = expert_indices[name]
        self.register_buffer(
            "route_expert_index",
            route_lookup,
            persistent=False,
        )

    @staticmethod
    def _prediction(value: object) -> torch.Tensor:
        prediction = value[0] if isinstance(value, tuple) else value
        if not isinstance(prediction, torch.Tensor):
            raise TypeError("expert forward must return a tensor or tuple")
        return prediction

    def forward(
        self,
        x0: torch.Tensor,
        xT: torch.Tensor,
        tau: torch.Tensor,
        cond: torch.Tensor | None = None,
        static: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch = x0.size(0)
        tau_flat = tau.reshape(-1)
        if tau_flat.numel() == 1 and batch != 1:
            tau_flat = tau_flat.expand(batch)
        if tau_flat.numel() != batch:
            raise ValueError("tau must contain one value per sample")

        tau_hours_float = tau_flat.float() * self.delta_t
        tau_hours = tau_hours_float.round().to(dtype=torch.int64)
        in_range = (tau_hours >= 0) & (
            tau_hours < self.route_expert_index.numel()
        )
        clamped_hours = tau_hours.clamp(
            min=0,
            max=self.route_expert_index.numel() - 1,
        )
        expert_index = self.route_expert_index.index_select(
            0,
            clamped_hours,
        )
        valid = (
            (tau_hours_float - tau_hours.float()).abs()
            <= self.tau_tolerance
        ) & in_range & (expert_index >= 0)
        message = "no temporal expert route for a non-integer or unknown tau"
        if tau_hours.device.type == "cuda" and hasattr(torch, "_assert_async"):
            torch._assert_async(valid.all(), message)
        elif not bool(valid.all()):
            raise ValueError(message)

        output: torch.Tensor | None = None
        for routed_index, (_, expert) in enumerate(self.experts.items()):
            index = torch.nonzero(
                expert_index == routed_index,
                as_tuple=False,
            ).flatten()
            if index.numel() == 0:
                continue
            cond_subset = cond
            if (
                cond is not None
                and cond.dim() > 0
                and cond.size(0) == batch
            ):
                cond_subset = cond.index_select(0, index)
            static_subset = static
            if static is not None and static.dim() == 4 and static.size(0) == batch:
                static_subset = static.index_select(0, index)
            prediction = self._prediction(
                expert(
                    x0.index_select(0, index),
                    xT.index_select(0, index),
                    tau_flat.index_select(0, index),
                    cond=cond_subset,
                    static=static_subset,
                )
            )
            if output is None:
                output = prediction.new_empty(
                    (batch, *prediction.shape[1:])
                )
            output.index_copy_(0, index, prediction)

        if output is None:
            raise RuntimeError("temporal router dispatched no samples")
        return output
