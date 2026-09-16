"""Compatibility API for the archived baseline evaluator.

Maintained code still imports the numerical interpolation helper from the
historical root module. Keep the lightweight functions importable without
loading the archived ResNet-ODE stack; delegate CLI-only behavior lazily.
"""
from __future__ import annotations

from typing import Dict, List

import torch
from torch.utils.data import DataLoader

from metrics import WeatherMetrics


def interpolate_time_with_f_interpolate(
    x0: torch.Tensor,
    x1: torch.Tensor,
    tau: torch.Tensor,
    mode: str,
    steps: int = 6,
) -> torch.Tensor:
    """Interpolate two temporal anchors exactly at each sample's ``tau``."""
    del steps
    if mode not in {"bilinear", "bicubic"}:
        raise ValueError("mode must be 'bilinear' or 'bicubic'")

    tau_flat = tau.float().reshape(-1)
    batch_size = x0.size(0)
    if tau_flat.numel() == 1 and batch_size > 1:
        tau_flat = tau_flat.expand(batch_size)
    if tau_flat.numel() != batch_size:
        raise ValueError(
            f"tau has {tau_flat.numel()} values for batch size {batch_size}"
        )
    tau_map = tau_flat.reshape(batch_size, 1, 1, 1)
    return (1.0 - tau_map) * x0 + tau_map * x1


def evaluate_per_hour_all(
    loader: DataLoader,
    model: torch.nn.Module | None,
    model_type: str,
    device: torch.device,
    channel_groups: Dict[str, List[int]],
    writer,
    n_ensemble_passes: int = 1,
    max_tau_hours: int = 6,
    skip_bicubic: bool = False,
) -> Dict[int, Dict[str, Dict[str, float]]]:
    """Evaluate every interior hour against its aligned target field."""
    del n_ensemble_passes
    totals: Dict[int, Dict[str, Dict[str, float]]] = {}
    counts: Dict[int, int] = {}
    eval_hours = list(range(1, max_tau_hours))

    with torch.no_grad():
        for batch in loader:
            x0 = batch["x0"].to(device)
            x1 = batch["x1"].to(device)
            targets = batch.get("target_all_hours")
            if targets is None:
                raise ValueError(
                    "per-hour evaluation requires target_all_hours; "
                    "construct the dataset with eval_all_hours=True"
                )
            targets = targets.to(device)
            batch_size = x0.size(0)
            static = batch.get("static")
            if static is not None:
                static = static.to(device)

            for target_index, hour in enumerate(eval_hours):
                tau = torch.full(
                    (batch_size,),
                    hour / max_tau_hours,
                    dtype=x0.dtype,
                    device=device,
                )
                predictions = {
                    "bilinear": interpolate_time_with_f_interpolate(
                        x0, x1, tau, mode="bilinear"
                    )
                }
                if not skip_bicubic:
                    predictions["bicubic"] = interpolate_time_with_f_interpolate(
                        x0, x1, tau, mode="bicubic"
                    )
                if model is not None:
                    if model_type == "weather_hermite":
                        cond = torch.full_like(tau, float(max_tau_hours))
                        predictions["model"], _ = model(x0, x1, tau, cond)
                    else:
                        output = model(
                            x0,
                            x1,
                            batch["time_emb"].to(device),
                            tau,
                            static=static,
                            return_endpoint=False,
                        )
                        predictions["model"] = getattr(output, "pred", output)

                target = targets[:, target_index]
                hour_totals = totals.setdefault(hour, {})
                counts[hour] = counts.get(hour, 0) + batch_size
                for name, prediction in predictions.items():
                    metrics = WeatherMetrics.all(
                        prediction, target, channel_groups=channel_groups
                    )
                    method_totals = hour_totals.setdefault(name, {})
                    for key, value in metrics.items():
                        scalar = float(value)
                        method_totals[key] = (
                            method_totals.get(key, 0.0) + scalar * batch_size
                        )
                        if writer is not None:
                            writer.add_scalar(
                                f"per_hour/{hour}/{name}/{key}", scalar, counts[hour]
                            )

    return {
        hour: {
            method: {
                key: value / counts[hour]
                for key, value in metrics.items()
            }
            for method, metrics in methods.items()
        }
        for hour, methods in totals.items()
    }


def load_weather_hermite_model(*args, **kwargs):
    from legacy.scripts.evaluate_baselines import load_weather_hermite_model as load

    return load(*args, **kwargs)


def main() -> None:
    from legacy.scripts.evaluate_baselines import main as legacy_main

    legacy_main()


if __name__ == "__main__":
    main()
