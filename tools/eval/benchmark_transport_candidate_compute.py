#!/usr/bin/env python3
"""Synthetic full-grid compute benchmark for transport candidates."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import statistics
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _optional_dependency_version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def _time_cuda(function, iterations: int) -> list[float]:
    timings = []
    for _ in range(iterations):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        function()
        end.record()
        end.synchronize()
        timings.append(float(start.elapsed_time(end)))
    return timings


def _build_model(
    arch: str,
    state: dict[str, torch.Tensor] | None = None,
) -> torch.nn.Module:
    if arch in {"amt", "amt_residual"}:
        from weather_time_interp.model.weather_amt_model import WeatherAMTModel

        model_type = WeatherAMTModel
        if arch == "amt_residual":
            from weather_time_interp.model.weather_amt_residual_model import (
                WeatherAMTResidualModel,
            )

            model_type = WeatherAMTResidualModel
        model = model_type(
            in_channels=24,
            out_channels=24,
            n_static_features=3,
            corr_radius=3,
            corr_levels=4,
            num_flows=5,
            channels=(48, 64, 72, 110),
            skip_channels=48,
            endpoint_envelope=True,
            max_field_displacement=16.0,
        )
    elif arch in {
        "upr_implicit_global_14m",
        "upr_local_corr_14m",
        "upr_query_match_14m",
    }:
        from weather_time_interp.model.weatherbridge_upr_lite_model import (
            WeatherBridgeUPRLiteModel,
        )
        from weather_time_interp.model.weatherbridge_upr_scaled_model import (
            upr_scaled_variant_kwargs,
        )

        model = WeatherBridgeUPRLiteModel(
            **upr_scaled_variant_kwargs(arch)
        )
    elif arch == "upr_spherical_implicit_global_14m":
        from weather_time_interp.model.weatherbridge_upr_scaled_model import (
            upr_scaled_variant_kwargs,
        )
        from weather_time_interp.model.weatherbridge_upr_spherical_model import (
            WeatherBridgeUPRSphericalModel,
        )

        model = WeatherBridgeUPRSphericalModel(
            **upr_scaled_variant_kwargs("upr_implicit_global_14m")
        )
    else:
        from weather_time_interp.model.weatherbridge_flow_model import (
            WeatherBridgeModel,
        )

        if arch not in {
            "flow_pp3",
            "flow_pp3_compact_l",
            "flow_spherical_ep",
            "flow_compact_vp3",
            "flow_compact_vp3_m",
            "flow_compact_vp3_l",
            "flow_compact_hermite_l",
            "flow_compact_lagrange_l",
        }:
            raise ValueError(f"unknown architecture {arch}")
        compact = arch.startswith("flow_compact_")
        compact_pp3 = arch == "flow_pp3_compact_l"
        spherical = arch in {
            "flow_spherical_ep",
            "flow_compact_vp3",
            "flow_compact_vp3_m",
            "flow_compact_vp3_l",
            "flow_compact_hermite_l",
            "flow_compact_lagrange_l",
        }
        compact_hidden = {
            "flow_compact_vp3": 32,
            "flow_compact_vp3_m": 40,
            "flow_compact_vp3_l": 56,
            "flow_compact_hermite_l": 56,
            "flow_compact_lagrange_l": 56,
        }
        model = WeatherBridgeModel(
            in_channels=24,
            out_channels=24,
            n_static_features=3,
            hidden=(
                compact_hidden[arch]
                if compact
                else 56 if compact_pp3
                else 72
            ),
            n_levels=3,
            use_skip=True,
            gated_skip=True,
            use_accel=not compact,
            strong_residual=False,
            mass_aware_gate=False,
            spectral_branch=not compact,
            hydro_couple=True,
            dual_stream=False,
            spherical_ops=spherical,
            endpoint_preserving=spherical,
            query_independent_trajectory=spherical,
            n_flow_modes=3 if compact else 1,
            cubic_trajectory=compact,
            global_tokens=8 if compact else 0,
            global_token_dim=32,
            endpoint_tangents=(
                arch
                in (
                    "flow_compact_hermite_l",
                    "flow_compact_lagrange_l",
                )
            ),
            base_field_knots=(arch == "flow_compact_lagrange_l"),
        )
    if state is not None:
        model.load_state_dict(state, strict=True)
    return model


def benchmark(
    arch: str,
    *,
    state: dict[str, torch.Tensor] | None,
    device: torch.device,
    batch_size: int,
    height: int,
    width: int,
    warmup: int,
    inference_iterations: int,
    training_iterations: int,
) -> dict[str, object]:
    torch.manual_seed(202707)
    model = _build_model(arch, state).to(device)
    generator = torch.Generator(device=device).manual_seed(202708)
    x0 = torch.randn(
        batch_size,
        24,
        height,
        width,
        device=device,
        generator=generator,
    )
    xT = torch.randn(
        batch_size,
        24,
        height,
        width,
        device=device,
        generator=generator,
    )
    static = torch.randn(
        batch_size,
        3,
        height,
        width,
        device=device,
        generator=generator,
    )
    tau = torch.full((batch_size,), 0.5, device=device)

    def forward() -> torch.Tensor:
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            output = model(x0, xT, tau, static=static)
            return output[0] if isinstance(output, tuple) else output

    model.eval()
    with torch.no_grad():
        for _ in range(warmup):
            forward()
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        inference_ms = _time_cuda(forward, inference_iterations)
        inference_peak = torch.cuda.max_memory_allocated(device)

    training_ms: list[float] = []
    training_peak = 0
    if training_iterations:
        model.train()

        def train_step() -> None:
            model.zero_grad(set_to_none=True)
            prediction = forward()
            prediction.square().mean().backward()

        for _ in range(warmup):
            train_step()
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        training_ms = _time_cuda(train_step, training_iterations)
        training_peak = torch.cuda.max_memory_allocated(device)

    return {
        "arch": arch,
        "parameters": sum(p.numel() for p in model.parameters()),
        "inference_ms_median": statistics.median(inference_ms),
        "inference_ms_all": inference_ms,
        "inference_peak_mib": inference_peak / 2**20,
        "training_ms_median": (
            statistics.median(training_ms) if training_ms else None
        ),
        "training_ms_all": training_ms,
        "training_peak_mib": (
            training_peak / 2**20 if training_ms else None
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--arch",
        required=True,
        choices=(
            "flow_pp3",
            "flow_pp3_compact_l",
            "flow_spherical_ep",
            "flow_compact_vp3",
            "flow_compact_vp3_m",
            "flow_compact_vp3_l",
            "flow_compact_hermite_l",
            "flow_compact_lagrange_l",
            "amt",
            "amt_residual",
            "upr_implicit_global_14m",
            "upr_local_corr_14m",
            "upr_query_match_14m",
            "upr_spherical_implicit_global_14m",
        ),
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--height", type=int, default=360)
    parser.add_argument("--width", type=int, default=720)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--inference-iterations", type=int, default=10)
    parser.add_argument("--training-iterations", type=int, default=3)
    parser.add_argument(
        "--state-blob",
        type=Path,
        help="Optional bare blob or state_dict loaded strictly into the model.",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    if args.batch_size < 1:
        raise SystemExit("--batch-size must be positive")
    device = torch.device(args.device)
    state = None
    state_source = {
        "kind": "deterministic_random_initialization",
        "seed": 202707,
    }
    if args.state_blob is not None:
        loaded = torch.load(
            args.state_blob,
            map_location="cpu",
            weights_only=False,
        )
        state = loaded.get("state_dict", loaded)
        state_source = {
            "kind": "checkpoint_state",
            "path": str(args.state_blob),
            "sha256": _sha256(args.state_blob),
            "tensor_count": len(state),
        }
    repo_root = Path(__file__).resolve().parents[2]
    source_paths = [Path(__file__)]
    if args.arch.startswith("flow_"):
        source_paths.append(
            repo_root
            / "weather_time_interp"
            / "model"
            / "weatherbridge_flow_model.py"
        )
    elif args.arch in {"amt", "amt_residual"}:
        source_paths.append(
            repo_root
            / "weather_time_interp"
            / "model"
            / "weather_amt_model.py"
        )
        if args.arch == "amt_residual":
            source_paths.append(
                repo_root
                / "weather_time_interp"
                / "model"
                / "weather_amt_residual_model.py"
            )
        source_paths.extend(
            sorted(
                (
                    repo_root
                    / "weather_time_interp"
                    / "model"
                    / "amt_upstream"
                ).glob("*.py")
            )
        )
    else:
        source_paths.extend(
            (
                repo_root
                / "weather_time_interp"
                / "model"
                / "weatherbridge_upr_lite_model.py",
                repo_root
                / "weather_time_interp"
                / "model"
                / "weatherbridge_upr_scaled_model.py",
            )
        )
        if args.arch == "upr_spherical_implicit_global_14m":
            source_paths.append(
                repo_root
                / "weather_time_interp"
                / "model"
                / "weatherbridge_upr_spherical_model.py"
            )

    payload = {
        "schema_version": 1,
        "evidence_level": "synthetic_compute_only",
        "device": torch.cuda.get_device_name(device),
        "device_total_mib": (
            torch.cuda.get_device_properties(device).total_memory / 2**20
        ),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "dependency_versions": {
            "torch_harmonics": _optional_dependency_version("torch-harmonics")
        }
        if args.arch in {"flow_pp3", "flow_spherical_ep"}
        else {},
        "precision": "bf16-mixed",
        "batch_size": args.batch_size,
        "height": args.height,
        "width": args.width,
        "warmup": args.warmup,
        "inference_iterations": args.inference_iterations,
        "training_iterations": args.training_iterations,
        "state_source": state_source,
        "code_sha256": {
            str(path.relative_to(repo_root)): _sha256(path)
            for path in source_paths
        },
        "result": benchmark(
            args.arch,
            state=state,
            device=device,
            batch_size=args.batch_size,
            height=args.height,
            width=args.width,
            warmup=args.warmup,
            inference_iterations=args.inference_iterations,
            training_iterations=args.training_iterations,
        ),
    }
    text = json.dumps(payload, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
