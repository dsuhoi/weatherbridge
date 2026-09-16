#!/usr/bin/env python3
"""Full-grid compute benchmark for state-identical UPR geometry variants."""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from weather_time_interp.model.weatherbridge_upr_lite_model import (
    WeatherBridgeUPRLiteModel,
)
from weather_time_interp.model.weatherbridge_upr_scaled_model import (
    upr_scaled_variant_kwargs,
)
from weather_time_interp.model.weatherbridge_upr_spherical_model import (
    WeatherBridgeUPRSphericalModel,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def _build_model(arch: str, state: dict[str, torch.Tensor]) -> torch.nn.Module:
    kwargs = upr_scaled_variant_kwargs("upr_implicit_global_14m")
    if arch == "base":
        model = WeatherBridgeUPRLiteModel(**kwargs)
    elif arch == "spherical":
        model = WeatherBridgeUPRSphericalModel(**kwargs)
    else:
        raise ValueError(f"unknown architecture {arch}")
    model.load_state_dict(state, strict=True)
    return model


def benchmark(
    arch: str,
    state: dict[str, torch.Tensor],
    *,
    device: torch.device,
    height: int,
    width: int,
    warmup: int,
    inference_iterations: int,
    training_iterations: int,
) -> dict[str, object]:
    model = _build_model(arch, state).to(device)
    generator = torch.Generator(device=device).manual_seed(202708)
    x0 = torch.randn(
        1, 24, height, width, device=device, generator=generator
    )
    xT = torch.randn(
        1, 24, height, width, device=device, generator=generator
    )
    static = torch.randn(
        1, 3, height, width, device=device, generator=generator
    )
    tau = torch.tensor([0.5], device=device)

    def forward() -> torch.Tensor:
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            return model(x0, xT, tau, static=static)[0]

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
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--height", type=int, default=360)
    parser.add_argument("--width", type=int, default=720)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--inference-iterations", type=int, default=10)
    parser.add_argument("--training-iterations", type=int, default=3)
    parser.add_argument(
        "--order",
        default="base,spherical",
        help="Comma-separated execution order containing base and spherical.",
    )
    parser.add_argument(
        "--state-blob",
        type=Path,
        help=(
            "Optional bare blob or state_dict shared strictly by both models. "
            "Otherwise use one deterministic random initialization."
        ),
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    execution_order = tuple(args.order.split(","))
    if len(execution_order) != 2 or set(execution_order) != {
        "base",
        "spherical",
    }:
        raise SystemExit("--order must contain base,spherical exactly once")
    device = torch.device(args.device)
    if args.state_blob is None:
        torch.manual_seed(202707)
        reference = WeatherBridgeUPRLiteModel(
            **upr_scaled_variant_kwargs("upr_implicit_global_14m")
        )
        state = reference.state_dict()
        state_source = {
            "kind": "deterministic_random_initialization",
            "seed": 202707,
        }
        del reference
    else:
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
    results = [
        benchmark(
            arch,
            state,
            device=device,
            height=args.height,
            width=args.width,
            warmup=args.warmup,
            inference_iterations=args.inference_iterations,
            training_iterations=args.training_iterations,
        )
        for arch in execution_order
    ]
    by_arch = {row["arch"]: row for row in results}
    base = by_arch["base"]
    spherical = by_arch["spherical"]
    payload = {
        "schema_version": 1,
        "evidence_level": "synthetic_compute_only",
        "device": torch.cuda.get_device_name(device),
        "device_total_mib": (
            torch.cuda.get_device_properties(device).total_memory / 2**20
        ),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "precision": "bf16-mixed",
        "batch_size": 1,
        "height": args.height,
        "width": args.width,
        "warmup": args.warmup,
        "inference_iterations": args.inference_iterations,
        "training_iterations": args.training_iterations,
        "execution_order": list(execution_order),
        "state_source": state_source,
        "code_sha256": {
            path.name: _sha256(path)
            for path in (
                Path(__file__),
                repo_root
                / "weather_time_interp"
                / "model"
                / "weatherbridge_upr_lite_model.py",
                repo_root
                / "weather_time_interp"
                / "model"
                / "weatherbridge_upr_scaled_model.py",
                repo_root
                / "weather_time_interp"
                / "model"
                / "weatherbridge_upr_spherical_model.py",
            )
        },
        "results": results,
        "spherical_relative_to_base": {
            "inference_latency_ratio": (
                spherical["inference_ms_median"]
                / base["inference_ms_median"]
            ),
            "inference_peak_memory_ratio": (
                spherical["inference_peak_mib"]
                / base["inference_peak_mib"]
            ),
            "training_latency_ratio": (
                spherical["training_ms_median"]
                / base["training_ms_median"]
                if args.training_iterations
                else None
            ),
            "training_peak_memory_ratio": (
                spherical["training_peak_mib"]
                / base["training_peak_mib"]
                if args.training_iterations
                else None
            ),
        },
    }
    text = json.dumps(payload, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
