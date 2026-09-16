#!/usr/bin/env python3
"""Synthetic latency and memory benchmark for UPR motion-pyramid variants."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import statistics
from pathlib import Path

import torch

from weather_time_interp.model.weatherbridge_upr_lite_model import (
    WeatherBridgeUPRLiteModel,
    upr_lite_variant_kwargs,
)


VARIANTS = (
    "upr_lite_implicit_global",
    "upr_lite_implicit_global_q4",
    "flow_pp3",
)


def _build_model(arch: str) -> torch.nn.Module:
    if arch == "flow_pp3":
        from weather_time_interp.model.weatherbridge_flow_model import (
            WeatherBridgeModel,
        )

        return WeatherBridgeModel(
            in_channels=24,
            out_channels=24,
            n_static_features=3,
            hidden=72,
            n_levels=3,
            use_skip=True,
            gated_skip=True,
            use_accel=True,
            strong_residual=False,
            mass_aware_gate=False,
            spectral_branch=True,
            hydro_couple=True,
            dual_stream=False,
        )
    return WeatherBridgeUPRLiteModel(
        **upr_lite_variant_kwargs(arch)
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


def benchmark(
    arch: str,
    *,
    device: torch.device,
    height: int,
    width: int,
    warmup: int,
    inference_iterations: int,
    training_iterations: int,
) -> dict:
    torch.manual_seed(202707)
    model = _build_model(arch).to(device)
    torch.manual_seed(202708)
    x0 = torch.randn(1, 24, height, width, device=device)
    xT = torch.randn(1, 24, height, width, device=device)
    static = torch.randn(1, 3, height, width, device=device)
    tau = torch.tensor([0.5], device=device)

    def forward() -> torch.Tensor:
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            return model(x0, xT, tau, static=static)[0]

    model.eval()
    with torch.no_grad():
        for _ in range(warmup):
            forward()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(device)
        inference_ms = _time_cuda(forward, inference_iterations)
        inference_peak = torch.cuda.max_memory_allocated(device)

    training_ms: list[float] = []
    training_peak = 0
    if training_iterations > 0:
        model.train()

        def train_step() -> None:
            model.zero_grad(set_to_none=True)
            prediction = forward()
            prediction.square().mean().backward()

        for _ in range(warmup):
            train_step()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(device)
        training_ms = _time_cuda(train_step, training_iterations)
        training_peak = torch.cuda.max_memory_allocated(device)

    return {
        "arch": arch,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "pyramid_divisors": (
            list(model.pyramid_divisors)
            if hasattr(model, "pyramid_divisors")
            else None
        ),
        "inference_ms_median": statistics.median(inference_ms),
        "inference_ms_all": inference_ms,
        "inference_peak_mib": inference_peak / 2**20,
        "training_ms_median": (
            statistics.median(training_ms) if training_ms else None
        ),
        "training_ms_all": training_ms,
        "training_peak_mib": training_peak / 2**20 if training_ms else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--height", type=int, default=360)
    parser.add_argument("--width", type=int, default=720)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--inference-iterations", type=int, default=10)
    parser.add_argument("--training-iterations", type=int, default=3)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    device = torch.device(args.device)
    repo_root = Path(__file__).resolve().parents[2]
    payload = {
        "schema_version": 1,
        "device": torch.cuda.get_device_name(device),
        "device_total_mib": (
            torch.cuda.get_device_properties(device).total_memory / 2**20
        ),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "dependency_versions": {
            "torch_harmonics": importlib.metadata.version("torch-harmonics"),
        },
        "precision": "bf16-mixed",
        "batch_size": 1,
        "height": args.height,
        "width": args.width,
        "warmup": args.warmup,
        "inference_iterations": args.inference_iterations,
        "training_iterations": args.training_iterations,
        "code_sha256": {
            "benchmark_upr_pyramid_compute.py": _sha256(Path(__file__)),
            "weatherbridge_upr_lite_model.py": _sha256(
                repo_root
                / "weather_time_interp"
                / "model"
                / "weatherbridge_upr_lite_model.py"
            ),
            "weatherbridge_flow_model.py": _sha256(
                repo_root
                / "weather_time_interp"
                / "model"
                / "weatherbridge_flow_model.py"
            ),
        },
        "results": [
            benchmark(
                arch,
                device=device,
                height=args.height,
                width=args.width,
                warmup=args.warmup,
                inference_iterations=args.inference_iterations,
                training_iterations=args.training_iterations,
            )
            for arch in VARIANTS
        ],
    }
    baseline = payload["results"][0]
    q4 = payload["results"][1]
    pp3 = payload["results"][2]
    payload["q4_relative_to_half"] = {
        "inference_latency_ratio": (
            q4["inference_ms_median"] / baseline["inference_ms_median"]
        ),
        "inference_peak_memory_ratio": (
            q4["inference_peak_mib"] / baseline["inference_peak_mib"]
        ),
        "training_latency_ratio": (
            q4["training_ms_median"] / baseline["training_ms_median"]
            if q4["training_ms_median"] is not None
            else None
        ),
        "training_peak_memory_ratio": (
            q4["training_peak_mib"] / baseline["training_peak_mib"]
            if q4["training_peak_mib"] is not None
            else None
        ),
    }
    payload["q4_relative_to_pp3"] = {
        "parameter_ratio": q4["parameters"] / pp3["parameters"],
        "inference_latency_ratio": (
            q4["inference_ms_median"] / pp3["inference_ms_median"]
        ),
        "inference_peak_memory_ratio": (
            q4["inference_peak_mib"] / pp3["inference_peak_mib"]
        ),
        "training_latency_ratio": (
            q4["training_ms_median"] / pp3["training_ms_median"]
            if q4["training_ms_median"] is not None
            else None
        ),
        "training_peak_memory_ratio": (
            q4["training_peak_mib"] / pp3["training_peak_mib"]
            if q4["training_peak_mib"] is not None
            else None
        ),
    }
    text = json.dumps(payload, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
