#!/usr/bin/env python3
"""Benchmark full-grid inference for capacity-matched checkpoints."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
from pathlib import Path

import torch

from tools.eval.capmatched_loader import load_capmatched_checkpoint


def parse_model(value: str) -> tuple[str, Path]:
    name, separator, checkpoint = value.partition(":")
    if not separator or not name or not checkpoint:
        raise argparse.ArgumentTypeError("expected NAME:CHECKPOINT")
    return name, Path(checkpoint)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _hash_required_sources(
    paths: tuple[Path, ...],
    repo_root: Path,
) -> dict[str, str]:
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "missing benchmark source files: " + ", ".join(missing)
        )
    return {
        str(path.relative_to(repo_root)): sha256(path)
        for path in paths
    }


def benchmark(
    name: str,
    checkpoint: Path,
    *,
    static_path: Path,
    device: torch.device,
    batch_size: int,
    height: int,
    width: int,
    warmup: int,
    iterations: int,
    repeats: int,
    input_seed: int,
    tau_values: tuple[float, ...],
) -> dict:
    model, model_type = load_capmatched_checkpoint(
        checkpoint,
        device,
        static_path=static_path,
    )
    torch.cuda.manual_seed_all(input_seed)
    x0 = torch.randn(
        batch_size,
        24,
        height,
        width,
        device=device,
        dtype=torch.float32,
    )
    xT = torch.randn_like(x0)
    if len(tau_values) == 1:
        tau_batches = [
            torch.full(
                (batch_size,),
                tau_values[0],
                device=device,
            )
        ]
        tau_mode = "constant"
    elif batch_size == 1:
        tau_batches = [
            torch.tensor([value], device=device)
            for value in tau_values
        ]
        tau_mode = "sequential_cycle"
    elif len(tau_values) == batch_size:
        tau_batches = [
            torch.tensor(tau_values, device=device)
        ]
        tau_mode = "mixed_batch"
    else:
        raise ValueError(
            "multiple tau values require batch_size=1 or one tau per sample"
        )
    torch.cuda.reset_peak_memory_stats(device)
    with torch.inference_mode(), torch.autocast(
        device_type="cuda",
        dtype=torch.bfloat16,
    ):
        for index in range(warmup):
            model(x0, xT, tau_batches[index % len(tau_batches)])
        torch.cuda.synchronize(device)
        starts = []
        ends = []
        for _ in range(repeats):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for index in range(iterations):
                model(
                    x0,
                    xT,
                    tau_batches[index % len(tau_batches)],
                )
            end.record()
            starts.append(start)
            ends.append(end)
        torch.cuda.synchronize(device)
    repeat_latency_ms = [
        float(start.elapsed_time(end) / iterations)
        for start, end in zip(starts, ends)
    ]
    latency_ms = float(statistics.median(repeat_latency_ms))
    params = sum(parameter.numel() for parameter in model.net.parameters())
    result = {
        "model_type": model_type,
        "params_m": params / 1e6,
        "batch_size": batch_size,
        "height": height,
        "width": width,
        "warmup": warmup,
        "iterations": iterations,
        "repeats": repeats,
        "latency_ms": latency_ms,
        "latency_mean_ms": float(statistics.fmean(repeat_latency_ms)),
        "latency_std_ms": float(statistics.pstdev(repeat_latency_ms)),
        "latency_p95_ms": float(
            torch.tensor(repeat_latency_ms).quantile(0.95).item()
        ),
        "repeat_latency_ms": repeat_latency_ms,
        "samples_per_second": 1000.0 * batch_size / latency_ms,
        "peak_memory_mib": torch.cuda.max_memory_allocated(device) / 2**20,
        "input_seed": input_seed,
        "tau_values": list(tau_values),
        "tau_mode": tau_mode,
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": sha256(checkpoint),
    }
    del model, x0, xT, tau_batches
    torch.cuda.empty_cache()
    print(
        f"{name}: {latency_ms:.2f} ms, "
        f"{result['samples_per_second']:.2f} samples/s, "
        f"{result['peak_memory_mib']:.0f} MiB"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", action="append", type=parse_model, required=True)
    parser.add_argument(
        "--static-path",
        type=Path,
        default=Path("data/static_features_0p5.pt"),
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--height", type=int, default=360)
    parser.add_argument("--width", type=int, default=720)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--input-seed", type=int, default=2027)
    parser.add_argument(
        "--tau-values",
        default="0.5",
        help=(
            "Comma-separated normalized target times. Multiple values cycle "
            "for batch 1 or form one mixed batch when their count equals the "
            "batch size."
        ),
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        default=Path("metrics/upr_lite_inference_cost.json"),
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for inference benchmarking")
    for label in (
        "batch_size",
        "height",
        "width",
        "warmup",
        "iterations",
        "repeats",
    ):
        value = int(getattr(args, label))
        if value < (0 if label == "warmup" else 1):
            raise ValueError(f"{label} has an invalid value: {value}")
    names = [name for name, _ in args.model]
    if len(names) != len(set(names)):
        raise ValueError("benchmark model names must be unique")
    for name, checkpoint in args.model:
        if not checkpoint.is_file():
            raise FileNotFoundError(f"{name}: missing checkpoint {checkpoint}")
    if not args.static_path.is_file():
        raise FileNotFoundError(f"missing static features {args.static_path}")
    tau_values = tuple(
        float(value)
        for value in args.tau_values.split(",")
        if value.strip()
    )
    if not tau_values or any(
        not math.isfinite(value) or not 0.0 < value < 1.0
        for value in tau_values
    ):
        raise ValueError("tau values must be finite and lie in (0, 1)")
    if (
        len(tau_values) > 1
        and args.batch_size != 1
        and len(tau_values) != args.batch_size
    ):
        raise ValueError(
            "multiple tau values require batch_size=1 or one per sample"
        )
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda:0")
    repo_root = Path(__file__).resolve().parents[2]
    supporting_paths = (
        repo_root / "tools" / "eval" / "capmatched_loader.py",
        repo_root / "tools" / "train" / "train_capacity_matched_6h.py",
        repo_root / "weather_time_interp" / "model"
        / "weatherbridge_upr_lite_model.py",
        repo_root / "weather_time_interp" / "model"
        / "weatherbridge_upr_scaled_model.py",
        repo_root / "weather_time_interp" / "model"
        / "weatherbridge_upr_spherical_model.py",
        repo_root / "weather_time_interp" / "model"
        / "temporal_expert_router.py",
        repo_root / "weather_time_interp" / "model"
        / "weatherbridge_flow_model.py",
        repo_root / "weather_time_interp" / "model"
        / "weather_amt_model.py",
        repo_root / "weather_time_interp" / "model"
        / "amt_upstream" / "feat_enc.py",
        repo_root / "weather_time_interp" / "model"
        / "amt_upstream" / "flow_utils.py",
        repo_root / "weather_time_interp" / "model"
        / "amt_upstream" / "ifrnet.py",
        repo_root / "weather_time_interp" / "model"
        / "amt_upstream" / "multi_flow.py",
        repo_root / "weather_time_interp" / "model"
        / "amt_upstream" / "raft.py",
        repo_root / "weather_time_interp" / "model"
        / "dcae_adaln_model.py",
        repo_root / "weather_time_interp" / "model"
        / "dcae_adaln_skip_model.py",
        repo_root / "legacy" / "scripts" / "train_atm_vfi_12h_oddskip.py",
    )
    output = {
        "schema_version": 2,
        "device": torch.cuda.get_device_name(device),
        "device_capability": list(torch.cuda.get_device_capability(device)),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "input_seed": args.input_seed,
        "tau_values": list(tau_values),
        "static_features": {
            "path": str(args.static_path.resolve()),
            "size_bytes": args.static_path.stat().st_size,
            "sha256": sha256(args.static_path),
        },
        "evaluation_script_sha256": sha256(Path(__file__).resolve()),
        "supporting_code_sha256": _hash_required_sources(
            supporting_paths,
            repo_root,
        ),
        "models": {
            name: benchmark(
                name,
                checkpoint,
                static_path=args.static_path,
                device=device,
                batch_size=args.batch_size,
                height=args.height,
                width=args.width,
                warmup=args.warmup,
                iterations=args.iterations,
                repeats=args.repeats,
                input_seed=args.input_seed,
                tau_values=tau_values,
            )
            for name, checkpoint in args.model
        },
    }
    for name, result in output["models"].items():
        for key in ("params_m", "latency_ms", "peak_memory_mib"):
            value = float(result[key])
            if not math.isfinite(value) or value <= 0.0:
                raise RuntimeError(f"{name}: invalid benchmark {key}={value}")
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out_json.with_name(
        f".{args.out_json.name}.{os.getpid()}.tmp"
    )
    with temporary.open("w") as handle:
        json.dump(output, handle, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(args.out_json)
    print(f"wrote {args.out_json}")


if __name__ == "__main__":
    main()
