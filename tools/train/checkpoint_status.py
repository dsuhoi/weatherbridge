#!/usr/bin/env python3
"""Check whether a Lightning checkpoint reached the requested epoch budget."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import string
from typing import Any


_SEQUENCE_PROTOCOL_KEYS = {
    "train_years",
    "val_years",
    "train_tau_hours",
    "eval_tau_hours",
}


def _integer_sequence(value: Any) -> tuple[int, ...] | None:
    if isinstance(value, (str, bytes, dict)) or value is None:
        return None
    try:
        return tuple(int(item) for item in value)
    except (TypeError, ValueError, OverflowError):
        return None


def _training_protocol_errors(
    protocol: dict[str, Any],
    *,
    delta_t: Any,
    training_seed: Any,
) -> list[str]:
    errors: list[str] = []
    sequences = {
        key: _integer_sequence(protocol.get(key))
        for key in _SEQUENCE_PROTOCOL_KEYS
    }
    for key, values in sequences.items():
        if values is None or not values or len(values) != len(set(values)):
            errors.append(f"{key} must be a non-empty unique integer list")
    train_years = sequences["train_years"]
    val_years = sequences["val_years"]
    train_taus = sequences["train_tau_hours"]
    eval_taus = sequences["eval_tau_hours"]
    if (
        train_years is not None
        and val_years is not None
        and set(train_years).intersection(val_years)
    ):
        errors.append("train_years and val_years overlap")
    if (
        train_taus is not None
        and eval_taus is not None
        and not set(train_taus).issubset(eval_taus)
    ):
        errors.append("train_tau_hours is not a subset of eval_tau_hours")
    try:
        window_hours = int(protocol.get("window_hours"))
        delta_matches = math.isclose(
            float(delta_t),
            float(window_hours),
            rel_tol=0.0,
            abs_tol=1e-9,
        )
    except (TypeError, ValueError, OverflowError):
        window_hours = 0
        delta_matches = False
    if window_hours < 2:
        errors.append("window_hours must be at least 2")
    elif not delta_matches:
        errors.append("window_hours does not match delta_t")
    for label, values in (
        ("train_tau_hours", train_taus),
        ("eval_tau_hours", eval_taus),
    ):
        if values is not None and any(
            tau <= 0 or tau >= window_hours
            for tau in values
        ):
            errors.append(f"{label} contains an endpoint or out-of-range tau")
    for key in (
        "global_effective_batch_size",
        "samples_per_date_train",
        "samples_per_date_val",
    ):
        try:
            positive = int(protocol.get(key)) > 0
        except (TypeError, ValueError, OverflowError):
            positive = False
        if not positive:
            errors.append(f"{key} must be positive")
    train_batches = protocol.get("train_batches_per_epoch")
    if train_batches is not None:
        try:
            train_batches = int(train_batches)
            accumulate = int(protocol.get("accumulate_grad_batches"))
            optimizer_steps = int(
                protocol.get("optimizer_steps_per_epoch")
            )
        except (TypeError, ValueError, OverflowError):
            errors.append("training batch schedule must contain integers")
        else:
            if train_batches <= 0:
                errors.append("train_batches_per_epoch must be positive")
            if accumulate <= 0 or train_batches % accumulate:
                errors.append(
                    "train_batches_per_epoch has an incomplete "
                    "accumulation group"
                )
            if optimizer_steps != train_batches // max(accumulate, 1):
                errors.append(
                    "optimizer_steps_per_epoch does not match "
                    "the accumulation schedule"
                )
    try:
        seed_matches = int(protocol.get("seed")) == int(training_seed)
    except (TypeError, ValueError, OverflowError):
        seed_matches = False
    if not seed_matches:
        errors.append("training protocol seed does not match training_seed")
    objective_keys = ("lambda_hf",) + tuple(
        key for key in ("lambda_spec", "lambda_band") if key in protocol
    )
    for key in objective_keys:
        try:
            value = float(protocol.get(key))
        except (TypeError, ValueError, OverflowError):
            value = math.nan
        if not math.isfinite(value) or value < 0.0:
            errors.append(f"{key} must be finite and non-negative")
    if "anchor_swap_probability" in protocol:
        try:
            swap_probability = float(protocol["anchor_swap_probability"])
        except (TypeError, ValueError, OverflowError):
            swap_probability = math.nan
        if (
            not math.isfinite(swap_probability)
            or not 0.0 <= swap_probability <= 1.0
        ):
            errors.append("anchor_swap_probability must lie in [0, 1]")
    highpass_boundary = protocol.get("highpass_boundary")
    if highpass_boundary is not None and highpass_boundary not in {
        "periodic_lon_replicate_lat",
        "antipodal_vector_parity",
    }:
        errors.append("highpass_boundary is unsupported")
    if not isinstance(protocol.get("precision"), str) or not protocol.get(
        "precision"
    ):
        errors.append("precision must be a non-empty string")
    return errors


def _protocol_value_matches(
    key: str,
    actual: Any,
    expected: Any,
) -> bool:
    if key in _SEQUENCE_PROTOCOL_KEYS:
        return _integer_sequence(actual) == _integer_sequence(expected)
    if key in {
        "lambda_hf",
        "lambda_spec",
        "lambda_band",
        "anchor_swap_probability",
    }:
        try:
            return math.isclose(
                float(actual),
                float(expected),
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        except (TypeError, ValueError, OverflowError):
            return False
    return actual == expected


def _resume_lineage_errors(lineage: Any) -> list[str]:
    if not isinstance(lineage, dict):
        return ["missing resume_lineage"]
    errors: list[str] = []
    path = lineage.get("checkpoint_path")
    if not isinstance(path, str) or not path:
        errors.append("resume_lineage.checkpoint_path must be non-empty")
    try:
        size_valid = int(lineage.get("checkpoint_size_bytes")) > 0
    except (TypeError, ValueError, OverflowError):
        size_valid = False
    if not size_valid:
        errors.append("resume_lineage.checkpoint_size_bytes must be positive")
    digest = lineage.get("checkpoint_sha256")
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in string.hexdigits for character in digest)
    ):
        errors.append("resume_lineage.checkpoint_sha256 must be SHA-256")
    for key in ("epoch", "global_step"):
        try:
            valid = int(lineage.get(key)) >= 0
        except (TypeError, ValueError, OverflowError):
            valid = False
        if not valid:
            errors.append(f"resume_lineage.{key} must be non-negative")
    for key in (
        "model",
        "previous_training_protocol",
        "previous_training_input_provenance",
        "previous_training_code_sha256",
    ):
        if not isinstance(lineage.get(key), dict) or not lineage.get(key):
            errors.append(f"resume_lineage.{key} must be a non-empty mapping")
    return errors


def _resume_lineage_sha_chain(
    lineage: Any,
) -> tuple[tuple[str, ...], list[str]]:
    hashes: list[str] = []
    errors: list[str] = []
    current = lineage
    for depth in range(64):
        if not isinstance(current, dict):
            errors.append(
                f"resume lineage ancestor depth {depth} is not a mapping"
            )
            break
        digest = current.get("checkpoint_sha256")
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(
                character not in string.hexdigits
                for character in digest
            )
        ):
            errors.append(
                f"resume lineage ancestor depth {depth} lacks a valid SHA-256"
            )
            break
        hashes.append(digest.lower())
        previous = current.get("previous_resume_lineage")
        if previous is None:
            break
        current = previous
    else:
        errors.append("resume lineage exceeds 64 ancestors")
    return tuple(hashes), errors


def checkpoint_status(
    path: Path,
    min_epochs: int,
    *,
    expected_arch: str | None = None,
    expected_total_steps: int | None = None,
    expected_delta_t: float | None = None,
    min_global_step: int | None = None,
    expected_training_protocol: dict[str, object] | None = None,
    require_training_protocol: bool = False,
    require_resume_lineage: bool = False,
    expected_resume_checkpoint_sha256: str | None = None,
    expected_resume_ancestor_sha256: str | None = None,
    expected_trainer_sha256: str | None = None,
) -> dict[str, object]:
    status: dict[str, object] = {
        "path": str(path),
        "min_epochs": int(min_epochs),
        "exists": path.is_file() and path.stat().st_size > 0,
        "complete": False,
    }
    if not status["exists"]:
        return status
    import torch

    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except (OSError, RuntimeError, EOFError) as error:
        status["error"] = f"{type(error).__name__}: {error}"
        return status
    epoch = int(checkpoint.get("epoch", -1))
    status["epoch"] = epoch
    status["epochs_completed"] = epoch + 1
    status["global_step"] = int(checkpoint.get("global_step", -1))
    hparams = dict(checkpoint.get("hyper_parameters", {}))
    actual = {
        "arch": hparams.get("arch"),
        "total_steps": hparams.get("total_steps"),
        "delta_t": hparams.get("delta_t"),
    }
    status["protocol"] = actual
    mismatches: dict[str, dict[str, object]] = {}
    if expected_arch is not None and actual["arch"] != expected_arch:
        mismatches["arch"] = {
            "expected": expected_arch,
            "actual": actual["arch"],
        }
    if (
        expected_total_steps is not None
        and actual["total_steps"] != expected_total_steps
    ):
        mismatches["total_steps"] = {
            "expected": expected_total_steps,
            "actual": actual["total_steps"],
        }
    if expected_delta_t is not None:
        try:
            delta_matches = math.isclose(
                float(actual["delta_t"]),
                expected_delta_t,
                rel_tol=0.0,
                abs_tol=1e-9,
            )
        except (TypeError, ValueError):
            delta_matches = False
        if not delta_matches:
            mismatches["delta_t"] = {
                "expected": expected_delta_t,
                "actual": actual["delta_t"],
            }
    training_protocol = hparams.get("training_protocol")
    protocol_errors: list[str] = []
    if isinstance(training_protocol, dict):
        protocol_errors = _training_protocol_errors(
            training_protocol,
            delta_t=actual["delta_t"],
            training_seed=hparams.get("training_seed"),
        )
    elif require_training_protocol or expected_training_protocol:
        protocol_errors.append("missing training_protocol")
        training_protocol = {}
    else:
        training_protocol = None
    status["training_protocol"] = training_protocol
    status["training_protocol_errors"] = protocol_errors
    training_code_sha256 = hparams.get("training_code_sha256")
    status["training_code_sha256"] = training_code_sha256
    if expected_trainer_sha256 is not None:
        actual_trainer_sha256 = (
            training_code_sha256.get("train_capacity_matched_6h.py")
            if isinstance(training_code_sha256, dict)
            else None
        )
        if actual_trainer_sha256 != expected_trainer_sha256:
            mismatches["training_code_sha256.trainer"] = {
                "expected": expected_trainer_sha256,
                "actual": actual_trainer_sha256,
            }
    resume_lineage = hparams.get("resume_lineage")
    resume_lineage_errors = (
        _resume_lineage_errors(resume_lineage)
        if (
            require_resume_lineage
            or expected_resume_checkpoint_sha256
            or expected_resume_ancestor_sha256
        )
        else []
    )
    resume_sha_chain: tuple[str, ...] = ()
    if expected_resume_ancestor_sha256 is not None:
        resume_sha_chain, ancestor_errors = _resume_lineage_sha_chain(
            resume_lineage
        )
        resume_lineage_errors.extend(ancestor_errors)
    status["resume_lineage"] = resume_lineage
    status["resume_lineage_sha256_chain"] = list(resume_sha_chain)
    status["resume_lineage_errors"] = resume_lineage_errors
    if expected_training_protocol:
        assert isinstance(training_protocol, dict)
        for key, expected in expected_training_protocol.items():
            current = training_protocol.get(key)
            if not _protocol_value_matches(key, current, expected):
                mismatches[f"training_protocol.{key}"] = {
                    "expected": expected,
                    "actual": current,
                }
    if expected_resume_checkpoint_sha256 is not None:
        actual_resume_sha = (
            resume_lineage.get("checkpoint_sha256")
            if isinstance(resume_lineage, dict)
            else None
        )
        if actual_resume_sha != expected_resume_checkpoint_sha256:
            mismatches["resume_lineage.checkpoint_sha256"] = {
                "expected": expected_resume_checkpoint_sha256,
                "actual": actual_resume_sha,
            }
    if expected_resume_ancestor_sha256 is not None:
        expected_ancestor = expected_resume_ancestor_sha256.lower()
        if expected_ancestor not in resume_sha_chain:
            mismatches["resume_lineage.ancestor_sha256"] = {
                "expected": expected_ancestor,
                "actual": list(resume_sha_chain),
            }
    status["protocol_mismatches"] = mismatches
    progress_mismatches: dict[str, dict[str, int]] = {}
    if (
        min_global_step is not None
        and status["global_step"] < min_global_step
    ):
        progress_mismatches["global_step"] = {
            "minimum": int(min_global_step),
            "actual": int(status["global_step"]),
        }
    status["progress_mismatches"] = progress_mismatches
    status["complete"] = (
        epoch + 1 >= min_epochs
        and not mismatches
        and not protocol_errors
        and not resume_lineage_errors
        and not progress_mismatches
    )
    return status


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--min-epochs", type=int, required=True)
    parser.add_argument("--expected-arch")
    parser.add_argument("--expected-total-steps", type=int)
    parser.add_argument("--expected-delta-t", type=float)
    parser.add_argument("--min-global-step", type=int)
    parser.add_argument("--require-training-protocol", action="store_true")
    parser.add_argument("--expected-train-years")
    parser.add_argument("--expected-val-years")
    parser.add_argument("--expected-train-taus")
    parser.add_argument("--expected-eval-taus")
    parser.add_argument("--expected-seed", type=int)
    parser.add_argument("--expected-effective-batch-size", type=int)
    parser.add_argument("--expected-batch-size-per-device", type=int)
    parser.add_argument("--expected-accumulate-grad-batches", type=int)
    parser.add_argument("--expected-train-batches-per-epoch", type=int)
    parser.add_argument("--expected-optimizer-steps-per-epoch", type=int)
    parser.add_argument("--expected-samples-per-date-train", type=int)
    parser.add_argument("--expected-samples-per-date-val", type=int)
    parser.add_argument("--expected-lambda-hf", type=float)
    parser.add_argument("--expected-lambda-spec", type=float)
    parser.add_argument("--expected-lambda-band", type=float)
    parser.add_argument("--expected-spectral-mask-profile")
    parser.add_argument("--expected-loss-profile")
    parser.add_argument("--expected-trainable-scope")
    parser.add_argument("--expected-anchor-swap-probability", type=float)
    parser.add_argument("--expected-highpass-boundary")
    parser.add_argument("--expected-precision")
    parser.add_argument("--expected-trainer-sha256")
    parser.add_argument("--require-resume-lineage", action="store_true")
    parser.add_argument("--expected-resume-checkpoint-sha256")
    parser.add_argument("--expected-resume-ancestor-sha256")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    def parse_ints(value: str | None) -> list[int] | None:
        if value is None:
            return None
        return [
            int(item)
            for item in value.split(",")
            if item.strip()
        ]

    expected_protocol = {
        key: value
        for key, value in {
            "train_years": parse_ints(args.expected_train_years),
            "val_years": parse_ints(args.expected_val_years),
            "train_tau_hours": parse_ints(args.expected_train_taus),
            "eval_tau_hours": parse_ints(args.expected_eval_taus),
            "seed": args.expected_seed,
            "global_effective_batch_size": (
                args.expected_effective_batch_size
            ),
            "batch_size_per_device": (
                args.expected_batch_size_per_device
            ),
            "accumulate_grad_batches": (
                args.expected_accumulate_grad_batches
            ),
            "train_batches_per_epoch": (
                args.expected_train_batches_per_epoch
            ),
            "optimizer_steps_per_epoch": (
                args.expected_optimizer_steps_per_epoch
            ),
            "samples_per_date_train": (
                args.expected_samples_per_date_train
            ),
            "samples_per_date_val": args.expected_samples_per_date_val,
            "lambda_hf": args.expected_lambda_hf,
            "lambda_spec": args.expected_lambda_spec,
            "lambda_band": args.expected_lambda_band,
            "spectral_mask_profile": args.expected_spectral_mask_profile,
            "loss_profile": args.expected_loss_profile,
            "trainable_scope": args.expected_trainable_scope,
            "anchor_swap_probability": (
                args.expected_anchor_swap_probability
            ),
            "highpass_boundary": args.expected_highpass_boundary,
            "precision": args.expected_precision,
        }.items()
        if value is not None
    }
    status = checkpoint_status(
        args.checkpoint,
        args.min_epochs,
        expected_arch=args.expected_arch,
        expected_total_steps=args.expected_total_steps,
        expected_delta_t=args.expected_delta_t,
        min_global_step=args.min_global_step,
        expected_training_protocol=expected_protocol or None,
        require_training_protocol=args.require_training_protocol,
        require_resume_lineage=args.require_resume_lineage,
        expected_resume_checkpoint_sha256=(
            args.expected_resume_checkpoint_sha256
        ),
        expected_resume_ancestor_sha256=(
            args.expected_resume_ancestor_sha256
        ),
        expected_trainer_sha256=args.expected_trainer_sha256,
    )
    if not args.quiet:
        print(json.dumps(status))
    raise SystemExit(0 if status["complete"] else 1)


if __name__ == "__main__":
    main()
