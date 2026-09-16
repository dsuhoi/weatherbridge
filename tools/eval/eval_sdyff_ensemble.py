#!/usr/bin/env python3
"""Matched S-DYff ensemble means using the journal's RMSE/ACC evaluator.

All sizes share prefixes of one stochastic-depth sample stream. Dropout stays
disabled, exactly as in the published single-output adaptation. This measures
averaging, not a change to the sampler or original DYffusion reproduction.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

from tools.eval.batch_eval_12h_memmap_fast import FastBatchModelRunner12h, base


def prefix_means(predict, sizes):
    """Accumulate in float32 without retaining every ensemble member."""
    sizes = tuple(sizes)
    if not sizes or sizes != tuple(sorted(set(sizes))) or sizes[0] < 1:
        raise ValueError("ensemble sizes must be sorted, unique positive integers")
    total = None
    outputs = {}
    for member in range(1, sizes[-1] + 1):
        prediction = predict().float()
        if not torch.isfinite(prediction).all():
            raise ValueError(f"non-finite S-DYff member {member}")
        if total is None:
            total = prediction.clone()
        else:
            total.add_(prediction)
        if member in sizes:
            outputs[member] = total / member
    return outputs


class EnsembleRunner(FastBatchModelRunner12h):
    sizes = (1, 4, 16, 21)
    seed = 20260914

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.model_type != "sdyff_dyffusion_residual_linear":
            raise ValueError("S-DYff-ENS requires the manuscript S-DYff adaptation")
        self.model.eval()
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)

    def method_names(self):
        return ["model", "bilinear"] + [f"ens{n}" for n in self.sizes[:-1]]

    def predictions_for_metrics(self, x0, xT, tau_h, cond, static):
        means = prefix_means(
            lambda: self._forward_model(
                x0=x0, xT=xT, tau_h=tau_h, cond=cond, static=static,
            ), self.sizes,
        )
        return (
            means[self.sizes[-1]], base._bilinear_2anchor(x0, xT, tau_h),
            *(means[n] for n in self.sizes[:-1]),
        )

    def _finalize_payload(self):
        payload = super()._finalize_payload()
        inner = getattr(self.model, "model", self.model)
        payload["ensemble_protocol"] = {
            "label": "S-DYff-ENS",
            "sizes": list(self.sizes),
            "method_sizes": {"model": self.sizes[-1], **{
                f"ens{n}": n for n in self.sizes[:-1]
            }},
            "seed": self.seed,
            "batch_size": self.cfg.batch_size,
            "sampling": "eval_mode_stochastic_depth_only",
            "dropout_enabled": False,
            "drop_path_rate": inner.drop_path,
            "refinement_steps": inner.n_inference_steps,
            "shared_sample_prefixes": True,
            "reduction": "average_fields_before_rmse_and_acc",
            "single_draw_control": "ens1; paired rerun, not the archived draw",
            "script_sha256": base._sha256_file(Path(__file__)),
            "model_source_sha256": {
                name: base._sha256_file(Path(__file__).resolve().parents[2] / name)
                for name in (
                    "weather_time_interp/model/sdyff_baseline_model.py",
                    "weather_time_interp/model/sdyff_dyffusion_model.py",
                    "legacy/scripts/trainer_weather_hermite.py",
                )
            },
            "torch_version": torch.__version__,
            "parameter_dtype": str(next(self.model.parameters()).dtype),
        }
        return payload


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--ensemble-sizes", default="1,4,16,21")
    parser.add_argument("--ensemble-seed", type=int, default=20260914)
    extra, remaining = parser.parse_known_args()
    sizes = tuple(int(n) for n in extra.ensemble_sizes.split(","))
    if sizes != tuple(sorted(set(sizes))) or not sizes or sizes[0] != 1 or sizes[-1] < 2:
        parser.error("sizes must start with 1 and increase strictly to at least 2")
    EnsembleRunner.sizes = sizes
    EnsembleRunner.seed = extra.ensemble_seed
    sys.argv = [sys.argv[0], *remaining]
    base.BatchModelRunner12h = EnsembleRunner
    base.main()


if __name__ == "__main__":
    main()
