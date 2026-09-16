#!/usr/bin/env python3
"""Select a provenance-checked HRES fine-tuning checkpoint on 2020 RMSE."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
from pathlib import Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validation_rows(run_dir: Path) -> list[dict[str, float | int | Path]]:
    records: list[dict[str, float | int | Path]] = []
    for path in sorted(run_dir.glob("lightning_logs/version_*/metrics.csv")):
        with path.open(newline="") as stream:
            for row in csv.DictReader(stream):
                raw_metric = row.get("val/rmse_mean", "")
                if not raw_metric:
                    continue
                metric = float(raw_metric)
                if not math.isfinite(metric):
                    raise ValueError(f"{path}: non-finite validation RMSE")
                records.append(
                    {
                        "epoch": int(float(row["epoch"])),
                        "step": int(float(row["step"])),
                        "rmse_mean": metric,
                        "metrics_csv": path,
                    }
                )
    if not records:
        raise ValueError(f"{run_dir}: no validation RMSE records")
    return records


def select_checkpoint(
    run_dir: Path,
    *,
    expected_arch: str,
    expected_seed: int,
    protocol_path: Path,
) -> dict[str, object]:
    frozen_protocol = json.loads(protocol_path.read_text())
    matching_models = [
        record
        for record in frozen_protocol.get("models", {}).values()
        if record.get("internal_arch") == expected_arch
    ]
    if (
        frozen_protocol.get("schema_version") != 2
        or frozen_protocol.get("status")
        != "frozen_before_hres_weight_finetuning"
        or len(matching_models) != 1
    ):
        raise ValueError("invalid frozen HRES protocol")
    expected_objective = matching_models[0].get(
        "fine_tuning_objective",
        {},
    )
    selected = min(
        _validation_rows(run_dir),
        key=lambda row: (row["rmse_mean"], row["epoch"], row["step"]),
    )
    epoch = int(selected["epoch"])
    step = int(selected["step"])
    matches: list[Path] = []
    checkpoint_progress: dict[Path, tuple[int, int]] = {}
    for path in sorted(run_dir.glob(f"*epoch={epoch}-step=*.ckpt")):
        match = re.search(
            r"epoch=(\d+)-step=(\d+)(?:-v\d+)?\.ckpt$",
            path.name,
        )
        if match is None:
            continue
        checkpoint_epoch, checkpoint_step = map(int, match.groups())
        # Lightning logs validation on the last zero-based batch step, while
        # ModelCheckpoint names the completed optimizer step one count later.
        if checkpoint_epoch == epoch and checkpoint_step in (step, step + 1):
            matches.append(path)
            checkpoint_progress[path] = (checkpoint_epoch, checkpoint_step)
    if len(matches) != 1:
        raise ValueError(
            f"{run_dir}: expected one checkpoint for epoch={epoch}, "
            f"metrics_step={step} (checkpoint step {step} or {step + 1}); "
            f"found {matches}"
        )
    checkpoint_path = matches[0]

    import torch

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    checkpoint_epoch, checkpoint_step = checkpoint_progress[checkpoint_path]
    if (
        int(checkpoint.get("epoch", -1)) != checkpoint_epoch
        or int(checkpoint.get("global_step", -1)) != checkpoint_step
    ):
        raise ValueError("selected checkpoint progress metadata mismatch")
    hparams = checkpoint.get("hyper_parameters", {})
    protocol = hparams.get("training_protocol", {})
    input_provenance = hparams.get("training_input_provenance", {})
    if hparams.get("arch") != expected_arch:
        raise ValueError("selected checkpoint architecture mismatch")
    if protocol.get("seed") != expected_seed:
        raise ValueError("selected checkpoint seed mismatch")
    if (
        protocol.get("data_source") != "hres_era5"
        or protocol.get("train_years") != [2017, 2018, 2019]
        or protocol.get("val_years") != [2020]
        or protocol.get("train_tau_hours") != [1, 3, 5]
        or protocol.get("eval_tau_hours") != [1, 2, 3, 4, 5]
        or protocol.get("future_analysis_as_input") is not False
    ):
        raise ValueError("selected checkpoint violates the frozen HRES protocol")
    actual_objective = {
        "lambda_highpass": protocol.get("lambda_hf"),
        "lambda_fft_magnitude": protocol.get("lambda_spec"),
        "lambda_multiband": protocol.get("lambda_band"),
        "lambda_sht": protocol.get("lambda_sht"),
        "spectral_mask_profile": protocol.get("spectral_mask_profile"),
    }
    if actual_objective != expected_objective:
        raise ValueError("selected checkpoint has the wrong fine-tuning objective")
    train_input = input_provenance.get("hres_train", {})
    validation_input = input_provenance.get("hres_validation", {})
    train_initialisations = set(train_input.get("selected_initialisations", []))
    validation_initialisations = set(
        validation_input.get("selected_initialisations", [])
    )
    if (
        train_input.get("years") != [2017, 2018, 2019]
        or validation_input.get("years") != [2020]
        or train_input.get("query_hours") != [1, 3, 5]
        or validation_input.get("query_hours") != [1, 2, 3, 4, 5]
        or len(train_initialisations) != 108
        or len(validation_initialisations) != 24
        or train_initialisations & validation_initialisations
        or not train_input.get("identity_sha256")
        or not validation_input.get("identity_sha256")
    ):
        raise ValueError("selected checkpoint has invalid HRES data lineage")
    metrics_csv = Path(selected["metrics_csv"])
    return {
        "schema_version": 1,
        "status": "selected_on_2020_validation",
        "criterion": "minimum_val_rmse_mean_all_five_interior_hours",
        "protocol_sha256": _sha256(protocol_path),
        "fine_tuning_objective": expected_objective,
        "architecture": expected_arch,
        "seed": expected_seed,
        "epoch": epoch,
        "step": step,
        "validation_rmse_mean": selected["rmse_mean"],
        "checkpoint": {
            "path": str(checkpoint_path.resolve()),
            "sha256": _sha256(checkpoint_path),
            "size_bytes": checkpoint_path.stat().st_size,
            "epoch": checkpoint_epoch,
            "global_step": checkpoint_step,
        },
        "metrics_csv": {
            "path": str(metrics_csv.resolve()),
            "sha256": _sha256(metrics_csv),
        },
        "training_input_identity_sha256": {
            "optimization": train_input["identity_sha256"],
            "validation": validation_input["identity_sha256"],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--expected-arch", required=True)
    parser.add_argument("--expected-seed", type=int, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = select_checkpoint(
        args.run_dir,
        expected_arch=args.expected_arch,
        expected_seed=args.expected_seed,
        protocol_path=args.protocol,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, args.output)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
