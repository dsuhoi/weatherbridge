import csv
import json

import torch

from tools.train.select_hres_finetune_checkpoint import select_checkpoint


def test_selects_lowest_2020_validation_rmse(tmp_path) -> None:
    logger = tmp_path / "lightning_logs" / "version_0"
    logger.mkdir(parents=True)
    with (logger / "metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=("epoch", "step", "val/rmse_mean"),
        )
        writer.writeheader()
        writer.writerows(
            (
                {"epoch": 0, "step": 10, "val/rmse_mean": 0.4},
                {"epoch": 1, "step": 20, "val/rmse_mean": 0.3},
            )
        )
    protocol = {
        "seed": 202707,
        "data_source": "hres_era5",
        "train_years": [2017, 2018, 2019],
        "val_years": [2020],
        "train_tau_hours": [1, 3, 5],
        "eval_tau_hours": [1, 2, 3, 4, 5],
        "future_analysis_as_input": False,
        "lambda_hf": 0.05,
        "lambda_spec": 0.02,
        "lambda_band": 0.0,
        "lambda_sht": 0.0,
        "spectral_mask_profile": "advected",
    }
    frozen_protocol_path = tmp_path / "protocol.json"
    frozen_protocol_path.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "status": "frozen_before_hres_weight_finetuning",
                "models": {
                    "weatherbridge": {
                        "internal_arch": "flow_pp3",
                        "fine_tuning_objective": {
                            "lambda_highpass": 0.05,
                            "lambda_fft_magnitude": 0.02,
                            "lambda_multiband": 0.0,
                            "lambda_sht": 0.0,
                            "spectral_mask_profile": "advected",
                        },
                    }
                },
            }
        )
    )
    input_provenance = {
        "hres_train": {
            "years": [2017, 2018, 2019],
            "query_hours": [1, 3, 5],
            "selected_initialisations": [
                f"init_train_{index}.bin" for index in range(108)
            ],
            "identity_sha256": "a" * 64,
        },
        "hres_validation": {
            "years": [2020],
            "query_hours": [1, 2, 3, 4, 5],
            "selected_initialisations": [
                f"init_validation_{index}.bin" for index in range(24)
            ],
            "identity_sha256": "b" * 64,
        },
    }
    for epoch, step in ((0, 10), (1, 20)):
        checkpoint_step = step + 1
        torch.save(
            {
                "epoch": epoch,
                "global_step": checkpoint_step,
                "hyper_parameters": {
                    "arch": "flow_pp3",
                    "training_protocol": protocol,
                    "training_input_provenance": input_provenance,
                }
            },
            tmp_path / f"epoch={epoch}-step={checkpoint_step}.ckpt",
        )

    selected = select_checkpoint(
        tmp_path,
        expected_arch="flow_pp3",
        expected_seed=202707,
        protocol_path=frozen_protocol_path,
    )

    assert selected["epoch"] == 1
    assert selected["step"] == 20
    assert selected["validation_rmse_mean"] == 0.3
    assert selected["protocol_sha256"]
    assert selected["checkpoint"]["path"].endswith(
        "epoch=1-step=21.ckpt"
    )
    assert selected["checkpoint"]["global_step"] == 21
    assert selected["training_input_identity_sha256"] == {
        "optimization": "a" * 64,
        "validation": "b" * 64,
    }
