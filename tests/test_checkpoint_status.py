from pathlib import Path

import torch

from tools.train.checkpoint_status import checkpoint_status


def test_checkpoint_status_distinguishes_partial_and_complete(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "last.ckpt"
    torch.save({"epoch": 2, "global_step": 42}, checkpoint)
    partial = checkpoint_status(checkpoint, min_epochs=4)
    complete = checkpoint_status(checkpoint, min_epochs=3)
    assert not partial["complete"]
    assert partial["epochs_completed"] == 3
    assert complete["complete"]


def test_checkpoint_status_handles_missing_file(tmp_path: Path) -> None:
    status = checkpoint_status(tmp_path / "missing.ckpt", min_epochs=8)
    assert not status["exists"]
    assert not status["complete"]


def test_checkpoint_status_rejects_protocol_mismatch(tmp_path: Path) -> None:
    checkpoint = tmp_path / "last.ckpt"
    torch.save(
        {
            "epoch": 7,
            "global_step": 100,
            "hyper_parameters": {
                "arch": "upr_lite_implicit_global",
                "total_steps": 13136,
                "delta_t": 6.0,
            },
        },
        checkpoint,
    )
    matching = checkpoint_status(
        checkpoint,
        min_epochs=8,
        expected_arch="upr_lite_implicit_global",
        expected_total_steps=13136,
        expected_delta_t=6.0,
        min_global_step=100,
    )
    assert matching["complete"]
    insufficient_progress = checkpoint_status(
        checkpoint,
        min_epochs=8,
        expected_arch="upr_lite_implicit_global",
        expected_total_steps=13136,
        expected_delta_t=6.0,
        min_global_step=13136,
    )
    assert not insufficient_progress["complete"]
    assert insufficient_progress["progress_mismatches"] == {
        "global_step": {"minimum": 13136, "actual": 100}
    }
    mismatched = checkpoint_status(
        checkpoint,
        min_epochs=8,
        expected_arch="flow_pp3",
        expected_total_steps=10930,
        expected_delta_t=12.0,
    )
    assert not mismatched["complete"]
    assert set(mismatched["protocol_mismatches"]) == {
        "arch",
        "total_steps",
        "delta_t",
    }


def test_checkpoint_status_validates_training_split_and_recipe(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "last.ckpt"
    protocol = {
        "train_years": [2014, 2015, 2016, 2017, 2018, 2019],
        "val_years": [2020],
        "train_tau_hours": [1, 3, 5],
        "eval_tau_hours": [1, 2, 3, 4, 5],
        "window_hours": 6,
        "batch_size_per_device": 4,
        "accumulate_grad_batches": 4,
        "global_effective_batch_size": 16,
        "samples_per_date_train": 4,
        "samples_per_date_val": 2,
        "seed": 202707,
        "lambda_hf": 0.05,
        "lambda_spec": 0.02,
        "lambda_band": 0.0,
        "spectral_mask_profile": "advected",
        "loss_profile": "uniform",
        "trainable_scope": "all",
        "anchor_swap_probability": 0.0,
        "highpass_boundary": "periodic_lon_replicate_lat",
        "precision": "bf16-mixed",
    }
    torch.save(
        {
            "epoch": 7,
            "global_step": 13136,
            "hyper_parameters": {
                "arch": "upr_implicit_global_14m",
                "total_steps": 13136,
                "delta_t": 6.0,
                "training_seed": 202707,
                "training_protocol": protocol,
            },
        },
        checkpoint,
    )
    expected = {
        "train_years": protocol["train_years"],
        "val_years": protocol["val_years"],
        "train_tau_hours": protocol["train_tau_hours"],
        "eval_tau_hours": protocol["eval_tau_hours"],
        "seed": 202707,
        "batch_size_per_device": 4,
        "accumulate_grad_batches": 4,
        "global_effective_batch_size": 16,
        "lambda_hf": 0.05,
        "lambda_spec": 0.02,
        "lambda_band": 0.0,
        "spectral_mask_profile": "advected",
        "loss_profile": "uniform",
        "trainable_scope": "all",
        "anchor_swap_probability": 0.0,
        "highpass_boundary": "periodic_lon_replicate_lat",
    }

    matching = checkpoint_status(
        checkpoint,
        min_epochs=8,
        expected_training_protocol=expected,
        require_training_protocol=True,
    )

    assert matching["complete"]
    assert matching["training_protocol_errors"] == []

    wrong_objective = checkpoint_status(
        checkpoint,
        min_epochs=8,
        expected_training_protocol={
            "lambda_spec": 0.0,
            "spectral_mask_profile": "all",
            "anchor_swap_probability": 0.5,
        },
        require_training_protocol=True,
    )
    assert not wrong_objective["complete"]
    assert set(wrong_objective["protocol_mismatches"]) == {
        "training_protocol.lambda_spec",
        "training_protocol.spectral_mask_profile",
        "training_protocol.anchor_swap_probability",
    }

    protocol["val_years"] = [2019, 2020]
    torch.save(
        {
            "epoch": 7,
            "global_step": 13136,
            "hyper_parameters": {
                "arch": "upr_implicit_global_14m",
                "total_steps": 13136,
                "delta_t": 6.0,
                "training_seed": 202707,
                "training_protocol": protocol,
            },
        },
        checkpoint,
    )
    leaking = checkpoint_status(
        checkpoint,
        min_epochs=8,
        expected_training_protocol=expected,
        require_training_protocol=True,
    )

    assert not leaking["complete"]
    assert "train_years and val_years overlap" in leaking[
        "training_protocol_errors"
    ]
    assert "training_protocol.val_years" in leaking["protocol_mismatches"]


def test_checkpoint_status_rejects_highpass_boundary_drift(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "last.ckpt"
    protocol = {
        "train_years": [2014, 2015],
        "val_years": [2020],
        "train_tau_hours": [1, 3, 5],
        "eval_tau_hours": [1, 2, 3, 4, 5],
        "window_hours": 6,
        "global_effective_batch_size": 16,
        "samples_per_date_train": 4,
        "samples_per_date_val": 2,
        "seed": 202707,
        "lambda_hf": 0.05,
        "highpass_boundary": "periodic_lon_replicate_lat",
        "precision": "bf16-mixed",
    }
    torch.save(
        {
            "epoch": 7,
            "global_step": 13136,
            "hyper_parameters": {
                "delta_t": 6.0,
                "training_seed": 202707,
                "training_protocol": protocol,
            },
        },
        checkpoint,
    )

    status = checkpoint_status(
        checkpoint,
        min_epochs=8,
        expected_training_protocol={
            "highpass_boundary": "antipodal_vector_parity"
        },
        require_training_protocol=True,
    )

    assert not status["complete"]
    assert "training_protocol.highpass_boundary" in status[
        "protocol_mismatches"
    ]


def test_checkpoint_status_requires_frozen_trainer_hash(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "last.ckpt"
    trainer_sha = "a" * 64
    torch.save(
        {
            "epoch": 7,
            "global_step": 13136,
            "hyper_parameters": {
                "training_code_sha256": {
                    "train_capacity_matched_6h.py": trainer_sha,
                },
            },
        },
        checkpoint,
    )

    matching = checkpoint_status(
        checkpoint,
        min_epochs=8,
        expected_trainer_sha256=trainer_sha,
    )
    mismatched = checkpoint_status(
        checkpoint,
        min_epochs=8,
        expected_trainer_sha256="b" * 64,
    )

    assert matching["complete"]
    assert not mismatched["complete"]
    assert mismatched["protocol_mismatches"][
        "training_code_sha256.trainer"
    ] == {
        "expected": "b" * 64,
        "actual": trainer_sha,
    }


def test_checkpoint_status_validates_complete_accumulation_schedule(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "last.ckpt"
    protocol = {
        "train_years": [2017, 2018, 2019],
        "val_years": [2020],
        "train_tau_hours": [1, 2, 3, 5, 7, 9, 10, 11],
        "eval_tau_hours": list(range(1, 12)),
        "window_hours": 12,
        "global_effective_batch_size": 16,
        "samples_per_date_train": 2,
        "samples_per_date_val": 2,
        "seed": 202707,
        "lambda_hf": 0.0,
        "precision": "bf16-mixed",
        "accumulate_grad_batches": 4,
        "train_batches_per_epoch": 4372,
        "optimizer_steps_per_epoch": 1093,
    }
    payload = {
        "epoch": 9,
        "global_step": 10930,
        "hyper_parameters": {
            "delta_t": 12.0,
            "training_seed": 202707,
            "training_protocol": protocol,
        },
    }
    torch.save(payload, checkpoint)

    matching = checkpoint_status(
        checkpoint,
        min_epochs=10,
        expected_training_protocol={"train_batches_per_epoch": 4372},
        require_training_protocol=True,
    )
    assert matching["complete"]

    protocol["train_batches_per_epoch"] = 4374
    torch.save(payload, checkpoint)
    partial_group = checkpoint_status(
        checkpoint,
        min_epochs=10,
        require_training_protocol=True,
    )
    assert not partial_group["complete"]
    assert any(
        "incomplete accumulation group" in error
        for error in partial_group["training_protocol_errors"]
    )


def test_checkpoint_status_requires_exact_resume_lineage(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "last.ckpt"
    digest = "a" * 64
    torch.save(
        {
            "epoch": 7,
            "global_step": 13136,
            "hyper_parameters": {
                "resume_lineage": {
                    "checkpoint_path": "/tmp/epoch1.ckpt",
                    "checkpoint_size_bytes": 123,
                    "checkpoint_sha256": digest,
                    "epoch": 1,
                    "global_step": 3284,
                    "model": {"arch": "upr_implicit_global_14m"},
                    "previous_training_protocol": {"seed": 202707},
                    "previous_training_input_provenance": {
                        "memmap": {"identity_sha256": "input"}
                    },
                    "previous_training_code_sha256": {
                        "trainer.py": "source"
                    },
                }
            },
        },
        checkpoint,
    )

    matching = checkpoint_status(
        checkpoint,
        min_epochs=8,
        require_resume_lineage=True,
        expected_resume_checkpoint_sha256=digest,
    )
    mismatched = checkpoint_status(
        checkpoint,
        min_epochs=8,
        require_resume_lineage=True,
        expected_resume_checkpoint_sha256="b" * 64,
    )

    assert matching["complete"]
    assert matching["resume_lineage_errors"] == []
    assert not mismatched["complete"]
    assert "resume_lineage.checkpoint_sha256" in mismatched[
        "protocol_mismatches"
    ]


def test_checkpoint_status_rejects_missing_resume_lineage(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "last.ckpt"
    torch.save({"epoch": 7, "global_step": 13136}, checkpoint)

    status = checkpoint_status(
        checkpoint,
        min_epochs=8,
        require_resume_lineage=True,
    )

    assert not status["complete"]
    assert status["resume_lineage_errors"] == ["missing resume_lineage"]


def test_checkpoint_status_accepts_required_resume_ancestor(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "last.ckpt"
    immediate = "a" * 64
    ancestor = "b" * 64
    torch.save(
        {
            "epoch": 7,
            "global_step": 13136,
            "hyper_parameters": {
                "resume_lineage": {
                    "checkpoint_path": "/tmp/epoch3.ckpt",
                    "checkpoint_size_bytes": 456,
                    "checkpoint_sha256": immediate,
                    "epoch": 3,
                    "global_step": 6568,
                    "model": {"arch": "upr_implicit_global_14m"},
                    "previous_training_protocol": {"seed": 202707},
                    "previous_training_input_provenance": {
                        "memmap": {"identity_sha256": "input"}
                    },
                    "previous_training_code_sha256": {
                        "trainer.py": "source"
                    },
                    "previous_resume_lineage": {
                        "checkpoint_sha256": ancestor,
                    },
                }
            },
        },
        checkpoint,
    )

    matching = checkpoint_status(
        checkpoint,
        min_epochs=8,
        require_resume_lineage=True,
        expected_resume_ancestor_sha256=ancestor,
    )
    immediate_only = checkpoint_status(
        checkpoint,
        min_epochs=8,
        require_resume_lineage=True,
        expected_resume_checkpoint_sha256=ancestor,
    )
    missing = checkpoint_status(
        checkpoint,
        min_epochs=8,
        require_resume_lineage=True,
        expected_resume_ancestor_sha256="c" * 64,
    )

    assert matching["complete"]
    assert matching["resume_lineage_sha256_chain"] == [
        immediate,
        ancestor,
    ]
    assert not immediate_only["complete"]
    assert not missing["complete"]
    assert "resume_lineage.ancestor_sha256" in missing[
        "protocol_mismatches"
    ]
