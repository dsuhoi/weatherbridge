from __future__ import annotations

from pathlib import Path

import torch

from tools.train.merge_flow_field_heads import (
    FIELD_INDICES,
    HYDRO_KEYS,
    ROW_KEYS,
    merge_field_heads,
)


def _checkpoint(path: Path, offset: float) -> None:
    state = {}
    for key in ROW_KEYS:
        shape = (24,) if key in {"net.scale", "net.warp_gate"} or key.endswith("bias") else (24, 2)
        state[key] = torch.arange(torch.tensor(shape).prod()).reshape(shape).float() + offset
    for key in HYDRO_KEYS:
        shape = (5,) if key.endswith("bias") else (5, 2)
        state[key] = torch.arange(torch.tensor(shape).prod()).reshape(shape).float() + offset
    state["net.shared.weight"] = torch.tensor([offset])
    torch.save(
        {
            "state_dict": state,
            "hyper_parameters": {"arch": "flow_pp3"},
            "optimizer_states": [{"state": "unused"}],
            "lr_schedulers": [{"state": "unused"}],
        },
        path,
    )


def test_merge_changes_only_selected_rows_and_hydro(tmp_path) -> None:
    control = tmp_path / "control.ckpt"
    distilled = tmp_path / "distilled.ckpt"
    output = tmp_path / "merged.ckpt"
    _checkpoint(control, 0.0)
    _checkpoint(distilled, 4.0)

    merge_field_heads(control, distilled, output, alpha=0.25)
    merged = torch.load(output, map_location="cpu", weights_only=False)
    control_state = torch.load(control, map_location="cpu", weights_only=False)["state_dict"]

    for key in ROW_KEYS:
        for index in range(24):
            expected = control_state[key][index] + (1.0 if index in FIELD_INDICES else 0.0)
            torch.testing.assert_close(merged["state_dict"][key][index], expected)
    for key in HYDRO_KEYS:
        torch.testing.assert_close(merged["state_dict"][key], control_state[key] + 1.0)
    torch.testing.assert_close(
        merged["state_dict"]["net.shared.weight"],
        control_state["net.shared.weight"],
    )
    assert merged["optimizer_states"] == []
    assert merged["lr_schedulers"] == []


def test_shared_trunk_merge_restores_excluded_head_rows(tmp_path) -> None:
    control = tmp_path / "control.ckpt"
    distilled = tmp_path / "distilled.ckpt"
    output = tmp_path / "merged.ckpt"
    _checkpoint(control, 0.0)
    _checkpoint(distilled, 4.0)

    merge_field_heads(
        control,
        distilled,
        output,
        alpha=0.25,
        include_shared_trunk=True,
    )
    merged = torch.load(output, map_location="cpu", weights_only=False)
    control_state = torch.load(control, map_location="cpu", weights_only=False)["state_dict"]

    for key in ROW_KEYS:
        for index in range(24):
            expected = control_state[key][index] + (1.0 if index in FIELD_INDICES else 0.0)
            torch.testing.assert_close(merged["state_dict"][key][index], expected)
    for key in HYDRO_KEYS:
        torch.testing.assert_close(merged["state_dict"][key], control_state[key] + 1.0)
    torch.testing.assert_close(
        merged["state_dict"]["net.shared.weight"],
        control_state["net.shared.weight"] + 1.0,
    )
    assert merged["checkpoint_field_merge"]["include_shared_trunk"] is True
