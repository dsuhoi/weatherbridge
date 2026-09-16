from __future__ import annotations

import json

from tools.eval.build_direct_distillation_route import build_route


def _evaluation(path, values: dict[str, float]) -> None:
    path.write_text(
        json.dumps(
            {
                "channel_names": ["a", "b"],
                "per_tau": {
                    "1": {
                        "model": {
                            f"rmse_norm_{name}": value
                            for name, value in values.items()
                        }
                    }
                },
            }
        )
    )


def test_route_keeps_only_teacher_improvements_above_margin(tmp_path) -> None:
    control = tmp_path / "control.json"
    teacher = tmp_path / "teacher.json"
    _evaluation(control, {"a": 1.0, "b": 2.0})
    _evaluation(teacher, {"a": 0.98, "b": 1.995})

    payload = build_route(
        control,
        teacher,
        minimum_improvement_pct=0.5,
    )

    assert payload["teacher_route"] == [[True, False]]
    assert payload["active_routes"] == 1


def test_field_route_uses_aggregate_skill_across_tau(tmp_path) -> None:
    control = tmp_path / "control.json"
    teacher = tmp_path / "teacher.json"
    control.write_text(
        json.dumps(
            {
                "channel_names": ["a", "b"],
                "per_tau": {
                    "1": {"model": {"rmse_norm_a": 1.0, "rmse_norm_b": 1.0}},
                    "2": {"model": {"rmse_norm_a": 2.0, "rmse_norm_b": 1.0}},
                },
            }
        )
    )
    teacher.write_text(
        json.dumps(
            {
                "channel_names": ["a", "b"],
                "per_tau": {
                    "1": {"model": {"rmse_norm_a": 1.2, "rmse_norm_b": 0.9}},
                    "2": {"model": {"rmse_norm_a": 1.0, "rmse_norm_b": 0.9}},
                },
            }
        )
    )

    payload = build_route(
        control,
        teacher,
        minimum_improvement_pct=0.5,
        routing_granularity="field",
    )

    assert payload["teacher_route"] == [[True, True], [True, True]]
    assert payload["routing_granularity"] == "field"
