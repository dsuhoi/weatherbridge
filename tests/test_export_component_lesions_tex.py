from tools.eval.export_component_lesions_tex import render_table


def _result(
    left: float,
    right: float,
    better: int,
    worse: int,
    *,
    include_hydrostatic_fields: bool = False,
) -> dict:
    result = {
        "left_rmse": left,
        "right_rmse": right,
        "delta_left_minus_right": left - right,
        "relative_delta_pct": 100.0 * (left / right - 1.0),
        "cellwise_family": {
            "n_significant_left_better_holm": better,
            "n_significant_left_worse_holm": worse,
        },
    }
    if include_hydrostatic_fields:
        result["per_channel_tau"] = {
            str(hour): {
                field: {
                    "left_rmse": left + 0.001 * hour,
                    "right_rmse": right + 0.001 * hour,
                    "p_left_worse_holm": 0.01,
                }
                for field in ("Z1000", "Z925", "Z850", "Z700", "mslp")
            }
            for hour in range(1, 6)
        }
    return result


def test_render_component_lesions_distinguishes_inference_from_training() -> None:
    table = render_table(
        _result(0.064, 0.063, 5, 115),
        _result(0.101, 0.063, 0, 120),
        _result(0.0635, 0.063, 12, 20, include_hydrostatic_fields=True),
    )

    assert "Same-checkpoint inference lesions" in table
    assert "No arm is retrained" in table
    assert "Acceleration disabled" in table
    assert "Warped-anchor transport disabled" in table
    assert "Hydrostatic head disabled" in table
    assert "Hydrostatic-head target fields" in table
    assert r"$Z_{1000}$" in table
    assert "MSLP" in table
    assert "[" in table
    assert "5/5" not in table
    assert "Significant worse hours" not in table
    assert "better/worse" not in table
    assert all(line.count(" & ") == 3 for line in table.splitlines() if " & " in line)
    assert "1/5001" in table
