import hashlib
import json

import pytest

from tools.eval.summarize_spectral_dominance import summarize


def _artifact(path, *, failure=False):
    per_channel = {}
    for index in range(24):
        energy_delta = 0.01 if failure and index == 3 else -0.1
        per_channel[f"field_{index}"] = {
            "energy_log_error": {
                "delta_left_minus_right": energy_delta,
                "p_paired_block_permutation": 1.0e-5,
                "p_holm": 2.4e-4,
            },
            "shape_log_error": {
                "delta_left_minus_right": -0.1,
                "p_paired_block_permutation": 1.0e-5,
                "p_holm": 2.4e-4,
            },
            "coherence": {
                "delta_left_minus_right": 0.1,
                "p_paired_block_permutation": 1.0e-5,
                "p_holm": 2.4e-4,
            },
            "signed_cospectrum": {
                "delta_left_minus_right": 0.1,
                "p_paired_block_permutation": 1.0e-5,
                "p_holm": 2.4e-4,
            },
        }
    path.parent.mkdir(parents=True, exist_ok=True)
    left = path.with_suffix(".left.npz")
    right = path.with_suffix(".right.npz")
    left.write_bytes(b"left")
    right.write_bytes(b"right")
    path.write_text(
        json.dumps(
            {
                "left_path": str(left),
                "left_sha256": hashlib.sha256(left.read_bytes()).hexdigest(),
                "right_paths": {"dcae": str(right)},
                "right_sha256": {
                    "dcae": hashlib.sha256(right.read_bytes()).hexdigest()
                },
                "comparisons": {
                    "dcae": {"per_channel": per_channel},
                }
            }
        )
    )


def test_summarizes_field_hour_failures(tmp_path) -> None:
    for tau in (1, 2):
        _artifact(
            tmp_path / f"2020/spectra/paired_tau{tau}.json",
            failure=tau == 2,
        )

    result = summarize(
        tmp_path,
        years=[2020],
        taus=[1, 2],
        pattern="{year}/spectra/paired_tau{tau}.json",
        reference="dcae",
    )

    energy = result["yearly"]["2020"]["energy_log_error"]
    assert energy["wins"] == 47
    assert energy["cell_count"] == 48
    assert len(energy["failures"]) == 1
    failure = energy["failures"][0]
    assert failure["tau"] == 2
    assert failure["channel"] == "field_3"
    assert failure["delta_left_minus_right"] == 0.01
    assert failure["p_raw"] == 1.0e-5
    assert failure["p_holm_within_tau"] == 2.4e-4
    assert failure["p_holm_global_tau_channel"] == pytest.approx(4.8e-4)
    assert not result["strict_pointwise_spectral_dominance"]
