from __future__ import annotations

import numpy as np
from pathlib import Path

from tools.eval.batch_eval_12h_memmap_fast import (
    batch4_reference_order,
    grouped_channel_sums,
)


def test_grouped_channel_sums_matches_scalar_accumulation() -> None:
    rng = np.random.default_rng(2027)
    values = rng.normal(size=(11, 24)).astype(np.float32) ** 2
    taus = np.asarray([1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1], dtype=np.int16)

    actual = grouped_channel_sums(values, taus, set(range(1, 6)))
    expected = {tau: np.zeros(24, dtype=np.float64) for tau in range(1, 6)}
    for sample, tau in zip(values, taus, strict=True):
        expected[int(tau)] += sample

    assert actual.keys() == expected.keys()
    for tau in expected:
        np.testing.assert_allclose(actual[tau], expected[tau], rtol=0, atol=0)


def test_fast_runner_exposes_only_paper_methods() -> None:
    from tools.eval.batch_eval_12h_memmap_fast import FastBatchModelRunner12h

    runner = FastBatchModelRunner12h.__new__(FastBatchModelRunner12h)
    assert runner.method_names() == ["model", "bilinear"]


def test_batch4_reference_order_restores_paired_index() -> None:
    anchors = [(2020, start) for start in range(8)]
    taus = (1, 2, 3)
    batch8 = [
        (year, start, tau)
        for tau in taus
        for year, start in anchors
    ]
    expected = [
        (year, start, tau)
        for offset in range(0, len(anchors), 4)
        for tau in taus
        for year, start in anchors[offset : offset + 4]
    ]
    years = np.asarray([row[0] for row in batch8])
    starts = np.asarray([row[1] for row in batch8])
    hours = np.asarray([row[2] for row in batch8])

    order = batch4_reference_order(years, starts, hours)
    actual = [batch8[int(index)] for index in order]

    assert actual == expected


def test_baseline_queue_uses_only_required_horizon_years() -> None:
    script = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "run_journal_baseline_geometry_v1_cloudru.sh"
    ).read_text()

    assert 'YEARS_6="${YEARS_6:-2020 2021}"' in script
    assert 'YEARS_12="${YEARS_12:-2020}"' in script
    assert "--batch-size 8 --num-workers 0" in script
    assert '"years": {"6h": [2020, 2021], "12h": [2020]}' in script
    assert "batch_eval_12h_memmap_fast.py" in script
