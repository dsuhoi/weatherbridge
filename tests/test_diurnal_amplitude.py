from tools.downstream.eval_diurnal_amplitude import month_hour_bounds


def test_month_hour_bounds_keeps_complete_interior_month() -> None:
    start, end, dropped = month_hour_bounds(2020, 11, 8784)

    assert end - start == 30 * 24
    assert dropped == 0


def test_month_hour_bounds_drops_unbounded_final_day() -> None:
    start, end, dropped = month_hour_bounds(2020, 12, 8784)

    assert end - start == 30 * 24
    assert end < 8784
    assert dropped == 24


def test_month_hour_bounds_handles_non_leap_year() -> None:
    start, end, dropped = month_hour_bounds(2021, 12, 8760)

    assert end - start == 30 * 24
    assert end < 8760
    assert dropped == 24
