import csv
from pathlib import Path

from tools.eval.export_selector_diagnostics_tex import (
    render_selector_table,
    render_vector_table,
)


def test_selector_tables_report_previously_hidden_gates() -> None:
    path = Path("paper/supplementary_data_1_statistics.csv")
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))

    selector = render_selector_table(rows, "digest")
    vector = render_vector_table(rows, "digest")

    assert selector.count("\\,h, 20") == 4
    assert "lower-tropospheric moisture bias" in selector
    assert "wins/losses" not in selector
    assert all(line.count(" & ") == 3 for line in selector.splitlines() if " & " in line)
    assert vector.count("Spheroidal") == 4
    assert vector.count("Toroidal") == 4
    assert "complete multiplicity-corrected vector family" in vector
