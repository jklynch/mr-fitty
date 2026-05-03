import os
import tempfile

import pytest
from sklearn.linear_model import LinearRegression

from mrfitty.base import AdaptiveEnergyRangeBuilder
from mrfitty.combination_fit import AllCombinationFitTask
from mrfitty.linear_model import NonNegativeLinearRegression, OlsWithStats


def test_fit_with_fixed_energy_range(tmpdir):
    pass


def test_fit():
    pass


def _make_task(ls, arsenic_references, arsenic_unknowns):
    unknown = next(s for s in arsenic_unknowns if s.file_name == "OTT3_55_spot0.e")
    task = AllCombinationFitTask(
        ls=ls,
        energy_range_builder=AdaptiveEnergyRangeBuilder(),
        reference_spectrum_list=arsenic_references,
        unknown_spectrum_list=[unknown],
        best_fits_plot_limit=0,
        component_count_range=range(1, 3 + 1),
    )
    with tempfile.TemporaryDirectory() as plots_pdf_dp:
        task.fit_all(plots_pdf_dp=plots_pdf_dp)
    return task, unknown


def test_write_table_without_std_err(tmp_path, arsenic_references, arsenic_unknowns):
    task, unknown = _make_task(
        NonNegativeLinearRegression, arsenic_references, arsenic_unknowns
    )
    table_path = str(tmp_path / "out" / "table.tsv")
    task.write_table(table_path)

    with open(table_path) as f:
        content = f.read()

    assert "std err" not in content
    header_line = content.splitlines()[0]
    assert "percent 1" in header_line
    assert "percent 2" in header_line


def test_write_table_with_std_err(tmp_path, arsenic_references, arsenic_unknowns):
    task, unknown = _make_task(OlsWithStats, arsenic_references, arsenic_unknowns)
    table_path = str(tmp_path / "out" / "table.tsv")
    task.write_table(table_path)

    with open(table_path) as f:
        lines = f.readlines()

    header = lines[0]
    assert "std err 1" in header
    assert "std err 2" in header

    data_line = lines[1]
    cols = data_line.strip().split("\t")
    # spectrum, NSS, residual %, ref1, pct1, stderr1, [ref2, pct2, stderr2, ...]
    assert len(cols) >= 6
    # std err value should be a parseable float
    float(cols[5])
