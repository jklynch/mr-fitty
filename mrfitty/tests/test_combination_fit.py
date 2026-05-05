import os
import tempfile

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

from mrfitty.base import AdaptiveEnergyRangeBuilder, Spectrum
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


def _make_task_and_fit(ls, synthetic_spectra):
    reference_spectra, unknown = synthetic_spectra
    task = AllCombinationFitTask(
        ls=ls,
        reference_spectrum_list=reference_spectra,
        unknown_spectrum_list=[unknown],
        energy_range_builder=AdaptiveEnergyRangeBuilder(),
        best_fits_plot_limit=0,
        component_count_range=(2,),
    )
    spectrum_fit, _ = task.fit(unknown_spectrum=unknown)
    return task, spectrum_fit, unknown


def test_build_reference_to_reference_label_keys_are_reference_names(synthetic_spectra):
    task, fit, unknown = _make_task_and_fit(
        NonNegativeLinearRegression, synthetic_spectra
    )
    result = task.build_reference_to_reference_label(
        spectrum=unknown, any_given_fit=fit
    )
    assert set(result.keys()) == {"ref_a.e", "ref_b.e"}


def test_build_reference_to_reference_label_keys_ordered_by_descending_contribution(
    synthetic_spectra,
):
    task, fit, unknown = _make_task_and_fit(
        NonNegativeLinearRegression, synthetic_spectra
    )
    result = task.build_reference_to_reference_label(
        spectrum=unknown, any_given_fit=fit
    )
    keys = list(result.keys())
    assert keys[0] == "ref_a.e"
    assert keys[1] == "ref_b.e"


def test_build_reference_to_reference_label_label_format_without_std_err(
    synthetic_spectra,
):
    task, fit, unknown = _make_task_and_fit(
        NonNegativeLinearRegression, synthetic_spectra
    )
    result = task.build_reference_to_reference_label(
        spectrum=unknown, any_given_fit=fit
    )
    for ref_name, label in result.items():
        assert ref_name in label
        assert "±" not in label
        assert " (" in label


def test_build_reference_to_reference_label_label_format_with_std_err(
    synthetic_spectra,
):
    task, fit, unknown = _make_task_and_fit(OlsWithStats, synthetic_spectra)
    result = task.build_reference_to_reference_label(
        spectrum=unknown, any_given_fit=fit
    )
    for ref_name, label in result.items():
        assert ref_name in label
        assert "±" in label


def test_build_reference_to_reference_label_include_ref_only_contribution_true(
    synthetic_spectra,
):
    task, fit, unknown = _make_task_and_fit(
        NonNegativeLinearRegression, synthetic_spectra
    )
    result = task.build_reference_to_reference_label(
        spectrum=unknown, any_given_fit=fit, include_ref_only_contribution=True
    )
    ref_only = fit.get_reference_only_contributions_sr()
    for name, label in result.items():
        assert f"({ref_only[name]:5.2f})" in label


def test_build_reference_to_reference_label_include_ref_only_contribution_false(
    synthetic_spectra,
):
    task, fit, unknown = _make_task_and_fit(
        NonNegativeLinearRegression, synthetic_spectra
    )
    result = task.build_reference_to_reference_label(
        spectrum=unknown, any_given_fit=fit, include_ref_only_contribution=False
    )
    ref_only = fit.get_reference_only_contributions_sr()
    for name, label in result.items():
        assert f"({ref_only[name]:5.2f})" not in label


def test_build_reference_to_reference_label_padding_uses_long_spectrum_file_name(
    synthetic_spectra,
):
    reference_spectra, _ = synthetic_spectra
    long_name = "a" * 40 + ".e"
    energies = np.linspace(11800, 11900, 50)
    unknown_df = pd.DataFrame(
        {"norm": np.sin(energies / 100)}, index=pd.Index(energies, name="energy")
    )
    unknown = Spectrum(long_name, unknown_df, {})
    task = AllCombinationFitTask(
        ls=NonNegativeLinearRegression,
        reference_spectrum_list=reference_spectra,
        unknown_spectrum_list=[unknown],
        energy_range_builder=AdaptiveEnergyRangeBuilder(),
        best_fits_plot_limit=0,
        component_count_range=(2,),
    )
    fit, _ = task.fit(unknown_spectrum=unknown)
    result = task.build_reference_to_reference_label(
        spectrum=unknown, any_given_fit=fit
    )
    expected_pad = len(long_name) + 4
    for ref_name, label in result.items():
        assert label.startswith(ref_name.ljust(expected_pad))
