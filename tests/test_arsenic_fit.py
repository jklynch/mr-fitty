"""
The MIT License (MIT)

Copyright (c) 2015-2018 Joshua Lynch, Sarah Nicholas

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
import logging
import tempfile

from sklearn.linear_model import LinearRegression

from mrfitty.base import AdaptiveEnergyRangeBuilder, FixedEnergyRangeBuilder
from mrfitty.combination_fit import AllCombinationFitTask

logging.basicConfig(level=logging.DEBUG, filename='test_arsenic_fit.log')
log = logging.getLogger(name=__name__)


"""
These are smoke tests for the fitting code. Configuration code is not tested.

pytest-catchlog is required by these tests
"""


def test_arsenic_1(caplog, arsenic_references, arsenic_unknowns):
    """
    Test fits for known arsenic data and reference_spectra using AdaptiveEnergyRangeBuilder.

    :param caplog: logging fixture
    :param arsenic_references: list of arsenic reference spectra from mr-fitty/src/example/arsenic
    :param arsenic_unknowns: list of arsenic unknown spectra from mr-fitty/src/example/arsenic
    :return:
    """
    caplog.set_level(logging.INFO)

    task = AllCombinationFitTask(
        ls=LinearRegression,
        energy_range_builder=AdaptiveEnergyRangeBuilder(),
        reference_spectrum_list=arsenic_references,
        unknown_spectrum_list=[arsenic_unknowns[0], ],
        best_fits_plot_limit=1,
        component_count_range=range(1, 3+1)
    )

    with tempfile.TemporaryDirectory() as plots_pdf_dp:
        task.fit_all(plots_pdf_dp=plots_pdf_dp)

        unknown_spectrum_fit = task.fit_table[arsenic_unknowns[0]]

        assert unknown_spectrum_fit.best_fit.interpolant_incident_energy.shape == unknown_spectrum_fit.best_fit.fit_spectrum_b.shape
        assert unknown_spectrum_fit.best_fit.interpolant_incident_energy.shape == unknown_spectrum_fit.best_fit.unknown_spectrum_b.shape
        assert unknown_spectrum_fit.best_fit.interpolant_incident_energy.shape == unknown_spectrum_fit.best_fit.residuals.shape

        assert 3 == len(unknown_spectrum_fit.best_fit.reference_spectra_seq)


def test_arsenic_2(caplog, arsenic_references, arsenic_unknowns):
    """
    Test fits for a single reference against all reference_spectra using FixedEnergyRangeBuilder.

    :param caplog: logging fixture
    :param arsenic_references: list of arsenic reference spectra from mr-fitty/src/example/arsenic
    :param arsenic_unknowns: list of arsenic unknown spectra from mr-fitty/src/example/arsenic
    :return:
    """
    caplog.set_level(logging.INFO)

    task = AllCombinationFitTask(
        ls=LinearRegression,
        energy_range_builder=FixedEnergyRangeBuilder(energy_start=11850.0, energy_stop=12090.0),
        reference_spectrum_list=arsenic_references,
        unknown_spectrum_list=[arsenic_unknowns[0], ],
        best_fits_plot_limit=1,
        component_count_range=range(1, 3+1)
    )

    with tempfile.TemporaryDirectory() as plots_pdf_dp:
        task.fit_all(plots_pdf_dp=plots_pdf_dp)

        unknown_spectrum_fit = task.fit_table[arsenic_unknowns[0]]

        best_fit_ref_count = len(unknown_spectrum_fit.best_fit.reference_spectra_seq)
        assert 2 <= best_fit_ref_count <= 3
