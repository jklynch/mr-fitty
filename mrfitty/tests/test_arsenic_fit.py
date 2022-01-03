import logging
import tempfile

from sklearn.linear_model import LinearRegression

from mrfitty.base import AdaptiveEnergyRangeBuilder, FixedEnergyRangeBuilder
from mrfitty.combination_fit import AllCombinationFitTask

logging_level = logging.INFO

logging.basicConfig(level=logging_level, filename="test_arsenic_fit.log")
log = logging.getLogger(name=__name__)


"""
These are smoke tests for the fitting code. Configuration code is not tested.
"""


def test_arsenic_1(caplog, arsenic_references, arsenic_unknowns):
    """
    Test fits for known arsenic data and reference_spectra using AdaptiveEnergyRangeBuilder.

    :param caplog: logging fixture
    :param arsenic_references: list of arsenic reference spectra from mr-fitty/src/example/arsenic
    :param arsenic_unknowns: list of arsenic unknown spectra from mr-fitty/src/example/arsenic
    :return:
    """
    caplog.set_level(logging_level)

    task = AllCombinationFitTask(
        ls=LinearRegression,
        energy_range_builder=AdaptiveEnergyRangeBuilder(),
        reference_spectrum_list=arsenic_references,
        unknown_spectrum_list=[arsenic_unknowns[0]],
        best_fits_plot_limit=1,
        component_count_range=range(1, 3 + 1),
    )

    with tempfile.TemporaryDirectory() as plots_pdf_dp:
        task.fit_all(plots_pdf_dp=plots_pdf_dp)

        unknown_spectrum_fit = task.fit_table[arsenic_unknowns[0]]

        assert (
            unknown_spectrum_fit.best_fit.interpolant_incident_energy.shape
            == unknown_spectrum_fit.best_fit.fit_spectrum_b.shape
        )
        assert (
            unknown_spectrum_fit.best_fit.interpolant_incident_energy.shape
            == unknown_spectrum_fit.best_fit.unknown_spectrum_b.shape
        )
        assert (
            unknown_spectrum_fit.best_fit.interpolant_incident_energy.shape
            == unknown_spectrum_fit.best_fit.residuals.shape
        )

        assert 3 == len(unknown_spectrum_fit.best_fit.reference_spectra_seq)


def test_arsenic_2(caplog, arsenic_references, arsenic_unknowns):
    """
    Test fits for a single reference against all reference_spectra using FixedEnergyRangeBuilder.

    :param caplog: logging fixture
    :param arsenic_references: list of arsenic reference spectra from mr-fitty/src/example/arsenic
    :param arsenic_unknowns: list of arsenic unknown spectra from mr-fitty/src/example/arsenic
    :return:
    """
    caplog.set_level(logging_level)

    task = AllCombinationFitTask(
        ls=LinearRegression,
        energy_range_builder=FixedEnergyRangeBuilder(
            energy_start=11850.0, energy_stop=12090.0
        ),
        reference_spectrum_list=arsenic_references,
        unknown_spectrum_list=[arsenic_unknowns[0]],
        best_fits_plot_limit=1,
        component_count_range=range(1, 3 + 1),
    )

    with tempfile.TemporaryDirectory() as plots_pdf_dp:
        task.fit_all(plots_pdf_dp=plots_pdf_dp)

        unknown_spectrum_fit = task.fit_table[arsenic_unknowns[0]]

        best_fit_ref_count = len(unknown_spectrum_fit.best_fit.reference_spectra_seq)
        assert 2 <= best_fit_ref_count <= 3
