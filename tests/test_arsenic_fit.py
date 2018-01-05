"""
pytest-catchlog is required by this test
"""
from glob import glob
import logging
import os
import tempfile

from sklearn.linear_model import LinearRegression

from mrfitty.base import AdaptiveEnergyRangeBuilder, FixedEnergyRangeBuilder, ReferenceSpectrum, Spectrum
from mrfitty.combination_fit import AllCombinationFitTask

logging.basicConfig(level=logging.DEBUG, filename='test_arsenic_fit.log')
log = logging.getLogger(name=__name__)


def test_arsenic_1(caplog, request):
    """
    Test fits for known arsenic data and reference_spectra.
    Expect to find PRM, data, and reference files in a directory called 'test_arsenic_fit'.
    See also: http://stackoverflow.com/questions/29627341/pytest-where-to-store-expected-data.

    :param request: pytest fixture with information about the path to this test file
    :return:
    """

    caplog.set_level(logging.INFO)

    test_arsenic_fit_fp = request.module.__file__
    log.info('test_arsenic_fit_fp: {}'.format(test_arsenic_fit_fp))
    test_arsenic_fit_dir_path, _ = os.path.splitext(test_arsenic_fit_fp)

    #reference_file_path_pattern = os.path.join(test_arsenic_fit_dir_path, 'reference', 'arsenate_*.e')
    reference_file_path_pattern = os.path.join(test_arsenic_fit_dir_path, 'reference', '*.e')
    data_file_path = os.path.join(test_arsenic_fit_dir_path, 'data', 'OTT3_55_spot0.e')

    reference_spectrum_list = [
        ReferenceSpectrum.read_file(file_path)
        for file_path
        in glob(reference_file_path_pattern)
    ]
    log.info(reference_spectrum_list)

    unknown_spectrum = Spectrum.read_file(data_file_path)
    log.info(unknown_spectrum)

    task = AllCombinationFitTask(
        ls=LinearRegression,
        energy_range_builder=AdaptiveEnergyRangeBuilder(),
        reference_spectrum_list=reference_spectrum_list,
        unknown_spectrum_list=[unknown_spectrum, ],
        best_fits_plot_limit=1,
        component_count_range=range(1, 3+1)
    )

    with tempfile.TemporaryDirectory() as plots_pdf_dp:
        task.fit_all(plots_pdf_dp=plots_pdf_dp)

        unknown_spectrum_fit = task.fit_table[unknown_spectrum]

        assert unknown_spectrum_fit.best_fit.interpolant_incident_energy.shape == unknown_spectrum_fit.best_fit.fit_spectrum_b.shape
        assert unknown_spectrum_fit.best_fit.interpolant_incident_energy.shape == unknown_spectrum_fit.best_fit.unknown_spectrum_b.shape
        assert unknown_spectrum_fit.best_fit.interpolant_incident_energy.shape == unknown_spectrum_fit.best_fit.residuals.shape

        assert 3 == len(unknown_spectrum_fit.best_fit.reference_spectra_seq)


def test_arsenic_2(caplog, request):
    """
    Test fits for a single reference against all reference_spectra..
    Expect to find PRM, data, and reference files in a directory called 'test_arsenic_fit'.
    See also: http://stackoverflow.com/questions/29627341/pytest-where-to-store-expected-data.

    This test is not reliable. Fix it.

    :param request: pytest fixture with information about the path to this test file
    :return:
    """

    caplog.set_level(logging.INFO)

    test_arsenic_fit_fp = request.module.__file__
    log.info('test_arsenic_fit_fp: {}'.format(test_arsenic_fit_fp))
    test_arsenic_fit_dir_path, _ = os.path.splitext(test_arsenic_fit_fp)

    #reference_file_path_pattern = os.path.join(test_arsenic_fit_dir_path, 'reference', 'arsenate_*.e')
    reference_file_path_pattern = os.path.join(test_arsenic_fit_dir_path, 'reference', '*.e')
    #data_file_path = os.path.join(test_arsenic_fit_dir_path, 'reference', 'arsenate_aqueous_avg_als_cal.e')

    reference_spectrum_list = [
        ReferenceSpectrum.read_file(file_path)
        for file_path
        in glob(reference_file_path_pattern)
    ]
    log.info(reference_spectrum_list)

    unknown_spectrum = reference_spectrum_list[0]
    log.info(unknown_spectrum)

    task = AllCombinationFitTask(
        ls=LinearRegression,
        energy_range_builder=FixedEnergyRangeBuilder(energy_start=11850.0, energy_stop=12090.0),
        reference_spectrum_list=reference_spectrum_list,
        unknown_spectrum_list=[unknown_spectrum],
        best_fits_plot_limit=1,
        component_count_range=range(1, 3+1)
    )

    with tempfile.TemporaryDirectory() as plots_pdf_dp:
        task.fit_all(plots_pdf_dp=plots_pdf_dp)

        unknown_spectrum_fit = task.fit_table[unknown_spectrum]

        best_fit_ref_count = len(unknown_spectrum_fit.best_fit.reference_spectra_seq)
        assert 2 <= best_fit_ref_count <= 3
