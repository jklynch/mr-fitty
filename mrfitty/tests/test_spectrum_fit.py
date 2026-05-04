import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from mrfitty.base import AdaptiveEnergyRangeBuilder, ReferenceSpectrum, Spectrum
from mrfitty.combination_fit import AllCombinationFitTask
from mrfitty.linear_model import NonNegativeLinearRegression, OlsWithStats


def _make_synthetic_spectra():
    energies = np.linspace(11800, 11900, 50)
    ref_a_norm = np.sin(energies / 100)
    ref_b_norm = np.cos(energies / 100)

    ref_a_df = pd.DataFrame(
        {"norm": ref_a_norm}, index=pd.Index(energies, name="energy")
    )
    ref_b_df = pd.DataFrame(
        {"norm": ref_b_norm}, index=pd.Index(energies, name="energy")
    )
    ref_a = ReferenceSpectrum("ref_a.e", ref_a_df, {})
    ref_b = ReferenceSpectrum("ref_b.e", ref_b_df, {})

    coef = np.array([0.6, 0.4])
    unknown_norm = (
        coef[0] * ref_a_norm
        + coef[1] * ref_b_norm
        + np.random.default_rng(0).normal(0, 0.001, 50)
    )
    unknown_df = pd.DataFrame(
        {"norm": unknown_norm}, index=pd.Index(energies, name="energy")
    )
    unknown = Spectrum("unknown.e", unknown_df, {})

    return [ref_a, ref_b], unknown


def _fit_spectrum(ls, reference_spectra, unknown):
    fitter = AllCombinationFitTask(
        ls=ls,
        reference_spectrum_list=reference_spectra,
        unknown_spectrum_list=(unknown,),
        energy_range_builder=AdaptiveEnergyRangeBuilder(),
        best_fits_plot_limit=0,
        component_count_range=(2,),
    )
    spectrum_fit, _ = fitter.fit(unknown_spectrum=unknown)
    return spectrum_fit


def test_get_reference_std_err_percent_sr_returns_none_when_no_std_err():
    reference_spectra, unknown = _make_synthetic_spectra()
    sf = _fit_spectrum(NonNegativeLinearRegression, reference_spectra, unknown)
    assert sf is not None
    assert sf.get_reference_std_err_percent_sr() is None
    coef_sr = pd.Series(
        sf.reference_spectra_coef_x, index=sf.reference_spectra_A_df.columns
    )
    np.testing.assert_allclose(coef_sr["ref_a.e"], 0.6, atol=0.05)
    np.testing.assert_allclose(coef_sr["ref_b.e"], 0.4, atol=0.05)


def test_get_reference_std_err_percent_sr_returns_series():
    reference_spectra, unknown = _make_synthetic_spectra()
    sf = _fit_spectrum(OlsWithStats, reference_spectra, unknown)
    assert sf is not None
    coef_sr = pd.Series(
        sf.reference_spectra_coef_x, index=sf.reference_spectra_A_df.columns
    )
    np.testing.assert_allclose(coef_sr["ref_a.e"], 0.6, atol=0.05)
    np.testing.assert_allclose(coef_sr["ref_b.e"], 0.4, atol=0.05)
    result = sf.get_reference_std_err_percent_sr()

    assert result is not None
    assert isinstance(result, pd.Series)
    assert len(result) == 2
    assert list(result.index) == list(sf.reference_spectra_A_df.columns)
    assert all(result >= 0)


def test_linear_regression_fit_coefficients():
    reference_spectra, unknown = _make_synthetic_spectra()
    sf = _fit_spectrum(LinearRegression, reference_spectra, unknown)
    assert sf is not None
    assert sf.get_reference_std_err_percent_sr() is None
    coef_sr = pd.Series(
        sf.reference_spectra_coef_x, index=sf.reference_spectra_A_df.columns
    )
    np.testing.assert_allclose(coef_sr["ref_a.e"], 0.6, atol=0.05)
    np.testing.assert_allclose(coef_sr["ref_b.e"], 0.4, atol=0.05)
