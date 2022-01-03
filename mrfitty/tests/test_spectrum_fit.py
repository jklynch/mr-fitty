from sklearn.linear_model import LinearRegression

from mrfitty.base import AdaptiveEnergyRangeBuilder
from mrfitty.combination_fit import AllCombinationFitTask


def generate_spectrum_fit(reference_count, reference_spectra, unknown_spectrum):
    """

    Parameters
    ----------
    reference_count: (int) count of reference spectra used in SpectrumFit

    Returns
    -------
    SpectrumFit instance
    """

    energy_range_builder = AdaptiveEnergyRangeBuilder()

    fitter = AllCombinationFitTask(
        ls=LinearRegression,
        reference_spectrum_list=reference_spectra,
        unknown_spectrum_list=(unknown_spectrum,),
        energy_range_builder=energy_range_builder,
        best_fits_plot_limit=0,
        component_count_range=(reference_count,),
    )

    spectrum_fit, _ = fitter.fit(unknown_spectrum=unknown_spectrum)

    return spectrum_fit
