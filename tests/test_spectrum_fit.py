"""
The MIT License (MIT)

Copyright (c) 2015-2019 Joshua Lynch, Sarah Nicholas

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
