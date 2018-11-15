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
import pandas as pd

import mrfitty.base as base


def generate_spectrum_fit(reference_count):
    """

    Parameters
    ----------
    reference_count: (int) count of reference spectra used in SpectrumFit

    Returns
    -------
    SpectrumFit instance
    """
    spectrum_fit = base.SpectrumFit(
        interpolant_incident_energy=None,
        reference_spectra_A_df=pd.DataFrame(),
        unknown_spectrum_b=pd.DataFrame(columns=('energy', 'norm')),
        reference_spectra_seq=(),
        reference_spectra_coef_x=pd.Series())


    return spectrum_fit
