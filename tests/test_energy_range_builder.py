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
import io

from mrfitty.base import AdaptiveEnergyRangeBuilder, FixedEnergyRangeBuilder
from mrfitty.base import Spectrum

unknown_1_text = """\
11760.0\t0.08123
11765.0\t0.08234
11771.0\t0.08345
11776.0\t0.08456
"""

reference_1_text = """\
11761.0\t0.08123
11764.0\t0.08234
11770.0\t0.08345
11775.0\t0.08456
"""

reference_2_text = """\
11759.0\t0.08123
11766.0\t0.08234
11772.0\t0.08345
11778.0\t0.08456
"""


def test_adaptive_energy_range_builder():
    unknown_1 = Spectrum.read_file(io.StringIO(unknown_1_text))
    reference_1 = Spectrum.read_file(io.StringIO(reference_1_text))
    reference_2 = Spectrum.read_file(io.StringIO(reference_2_text))

    fit_energies, fit_energy_indices = AdaptiveEnergyRangeBuilder().build_range(
        unknown_spectrum=unknown_1, reference_spectrum_seq=[reference_1, reference_2]
    )

    assert fit_energies.shape == (2,)
    assert fit_energies[0] == 11765.0
    assert fit_energies[-1] == 11771.0
    assert fit_energy_indices.tolist() == [False, True, True, False]


def test_fixed_energy_range_builder():
    unknown_1 = Spectrum.read_file(io.StringIO(unknown_1_text))
    reference_1 = Spectrum.read_file(io.StringIO(reference_1_text))
    reference_2 = Spectrum.read_file(io.StringIO(reference_2_text))

    fit_energies, fit_energy_indices = FixedEnergyRangeBuilder(
        energy_start=11761.0, energy_stop=11771.0
    ).build_range(
        unknown_spectrum=unknown_1, reference_spectrum_seq=[reference_1, reference_2]
    )

    assert fit_energies.shape == (2,)
    assert fit_energies[0] == 11765.0
    assert fit_energies[-1] == 11771.0
    assert fit_energy_indices.tolist() == [False, True, True, False]
