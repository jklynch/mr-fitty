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
        unknown_spectrum=unknown_1,
        reference_spectrum_seq=[reference_1, reference_2])

    assert fit_energies.shape == (2,)
    assert fit_energies.iloc[0] == 11765.0
    assert fit_energies.iloc[-1] == 11771.0
    assert fit_energy_indices.tolist() == [False, True, True, False]


def test_fixed_energy_range_builder():
    unknown_1 = Spectrum.read_file(io.StringIO(unknown_1_text))
    reference_1 = Spectrum.read_file(io.StringIO(reference_1_text))
    reference_2 = Spectrum.read_file(io.StringIO(reference_2_text))

    fit_energies, fit_energy_indices = FixedEnergyRangeBuilder(energy_start=11761.0, energy_stop=11771.0).build_range(
        unknown_spectrum=unknown_1,
        reference_spectrum_seq=[reference_1, reference_2])

    assert fit_energies.shape == (2,)
    assert fit_energies.iloc[0] == 11765.0
    assert fit_energies.iloc[-1] == 11771.0
    assert fit_energy_indices.tolist() == [False, True, True, False]
