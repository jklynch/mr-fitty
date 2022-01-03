import io

import numpy as np
import pytest

from mrfitty import base


class TestSpectrum:
    def test_read_file_with_one_column(self, tmpdir):
        # generate a test file
        a_file = tmpdir.join("test_read_file.e")
        a_file.write(
            """
1000.0
1001.0
1002.0
"""
        )
        a_file_path = str(a_file)
        with pytest.raises(Exception):
            base.Spectrum.read_file(a_file_path)

    def test_read_file_with_zero_comments(self, tmpdir):
        # generate a test file
        a_file = tmpdir.join("test_read_file.e")
        a_file.write(
            """
1000.0\t0.123
1001.0\t0.234
1002.0\t0.345
"""
        )
        a_file_path = str(a_file)
        a_spectrum = base.Spectrum.read_file(a_file_path)

        self.assert_spectrum_correct(a_spectrum)

    def test_read_file_with_one_comment(self, tmpdir):
        # generate a test file
        a_file = tmpdir.join("test_read_file.e")
        a_file.write(
            """
# comments
1000.0\t0.123
1001.0\t0.234
1002.0\t0.345
"""
        )
        a_file_path = str(a_file)
        a_spectrum = base.Spectrum.read_file(a_file_path)

        self.assert_spectrum_correct(a_spectrum)

    def test_read_file_with_two_comments(self, tmpdir):
        # generate a test file
        a_file = tmpdir.join("test_read_file.e")
        a_file.write(
            """
# comments
# more comments
1000.0\t0.123
1001.0\t0.234
1002.0\t0.345
"""
        )
        a_file_path = str(a_file)
        a_spectrum = base.Spectrum.read_file(a_file_path)

        self.assert_spectrum_correct(a_spectrum)

    def test_read_file_with_three_columns(self, tmpdir):
        # generate a test file
        a_file = tmpdir.join("test_read_file.e")
        a_file.write(
            """
# comments
1000.0\t0.123\t0.012
1001.0\t0.234\t0.023
1002.0\t0.345\t0.034
"""
        )
        a_file_path = str(a_file)
        a_spectrum = base.Spectrum.read_file(a_file_path)

        self.assert_spectrum_correct(a_spectrum)

    def assert_spectrum_correct(self, a_spectrum):
        assert a_spectrum.data_df.shape == (3, 1)
        assert len(a_spectrum.header_fields) == 0
        np.testing.assert_approx_equal(a_spectrum.data_df.index[0], 1000.0)
        np.testing.assert_approx_equal(a_spectrum.data_df.index[-1], 1002.0)
        np.testing.assert_approx_equal(a_spectrum.data_df.norm.iloc[0], 0.123)
        np.testing.assert_approx_equal(a_spectrum.data_df.norm.iloc[-1], 0.345)


def test_parse_header():
    header = io.StringIO(
        """
# Valence group: Fe3_silicate
# Athena data file -- Athena version 0.8.056
# Saving Aegirine as normalized mu(E)
# .  Element=Fe   Edge=K
# Background parameters
# .  E0=7123.000  Eshift=0.000  Rbkg=1.000
# .  Standard=0: None
# .  Kweight=2.0  Edge step=0.976
# .  Fixed step=no    Flatten=yes
# .  Pre-edge range: [ -102.998 : -30.000 ]
# .  Pre-edge line: -0.37571 + 5.3729e-005 * E
# .  Normalization range: [ 169.345 : 272.399 ]
# .  Post-edge polynomial: 0.5218 + 6.4716e-005 * E + 0 * E^2
# .  Spline range: [ 0.000 : 275.984 ]   Clamps: None/Strong
# Foreward FT parameters
# .  Kweight=0.5   Window=hanning   Phase correction=no
# .  k-range: [ 2.000 : 6.511 ]   dk=1.00
# Backward FT parameters
# .  R-range: [ 1.000 : 3.000 ]
# .  dR=0.00   Window=hanning
# Plotting parameters
# .  Multiplier=1   Y-offset=0.000
# .
# Name: Aegirine powder average
# Empirical formula: NaFe3+(Si2O6)/ Webmineral
# Mineral group: clinopyroxenes
# Xtal system: monoclinic
# Edge: K
# Sample prep: powder on kapton tape
# Beamline: ALS 10.3.2
# Det:  I0=N2
# Ref/notes:  Mineralogical Research Company
# Temp: Room temp (28 degC)
# Time&date: 9:06 PM 03/14/2007
# Provider: Matthew A. Marcus
# ===========================================================
#------------------------
#  energy norm bkg_norm der_norm
  7010.6674       0.10203199E-02   0.10202176E-02  -0.17394430E-05
  7015.6673       0.10110155E-02   0.10109131E-02  -0.22488155E-04
"""
    )

    a_spectrum = base.Spectrum.read_file(header)

    print(a_spectrum.header_fields)

    assert a_spectrum.header_fields["Valence group"] == "Fe3_silicate"
    assert a_spectrum.header_fields["Time&date"] == "9:06 PM 03/14/2007"
    assert a_spectrum.data_df.shape == (2, 1)
