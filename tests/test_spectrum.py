"""

"""
import pytest

import numpy as np

from mrfitty import base


class TestSpectrum(object):
    def test_read_file_with_one_column(self, tmpdir):
        # generate a test file
        a_file = tmpdir.join('test_read_file.e')
        a_file.write(
'''
1000.0
1001.0
1002.0
'''
        )
        a_file_path = str(a_file)
        with pytest.raises(Exception):
            a_spectrum = base.Spectrum.read_file(a_file_path)

    def test_read_file_with_zero_comments(self, tmpdir):
        # generate a test file
        a_file = tmpdir.join('test_read_file.e')
        a_file.write(
'''
1000.0\t0.123
1001.0\t0.234
1002.0\t0.345
'''
        )
        a_file_path = str(a_file)
        a_spectrum = base.Spectrum.read_file(a_file_path)

        self.assert_spectrum_correct(a_spectrum)

    def test_read_file_with_one_comment(self, tmpdir):
        # generate a test file
        a_file = tmpdir.join('test_read_file.e')
        a_file.write(
'''
# comments
1000.0\t0.123
1001.0\t0.234
1002.0\t0.345
'''
        )
        a_file_path = str(a_file)
        a_spectrum = base.Spectrum.read_file(a_file_path)

        self.assert_spectrum_correct(a_spectrum)

    def test_read_file_with_two_comments(self, tmpdir):
        # generate a test file
        a_file = tmpdir.join('test_read_file.e')
        a_file.write(
'''
# comments
# more comments
1000.0\t0.123
1001.0\t0.234
1002.0\t0.345
'''
        )
        a_file_path = str(a_file)
        a_spectrum = base.Spectrum.read_file(a_file_path)

        self.assert_spectrum_correct(a_spectrum)

    def test_read_file_with_three_columns(self, tmpdir):
        # generate a test file
        a_file = tmpdir.join('test_read_file.e')
        a_file.write(
'''
# comments
1000.0\t0.123\t0.012
1001.0\t0.234\t0.023
1002.0\t0.345\t0.034
'''
        )
        a_file_path = str(a_file)
        a_spectrum = base.Spectrum.read_file(a_file_path)

        self.assert_spectrum_correct(a_spectrum)

    def assert_spectrum_correct(self, a_spectrum):
        assert a_spectrum.data_df.shape == (3,2)
        np.testing.assert_approx_equal(a_spectrum.data_df.energy.iloc[0], 1000.0)
        np.testing.assert_approx_equal(a_spectrum.data_df.energy.iloc[-1], 1002.0)
        np.testing.assert_approx_equal(a_spectrum.data_df.norm.iloc[0], 0.123)
        np.testing.assert_approx_equal(a_spectrum.data_df.norm.iloc[-1], 0.345)
