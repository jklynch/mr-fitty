"""
Run tests from venv34.  Also run tox from venv34.
Set PYTHONPATH to the src directory and run py.test tests/ --cov=src/mrfitty.
"""
from io import StringIO
import os.path


from click.testing import CliRunner

from mrfitty.combination_fit import ReferenceSpectrum
from mrfitty.combination_fit import AllCombinationFitTask


class TestAllCombinationFitTask(object):
    def test_fit_with_fixed_energy_range(self, tmpdir):
        pass

    def test_fit(self):
        ReferenceSpectrum.read_file(
            ''
        )