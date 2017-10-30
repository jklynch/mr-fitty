"""
Run tests from venv34.  Also run tox from venv34.
Set PYTHONPATH to the src directory and run py.test tests/ --cov=src/mrfitty.

export PYTHONPATH=./src/
source venv343/bin/activate
py.test tests/ --cov=src/mrfitty

"""
import logging
import os.path

import pytest

from mrfitty.__main__ import cli
from mrfitty.fit_task_builder import get_config_parser

logging.basicConfig(level=logging.DEBUG)
log = logging.getLogger(name=__name__)


@pytest.mark.skip()
def test_main(fs):

    # write the necessary input files:
    #   configuration file
    #   reference files
    #   sample files
    #   prm file
    # then check for the expected output files
    #   plot pdf
    #   table txt
    #   sample fit files

    fs.CreateFile('ref_1.e', '''\
1000.0\t0.01
1000.1\t0.02
1000.2\t0.03
1000.3\t0.04
1000.4\t0.05
''')

    fs.CreateFile('ref_2.e', '''\
1000.0\t0.05
1000.1\t0.04
1000.2\t0.03
1000.3\t0.02
1000.4\t0.01
''')

    fs.CreateFile('ref_3.e', '''\
1000.0\t0.01
1000.1\t0.01
1000.2\t0.01
1000.3\t0.01
1000.4\t0.01
''')

    fs.CreateFile('test_main.prm', '''\
NbCompoMax=3
NbCompoMin=1
ref=ref_1.e
ref=ref_2.e
ref=ref_3.e
''')

    # sample 1 is twice ref_1
    fs.CreateFile('sample_1.e', '''\
1000.1\t0.02
1000.2\t0.04
1000.3\t0.06
''')

    # sample 2 is half ref_2
    fs.CreateFile('sample_2.e', '''\
1000.1\t0.015
1000.2\t0.010
1000.3\t0.005
''')

    # sample 3 is ref_1 plus ref_3
    fs.CreateFile('sample_3.e', '''\
1000.1\t0.03
1000.2\t0.04
1000.3\t0.05
''')

    test_main_config = get_config_parser()
    fs.CreateFile('test_main.cfg', '''\
[reference_spectra]
prm = test_main.prm

[data]
sample*.e

[output]
best_fit_files_dir = .
plots_pdf_fp = test_main_plots.pdf
table_fp = test_main_table.txt
reference_plots_pdf = test_main_reference_plots.pdf
''')

    result = cli(['test_main.cfg'])

    assert result.exit_code == 0
    assert os.path.exists('test_main_plots.pdf')
    assert os.path.exists('test_main_table.txt')
    assert os.path.exists('sample_1_fit.txt')
