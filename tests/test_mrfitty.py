"""
Run tests from venv34.  Also run tox from venv34.
Set PYTHONPATH to the src directory and run py.test tests/ --cov=src/mrfitty.

export PYTHONPATH=./src/
source venv343/bin/activate
py.test tests/ --cov=src/mrfitty

"""
import logging
import os.path

from click.testing import CliRunner

from mrfitty.__main__ import main, get_config_parser


logging.basicConfig(level=logging.DEBUG)
log = logging.getLogger(name=__name__)


def test_main():
    runner = CliRunner()

    # write the necessary input files:
    #   configuration file
    #   reference files
    #   sample files
    #   prm file
    # then check for the expected output files
    #   plot pdf
    #   table txt
    #   sample fit files
    with runner.isolated_filesystem():
        # write three reference files
        # write a prm file
        # write three sample files
        # write a configuration file for this test
        write_reference_file(
            'ref_1.e',
'''
1000.0\t0.01
1000.1\t0.02
1000.2\t0.03
1000.3\t0.04
1000.4\t0.05
'''
        )
        write_reference_file(
            'ref_2.e',
'''
1000.0\t0.05
1000.1\t0.04
1000.2\t0.03
1000.3\t0.02
1000.4\t0.01
'''
        )
        write_reference_file(
            'ref_3.e',
'''
1000.0\t0.01
1000.1\t0.01
1000.2\t0.01
1000.3\t0.01
1000.4\t0.01
'''
        )
        write_prm_file(
            prm_file_path='test_main.prm',
            min_component_count=1,
            max_component_count=3,
            ref_file_block=
'''
ref=ref_1.e
ref=ref_2.e
ref=ref_3.e
'''
        )
        # sample 1 is twice ref_1
        write_spectrum_file(
            'sample_1.e',
'''
1000.1\t0.02
1000.2\t0.04
1000.3\t0.06
'''
        )
        # sample 2 is half ref_2
        write_spectrum_file(
            'sample_2.e',
'''
1000.1\t0.015
1000.2\t0.010
1000.3\t0.005
'''
        )
        # sample 3 is ref_1 plus ref_3
        write_spectrum_file(
            'sample_3.e',
'''
1000.1\t0.03
1000.2\t0.04
1000.3\t0.05
'''
        )
        test_main_config = get_config_parser()
        test_main_config.read_string('''
[references]
prm = test_main.prm

[data]
sample*.e

[parameters]

[output]
best_fit_files_dir = .
plots_pdf_fp = test_main_plots.pdf
table_fp = test_main_table.txt
reference_plots_pdf = test_main_reference_plots.pdf
''')

        with open('test_main.cfg', 'w') as test_main_config_file:
            test_main_config.write(test_main_config_file)

        result = runner.invoke(main, ['test_main.cfg'])

        assert result.exit_code == 0
        assert os.path.exists('test_main_plots.pdf')
        assert os.path.exists('test_main_table.txt')
        assert os.path.exists('sample_1_fit.txt')


def write_reference_file(reference_file_path, file_contents):
    log.info('writing reference file {}'.format(reference_file_path))
    with open(reference_file_path, 'w') as reference_file:
        reference_file.write(file_contents)


def write_prm_file(prm_file_path, min_component_count, max_component_count, ref_file_block):
    log.info('writing PRM file {}'.format(prm_file_path))
    with open(prm_file_path, 'w') as prm_file:
        prm_file.write('NbCompoMax={}\n'.format(max_component_count))
        prm_file.write('NbCompoMin={}\n'.format(min_component_count))
        prm_file.write(ref_file_block)


def write_spectrum_file(spectrum_file_path, file_contents):
    log.info('writing spectrum file {}'.format(spectrum_file_path))
    with open(spectrum_file_path, 'w') as spectrum_file:
        spectrum_file.write(file_contents)
