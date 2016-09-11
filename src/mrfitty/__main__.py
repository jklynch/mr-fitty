"""
The MIT License (MIT)

Copyright (c) 2015 Joshua Lynch, Sarah Nicholas

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
import configparser
import logging
import io
import os

import click

from mrfitty.combination_fit import AllCombinationFitTask
from mrfitty.best_subset_selection import BestSubsetSelectionFitTask


def get_config_parser():
    cp = configparser.ConfigParser(
        allow_no_value=True,
        delimiters=('=',)
    )
    cp.optionxform = lambda option: option
    return cp

@click.command()
@click.argument('config_fp', type=click.Path(exists=True))
def main(config_fp):
    logging.basicConfig(level=logging.INFO)
    log = logging.getLogger(name=__name__)

    config = get_config_parser()
    log.info('reading configuration file path: {}'.format(config_fp))
    with open(config_fp) as config_file:
        config.read_file(config_file)

    # log the configuration file
    with io.StringIO() as string_buffer:
        config.write(string_buffer)
        log.info('configuration file contents:\n{}'.format(string_buffer.getvalue()))

    fitter = BestSubsetSelectionFitTask.build(config)
    fitter.fit_all()

    table_file_path = config.get('output', 'table_fp', fallback=None)
    if table_file_path:
        log.info('writing table output to file path {}'.format(table_file_path))
        fitter.write_table(table_file_path)
    else:
        log.warning('No file path specified for table output')

    plots_pdf_file_path = config.get('output', 'plots_pdf_fp', fallback=None)
    if plots_pdf_file_path:
        log.info('writing plots to PDF {}'.format(plots_pdf_file_path))
        fitter.draw_plots_matplotlib(plots_pdf_file_path)
        plots_html_file_path = os.path.splitext(plots_pdf_file_path)[0] + '.html'
        log.info('writing plots to HTML {}'.format(plots_html_file_path))
        fitter.draw_plots_bokeh(plots_html_file_path)
    else:
        log.warning('No file path specified for plot output')

    best_fit_files_dir_path = config.get('output', 'best_fit_files_dir', fallback=None)
    if best_fit_files_dir_path:
        fitter.write_best_fit_arrays(best_fit_files_dir_path)
    else:
        log.warning('No directory specified for best fit files')
