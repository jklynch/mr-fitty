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
import argparse
import io
import logging
import os
import sys

from mrfitty import __version__
from mrfitty.fit_task_builder import ConfigurationFileError, build_fit_task, get_config_parser


def main():
    cli(sys.argv[1:])


def cli(argv):
    print('argv: {}'.format(argv))
    argparser = argparse.ArgumentParser()
    argparser.add_argument('config_fp', metavar='FILE', help='path to a configuration file')
    argparser.add_argument('--version', action='version', version='MrFitty version {}'.format(__version__))

    parsed_args = argparser.parse_args(argv)
    print('parsed args: {}'.format(parsed_args))
    fit(parsed_args)


def fit(args):
    config_fp = args.config_fp
    config = get_config_parser()
    print('reading configuration file path: {}'.format(config_fp))
    with open(config_fp) as config_file:
        config.read_file(config_file)

    level = config.get('log', 'level', fallback='INFO')
    logging.basicConfig(level=logging.getLevelName(level))
    log = logging.getLogger(name=__name__)

    # log the configuration file
    with io.StringIO() as string_buffer:
        config.write(string_buffer)
        log.info('configuration file contents:\n{}'.format(string_buffer.getvalue()))

    try:
        fitter = build_fit_task(config)
        plots_pdf_file_dir = os.path.expanduser(config.get('output', 'plots_pdf_dir', fallback=None))
        if not os.path.exists(plots_pdf_file_dir):
            log.info('output directory "%s" will be created', plots_pdf_file_dir)
            os.makedirs(plots_pdf_file_dir)
        fitter.fit_all(plots_pdf_dp=plots_pdf_file_dir)
    except ConfigurationFileError as e:
        log.error(e)
        log.exception('configuration file error in "%s"', config_fp)
        quit()

    table_file_path = os.path.expanduser(config.get('output', 'table_fp', fallback=None))
    if table_file_path:
        log.info('writing table output to file path {}'.format(table_file_path))
        fitter.write_table(table_file_path)
    else:
        log.warning('No file path specified for table output')

        #plots_pdf_file_path = os.path.expanduser(config.get('output', 'plots_pdf_fp', fallback=None))
        #if plots_pdf_file_path:
        #    log.info('writing plots to PDF {}'.format(plots_pdf_file_path))
        #    fitter.draw_plots_matplotlib(plots_pdf_file_path)
        #    plots_html_file_path = os.path.splitext(plots_pdf_file_path)[0] + '.html'
        #    log.info('writing plots to HTML {}'.format(plots_html_file_path))
        #    fitter.draw_plots_bokeh(plots_html_file_path)
        #else:
        #    log.warning('No file path specified for plot output')

        best_fit_files_dir_path = os.path.expanduser(config.get('output', 'best_fit_files_dir', fallback=None))
        if best_fit_files_dir_path:
            if not os.path.expanduser(best_fit_files_dir_path):
                log.info('creating directory %s', best_fit_files_dir_path)
                os.makedirs(best_fit_files_dir_path)
            fitter.write_best_fit_arrays(best_fit_files_dir_path)
        else:
            log.warning('No directory specified for best fit files')


if __name__ == '__main__':
    main()
