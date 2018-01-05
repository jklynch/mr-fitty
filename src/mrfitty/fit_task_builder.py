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
from glob import glob
import logging
import os

from sklearn.linear_model import LinearRegression

from mrfitty.base import AdaptiveEnergyRangeBuilder, FixedEnergyRangeBuilder, PRM, ReferenceSpectrum, Spectrum
from mrfitty.best_subset_selection import BestSubsetSelectionFitTask
from mrfitty.combination_fit import AllCombinationFitTask
from mrfitty.linear_model import NonNegativeLinearRegression


class ConfigurationFileError(ValueError):
    pass


def get_config_parser():
    cp = configparser.ConfigParser(
        allow_no_value=True,
        delimiters=('=',)
    )
    cp.optionxform = lambda option: option
    return cp


def build_reference_spectrum_list_from_prm_file(prm_file_path):
    """
    Read a PRM file to create a list of ReferenceSpectrum
    instances, maximum component count, and minimum component
    count from a PRM file.

    :param prm_file_path:
    :return:
        list of ReferenceSpectrum instances
        maximum component count
        minimum component count
    """
    log = logging.getLogger(name=__file__)
    reference_spectrum_list = []
    log.info('reading PRM file {}'.format(prm_file_path))
    prm = PRM.read_prm(prm_file_path)
    # read reference files
    for i, fp in enumerate(prm.reference_file_path_list):
        log.info('reading reference file {}: {}'.format(i, fp))
        reference_spectrum = ReferenceSpectrum.read_file(fp)
        reference_spectrum_list.append(reference_spectrum)

    return reference_spectrum_list, prm.nb_component_max, prm.nb_component_min


def _get_required_config_value(config, section, option):
    if not config.has_option(section=section, option=option):
        raise ConfigurationFileError('section [{}] missing required option "{}"'.format(section, option))
    else:
        return config.get(section=section, option=option)


def _get_required_config_value_list(config, section):
    if not config.has_section(section=section):
        raise ConfigurationFileError('required section [{}] missing'.format(section))
    else:
        return config.items(section=section)


def build_reference_spectrum_list_from_config_prm_section(config):
    log = logging.getLogger(name=__name__)

    max_component_count = int(_get_required_config_value(config, 'prm', 'NBCompoMax'))
    min_component_count = int(_get_required_config_value(config, 'prm', 'NBCompoMin'))
    reference_spectrum_list = [
        ReferenceSpectrum.read_file(file_path_or_buffer=option_name)
        for option_name, option_value
        in _get_required_config_value_list(config, 'prm')
        if len(option_value) == 0]

    log.debug('NBCompoMax: %d', max_component_count)
    log.debug('NBCompoMin: %d', min_component_count)
    log.debug('Reference list length:\n  %d', len(reference_spectrum_list))

    if min_component_count <= 0:
        raise ConfigurationFileError('NBCompoMin must be greater than zero, not "{}"'.format(min_component_count))
    elif max_component_count <= 0:
        raise ConfigurationFileError('NBCompoMax must be greater than zero, not "{}"'.format(max_component_count))
    elif min_component_count > max_component_count:
        raise ConfigurationFileError(
            'NBCompoMin "{}" is greater than NBCompoMax "{}"'.format(min_component_count, max_component_count))
    else:
        return max_component_count, min_component_count, reference_spectrum_list


def build_reference_spectrum_list_from_config_file(config):
    """
    Read reference spectrum file glob(s) from configuration file to create
    and return a list of ReferenceSpectrum instances.

    :param config: configparser instance
    :return: list of ReferenceSpectrum instances
    """
    log = logging.getLogger(name=__name__)
    references = config.items('references')
    log.debug(references)
    reference_spectrum_list, _ = ReferenceSpectrum.read_all(
        [os.path.expanduser(reference_file_glob) for reference_file_glob, _ in references])

    if len(reference_spectrum_list) == 0:
        raise ConfigurationFileError('no reference spectrum files were found using globs "{}"'.format(references))
    else:
        return reference_spectrum_list


def build_unknown_spectrum_list_from_config_file(config):
    log = logging.getLogger(name=__name__)

    unknown_spectrum_file_path_list = []
    for j, (unknown_spectrum_glob, _) in enumerate(config.items('data')):
        log.info('unknown spectrum glob: {}'.format(unknown_spectrum_glob))
        glob_pattern_expanded = os.path.expanduser(unknown_spectrum_glob)
        unknown_spectrum_file_path_list.extend(glob(glob_pattern_expanded))
    log.info('found {} data files'.format(len(unknown_spectrum_file_path_list)))

    unknown_spectrum_list = []
    for unknown_spectrum_file_path in unknown_spectrum_file_path_list:
        log.info('reading data file {}'.format(unknown_spectrum_file_path))
        unknown_spectrum = Spectrum.read_file(unknown_spectrum_file_path)
        unknown_spectrum_list.append(unknown_spectrum)

    if len(unknown_spectrum_list) == 0:
        raise ConfigurationFileError('no spectrum files were found using globs "{}"'.format(config.items('data')))
    else:
        return unknown_spectrum_list


def get_fit_parameters_from_config_file(config, prm_max_cmp, prm_min_cmp):
    # these are the options specified in the [fit] section:
    #   maximum_component_count
    #   minimum_component_count
    #   fit_method: lsq or nnlsq
    #   component_count_method: combination_fit or best_subset_selection
    #
    log = logging.getLogger(name=__name__)

    if not config.has_section('fit'):
        raise ConfigurationFileError('required section [fit] is missing from configuration file')
    else:
        if (prm_max_cmp is None) and (not config.has_option('fit', 'maximum_component_count')):
            raise ConfigurationFileError(
                'required parameter maximum_component_count is missing '
                'from section [fit] in configuration file "{}"'.format(config))
        elif (prm_min_cmp is None) and (not config.has_option('fit', 'minimum_component_count')):
            raise ConfigurationFileError(
                'required parameter minimum_component_count is missing '
                'from section [fit] in configuration file {}'.format(config))
        else:
            max_cmp = config.getint('fit', 'maximum_component_count', fallback=2)
            if prm_max_cmp is not None:
                log.warning(
                    'MaxCompo={} from PRM will be used instead of'
                    'maximum_component_count={} from [fit] section.'.format(prm_max_cmp, max_cmp))
                max_cmp = prm_max_cmp

            min_cmp = config.getint('fit', 'minimum_component_count', fallback=1)
            if prm_min_cmp is not None:
                log.warning(
                    'MinCompo={} from PRM will be used instead of'
                    'minimum_component_count={} from [fit] section.'.format(prm_min_cmp, min_cmp))
                min_cmp = prm_min_cmp

        config_fit_method = config.get('fit', 'fit_method', fallback='lsq')
        if config_fit_method == 'lsq':
            fit_method_class = LinearRegression
        elif config_fit_method == 'nnlsq':
            fit_method_class = NonNegativeLinearRegression
        else:
            raise ConfigurationFileError(
                'Unrecognized fit_method "{}" in section [fit]. '
                'Use lsq for least-squares or nnlsq for non-negative least squares.'.format(config_fit_method))

        config_component_count_method = config.get('fit', 'component_count_method', fallback='combination_fit')
        if config_component_count_method == 'combination_fit':
            fit_task_class = AllCombinationFitTask
        elif config_component_count_method == 'best_subset_selection':
            fit_task_class = BestSubsetSelectionFitTask
        else:
            raise ConfigurationFileError(
                'unrecognized component_count_method "{}" in section [fit]'.format(config_component_count_method))

    return max_cmp, min_cmp, fit_method_class, fit_task_class


def get_plotting_parameters_from_config_file(config):
    if not config.has_section('plots'):
        raise ConfigurationFileError(
            'required section [plots] is missing from configuration file' )
    else:
        best_fits_plot_limit = config.getint('plots', 'best_fits_plot_limit', fallback=3)

    return best_fits_plot_limit


def build_fit_task(config):
    log = logging.getLogger(name=__name__)

    # read section [references]
    # support a PRM file such as
    #   prm = path/to/one.prm
    # or
    # a list of one or more file globs such as
    #   arsenic_2_reference_spectra/*.e
    #   arsenic_3_reference_spectra/*.e

    prm_max_cmp = None
    prm_min_cmp = None
    if config.has_section('references'):
        if config.has_option('references', 'prm'):
            prm_file_path = os.path.expanduser(config.get('references', 'prm'))
            reference_spectrum_list, prm_max_cmp, prm_min_cmp = build_reference_spectrum_list_from_prm_file(prm_file_path)
        else:
            reference_spectrum_list = build_reference_spectrum_list_from_config_file(config)
    elif config.has_section('reference_spectra'):
        if config.has_option('reference_spectra', 'prm'):
            prm_file_path = os.path.expanduser(config.get('reference_spectra', 'prm'))
            reference_spectrum_list, prm_max_cmp, prm_min_cmp = build_reference_spectrum_list_from_prm_file(prm_file_path)
        else:
            raise ConfigurationFileError('section [reference_spectra] is missing required parameter prm')
    else:
        raise ConfigurationFileError('configuration file is missing required section [references]')

    energy_range = get_energy_range_from_config(config)

    unknown_spectrum_list = build_unknown_spectrum_list_from_config_file(config)

    max_cmp, min_cmp, fit_method_class, fit_task_class = get_fit_parameters_from_config_file(
        config, prm_max_cmp, prm_min_cmp)

    best_fits_plot_limit = get_plotting_parameters_from_config_file(config)

    if 0 < min_cmp <= max_cmp:
        component_count_range = range(min_cmp, max_cmp+1)
        logging.info('component count range: {}'.format(component_count_range))
    else:
        raise ConfigurationFileError('minimum and maximum component counts are not valid')

    fit_task = fit_task_class(
        ls=fit_method_class,
        reference_spectrum_list=reference_spectrum_list,
        unknown_spectrum_list=unknown_spectrum_list,
        energy_range_builder=energy_range,
        component_count_range=component_count_range,
        best_fits_plot_limit=best_fits_plot_limit
    )

    return fit_task


def get_energy_range_from_config(config):
    log = logging.getLogger(name=__name__)
    if config.has_option('parameters', 'fit_energy_start') and config.has_option('parameters', 'fit_energy_stop'):
        fit_energy_start = config.getfloat('parameters', 'fit_energy_start')
        fit_energy_stop = config.getfloat('parameters', 'fit_energy_stop')
        energy_range = FixedEnergyRangeBuilder(fit_energy_start, fit_energy_stop)
        log.info('fitting with fixed energy range %d to %d', fit_energy_start, fit_energy_stop)
    elif not(config.has_option('parameters', 'fit_energy_start')) \
            and not(config.has_option('parameters', 'fit_energy_stop')):
        energy_range = AdaptiveEnergyRangeBuilder()
        log.info('fitting with adaptive energy ranges')
    else:
        raise Exception('only one of fit_energy_start and fit_energy_stop was specified in the configuration')

    return energy_range
