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
import collections
import glob
import itertools
import logging
from operator import attrgetter
import os.path

import bokeh.io
import bokeh.models.layouts
import bokeh.plotting

import matplotlib
matplotlib.use('pdf', warn=False, force=True)
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.offsetbox import AnchoredText
import numpy as np
import pandas as pd

from mrfitty.base import PRM, ReferenceSpectrum, Spectrum, InterpolatedReferenceSpectraSet, SpectrumFit
from mrfitty.linear_model import NonNegativeLinearRegression


class CombinationFitResults:
    """CombinationFitResults

    """
    def __init__(self, spectrum, best_fit, sorted_component_count_fit_lists):
        self.spectrum = spectrum
        self.best_fit = best_fit
        self.sorted_component_count_fit_lists = sorted_component_count_fit_lists


class AllCombinationFitTask:
    def __init__(self, ls, reference_spectrum_list, unknown_spectrum_list, energy_range_builder, component_count_range=range(4)):
        self.ls = ls
        self.reference_spectrum_list = reference_spectrum_list
        self.unknown_spectrum_list = unknown_spectrum_list
        self.energy_range_builder = energy_range_builder
        self.component_count_range = component_count_range
        self.fit_table = collections.OrderedDict()

    def fit_all(self):
        log = logging.getLogger(name='fit_all')
        for unknown_spectrum in self.unknown_spectrum_list:
            log.debug('fitting %s', unknown_spectrum.file_name)
            best_fit, sorted_component_count_fit_lists = self.fit(unknown_spectrum)
            self.fit_table[unknown_spectrum] = CombinationFitResults(
                spectrum=unknown_spectrum,
                best_fit=best_fit,
                sorted_component_count_fit_lists=sorted_component_count_fit_lists
            )
        return self.fit_table

    def fit(self, unknown_spectrum):
        log = logging.getLogger(name=unknown_spectrum.file_name)
        log.info('fitting unknown spectrum %s', unknown_spectrum.file_name)
        interpolated_reference_spectra = InterpolatedReferenceSpectraSet(
            unknown_spectrum=unknown_spectrum,
            reference_set=self.reference_spectrum_list)
        # fit all combinations of reference_spectra
        all_counts_spectrum_fit_table = collections.defaultdict(list)
        for reference_spectra_combination in self.reference_combination_iter(self.component_count_range):
            log.debug('fitting to reference_spectra {}'.format(reference_spectra_combination))

            #spectrum_fit = self.fit_unknown_spectrum_to_references(
            #    unknown_spectrum=unknown_spectrum,
            #    reference_spectra_combination=reference_spectra_combination
            #)
            spectrum_fit = self.fit_references_to_unknown(
                interpolated_reference_spectra=interpolated_reference_spectra,
                reference_spectra_subset=reference_spectra_combination)
            reference_count = len(reference_spectra_combination)
            spectrum_fit_list = all_counts_spectrum_fit_table[reference_count]
            spectrum_fit_list.append(spectrum_fit)
            spectrum_fit_list.sort(key=attrgetter('nss'))
            # when there are many reference spectra the list of fits can get extremely long
            # and eat up all the memory
            if len(spectrum_fit_list) > 100:
                spectrum_fit_list.pop()

        all_counts_spectrum_fit_list = []
        for reference_count, spectrum_fit_list in all_counts_spectrum_fit_table.items():
            all_counts_spectrum_fit_list.extend(spectrum_fit_list)
        best_fit, sorted_component_count_fits = self.sort_fits(unknown_spectrum, all_counts_spectrum_fit_list)
        return best_fit, sorted_component_count_fits

    def reference_combination_iter(self, component_count_range):
        for component_count in component_count_range:
            for reference_spectra_combination in itertools.combinations(self.reference_spectrum_list, component_count):
                yield reference_spectra_combination

    #@profile
    def fit_references_to_unknown(self, interpolated_reference_spectra, reference_spectra_subset):
        interpolated_reference_spectra_subset_df, unknown_spectrum_df = \
            interpolated_reference_spectra.get_reference_subset_and_unknown_df(
                reference_list=reference_spectra_subset)

        #self.log.debug('unknown_spectrum_df:\n%s', unknown_spectrum_df.head())
        #self.log.debug('interpolated_reference_spectra_subset_df:\n%s', interpolated_reference_spectra_subset_df.head())

        lm = self.ls()
        lm.fit(interpolated_reference_spectra_subset_df.values, unknown_spectrum_df.norm.values)
        reference_spectra_coef_x = lm.coef_

        spectrum_fit = SpectrumFit(
            interpolant_incident_energy=interpolated_reference_spectra_subset_df.index,
            reference_spectra_A_df=interpolated_reference_spectra_subset_df,
            unknown_spectrum_b=unknown_spectrum_df,
            reference_spectra_seq=reference_spectra_subset,
            reference_spectra_coef_x=reference_spectra_coef_x
        )
        return spectrum_fit

    #@profile
    def fit_unknown_spectrum_to_references(self, unknown_spectrum, reference_spectra_combination):
        log = logging.getLogger(name=self.__class__.__name__)
        fit_energies, fit_energies_ndx = self.energy_range_builder.build_range(
            unknown_spectrum,
            reference_spectra_combination
        )
        # interpolate the reference spectra at the fit_energies
        log.debug('fit energies.shape: %s', fit_energies.shape)
        # reference_spectra_A_df is energies x components
        unknown_spectrum_b = unknown_spectrum.data_df.loc[fit_energies_ndx, 'norm']
        # log.debug('unknown_spectrum_b.shape: {}'.format(unknown_spectrum_b.shape))
        log.debug('unknown_spectrum_b      : %s', unknown_spectrum_b)
        # log.debug('reference combination  : {}'.format(reference_spectra_combination))
        reference_spectra_A_column_list = []
        # create an array to hold the interpolated reference spectra
        data = np.zeros((fit_energies.shape[0], len(reference_spectra_combination)))
        for i, rs in enumerate(reference_spectra_combination):
            reference_spectra_A_column_list.append(rs.file_name)
            data[:, i] = rs.interpolant(fit_energies)
        # use the (time) index from the unknown spectrum for the interpolated reference spectra
        # this is important because pandas Series and Dataframes will align on their
        # indexes for most operations so for example calculating residuals can result
        # in the wrong shape
        reference_spectra_A_df = pd.DataFrame(
            data=data, index=unknown_spectrum_b.index, columns=reference_spectra_A_column_list)
        log.debug('reference_spectra_A_df columns: %s', reference_spectra_A_df.columns)

        # it is important to label the columns in the order they were appended

        ls = NonNegativeLinearRegression()
        ls.fit(reference_spectra_A_df.values, unknown_spectrum_b)
        reference_spectra_coef_x = ls.coef_

        #reference_spectra_coef_x, residual, *extra = nnls(
        #    reference_spectra_A_df.values,
        #    unknown_spectrum_b
        #)
        #reference_spectra_coef_x, residual, rank, sigma = lstsq(
        #    reference_spectra_A_df.values,
        #    unknown_spectrum_b
        #)
        # log.debug('A        :\n{}'.format(reference_spectra_A_df))
        # log.debug('coef     : {}'.format(reference_spectra_coef_x))
        # log.debug('residual : {}'.format(residual))
        # log.debug('solution :\n{}'.format(reference_spectra_A_df.dot(reference_spectra_coef_x)))
        #if np.any(reference_spectra_coef_x < 0.0):
        #    # this happens a lot with lstsq and is generally not a problem
        #    # print('{}: least-squares fit has negative coefficients'.format(
        #    #    unknown_spectrum_file_name
        #    # ))
        #    continue
        ##else:
        spectrum_fit = SpectrumFit(
            interpolant_incident_energy=fit_energies,
            reference_spectra_A_df=reference_spectra_A_df,
            unknown_spectrum_b=unknown_spectrum_b,
            reference_spectra_seq=reference_spectra_combination,
            reference_spectra_coef_x=reference_spectra_coef_x
        )
        return spectrum_fit

    def sort_fits(self, spectrum, spectrum_fit_list):
        # sort all fits with the same component count
        log = logging.getLogger(name=spectrum.file_name)
        # compare the top fits for each component count
        # component_count_fit_lists looks like this:
        #   [ [list of 0-component fits], [list of 1-component fits], ..., [list of n-component fits] ]
        component_count_fit_lists = [[] for component_count in self.component_count_range]
        # append an extra empty list for 0-component fits even though there are none
        # this allows component count to work as the list index
        component_count_fit_lists.append([])
        log.debug(
            'creating one fit list for each component count in self.component_count_range: %s',
            self.component_count_range)
        log.debug('component_count_fit_lists: %s', component_count_fit_lists)

        # populate the component count lists from a sorted list of all fits
        for spectrum_fit in sorted(spectrum_fit_list, key=attrgetter('nss')):
            component_count = len(spectrum_fit.reference_spectra_seq)
            component_count_fit_lists[component_count].append(spectrum_fit)

        for component_count, component_count_fit_list in enumerate(component_count_fit_lists):
            if len(component_count_fit_list) > 0:
                log.debug('best fit for %d component(s): %s', component_count, component_count_fit_list[0])
            else:
                log.debug('no fits for component count %s', component_count)

        best_fit_for_component_count_list = [c[0] for c in component_count_fit_lists[1:]]
        best_fit = self.choose_best_component_count(best_fit_for_component_count_list)
        return best_fit, component_count_fit_lists[1:]
        # choose the best fit from the top fits for all component counts
        # skip the 0-component list since it is empty

    def choose_best_component_count(self, best_fit_for_component_count_list):
        """
        Choose the best fit from the best fits for each component count.
        :param best_fit_for_component_count_list:
          a list of fits for component counts 0 to N; there is no 0-component fit
        :return:
        """
        log = logging.getLogger(name=self.__class__.__name__)
        best_fit = None
        previous_nss = 1.0
        for component_count, component_count_fit_list in enumerate(best_fit_for_component_count_list):
            best_fit_for_component_count = component_count_fit_list
            improvement = (previous_nss - best_fit_for_component_count.nss) / previous_nss
            log.debug('improvement: {:5.3f} for {}'.format(improvement, best_fit_for_component_count))
            if improvement < 0.10:
                break
            else:
                best_fit = best_fit_for_component_count
                previous_nss = best_fit.nss
        log.debug('best fit: {}'.format(best_fit))
        return best_fit

    def write_table(self, table_file_path):
        """
        sample name, residual, reference 1, fraction 1, reference 2, fraction 2, ...
        :param table_file_path:
        :return:
        """
        with open(table_file_path, 'wt') as table_file:
            table_file.write('spectrum\tNSS\tresidual percent\treference 1\tpercent 1\treference 2\tpercent 2\treference 3\tpercent 3\n')
            for spectrum, fit_results in self.fit_table.items():
                table_file.write(spectrum.file_name)
                table_file.write('\t')
                table_file.write('{:5.3f}\t'.format(fit_results.best_fit.nss))
                table_file.write('{:5.3f}'.format(fit_results.best_fit.residuals_contribution))
                for row in fit_results.best_fit.reference_contribution_percent_sr.sort_values(ascending=False).iteritems():
                    table_file.write('\t')
                    table_file.write(row[0])
                    table_file.write('\t{:5.3f}'.format(row[1]))
                table_file.write('\n')

    def draw_plots_matplotlib(self, plots_pdf_file_path):
        log = logging.getLogger(name=self.__class__.__name__)
        with PdfPages(plots_pdf_file_path) as plot_file:
            log.info('writing plots file {}'.format(plots_pdf_file_path))
            for spectrum, fit_results in self.fit_table.items():
                log.info('plotting fit for {}'.format(spectrum.file_name))

                longest_name_len = max([len(name) for name in fit_results.best_fit.reference_contribution_percent_sr.index])
                # the format string should look like '{:N}{:5.2f}' where N is the length of the longest reference name
                contribution_format_str = '{:' + str(longest_name_len + 4) + '}{:5.2f}'
                contribution_desc_lines = []
                fit_results.best_fit.reference_contribution_percent_sr.sort_values(ascending=False, inplace=True)
                for name, value in fit_results.best_fit.reference_contribution_percent_sr.iteritems():
                    contribution_desc_lines.append(contribution_format_str.format(name, value))
                contribution_desc_lines.append(
                    contribution_format_str.format('residual', fit_results.best_fit.residuals_contribution))
                contribution_desc = '\n'.join(contribution_desc_lines)

                f, ax = plt.subplots()
                f.suptitle(spectrum.file_name)
                log.info(fit_results.best_fit.fit_spectrum_b.shape)
                ax.plot(fit_results.best_fit.interpolant_incident_energy, fit_results.best_fit.fit_spectrum_b)
                log.info(fit_results.best_fit.residuals.shape)
                ax.plot(fit_results.best_fit.interpolant_incident_energy, fit_results.best_fit.unknown_spectrum_b, '.')
                ax.plot(fit_results.best_fit.interpolant_incident_energy, fit_results.best_fit.residuals)

                at = AnchoredText(contribution_desc, loc=1, prop=dict(fontname='Monospace', size=10))
                ax.add_artist(at)

                #log.info('fit_results.best_fit.interpolant_incident_energy:\n{}'.format(
                #    fit_results.best_fit.interpolant_incident_energy)
                #)
                #ax.vlines(x=[
                #        fit_results.best_fit.interpolant_incident_energy.iloc[0],
                #        fit_results.best_fit.interpolant_incident_energy.iloc[-1]
                #    ],
                #    ymin=fit_results.best_fit.unknown_spectrum_b.min(),
                #    ymax=fit_results.best_fit.unknown_spectrum_b.max()
                #)
                plot_file.savefig(f)

    def draw_plots_bokeh(self, plots_html_file_path):
        log = logging.getLogger(name=self.__class__.__name__)
        bokeh.io.output_file(plots_html_file_path)
        #with PdfPages(plots_pdf_file_path) as plot_file:
        log.info('writing plots file {}'.format(plots_html_file_path))
        plot_list = []
        for spectrum, fit_results in self.fit_table.items():
            log.info('plotting fit for {}'.format(spectrum.file_name))
            #f, ax = plt.subplots()
            f = bokeh.plotting.figure(
                title='{}\n???'.format(spectrum.file_name)
            )
            plot_list.append(f)

            log.info(fit_results.best_fit.fit_spectrum_b.shape)
            #ax.plot(fit_results.best_fit.interpolant_incident_energy, fit_results.best_fit.fit_spectrum_b)
            f.line(
                fit_results.best_fit.interpolant_incident_energy,
                fit_results.best_fit.fit_spectrum_b,
                line_width=2
            )
            log.info(fit_results.best_fit.residuals.shape)
            #ax.plot(fit_results.best_fit.interpolant_incident_energy, fit_results.best_fit.unknown_spectrum_b, '.')
            f.circle(
                fit_results.best_fit.interpolant_incident_energy,
                fit_results.best_fit.unknown_spectrum_b
            )
            #ax.plot(fit_results.best_fit.interpolant_incident_energy, fit_results.best_fit.residuals)
            f.line(
                fit_results.best_fit.interpolant_incident_energy,
                fit_results.best_fit.residuals
            )
            #log.info('fit_results.best_fit.interpolant_incident_energy:\n{}'.format(
            #    fit_results.best_fit.interpolant_incident_energy)
            #)
            #ax.vlines(x=[
            #        fit_results.best_fit.interpolant_incident_energy.iloc[0],
            #        fit_results.best_fit.interpolant_incident_energy.iloc[-1]
            #    ],
            #    ymin=fit_results.best_fit.unknown_spectrum_b.min(),
            #    ymax=fit_results.best_fit.unknown_spectrum_b.max()
            #)
            #plot_file.savefig(f)
        p = bokeh.models.layouts.Column(*plot_list)
        bokeh.io.save(p)

    def write_best_fit_arrays(self, best_fit_dir_path):
        log = logging.getLogger(name=self.__class__.__name__)
        for spectrum, fit_results in self.fit_table.items():
            file_base_name, file_name_ext = os.path.splitext(spectrum.file_name)
            fit_file_path = os.path.join(best_fit_dir_path, file_base_name + '_fit.txt')
            log.info('writing best fit to {}'.format(fit_file_path))

            fit_df = pd.DataFrame(
                {
                    'energy': fit_results.best_fit.interpolant_incident_energy,
                    'spectrum': fit_results.best_fit.unknown_spectrum_b,
                    'fit': fit_results.best_fit.fit_spectrum_b,
                    'residual': fit_results.best_fit.residuals}
            )
            fit_df.to_csv(fit_file_path, sep='\t', float_format='%8.4f', index=False)

    @classmethod
    def build_reference_spectrum_list_from_prm_file(cls, prm_file_path):
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
        log = logging.getLogger(name=cls.__class__.__name__)
        reference_spectrum_list = []
        log.info('reading PRM file {}'.format(prm_file_path))
        prm = PRM.read_prm(prm_file_path)
        # read reference files
        for i, fp in enumerate(prm.reference_file_path_list):
            log.info('reading reference file {}: {}'.format(i, fp))
            reference_spectrum = ReferenceSpectrum.read_file(fp)
            reference_spectrum_list.append(reference_spectrum)

        return reference_spectrum_list, prm.nb_component_max, prm.nb_component_min

    @classmethod
    def build_reference_spectrum_list_from_config_file(cls, config):
        """
        Read reference spectrum file glob(s) from configuration file to create
        a list of ReferenceSpectrum instances, maximum component count, and
        minimum component count.

        :param config: configparser instance
        :return: list of ReferenceSpectrum instances
        """

        reference_spectrum_list, reference_spectrum_file_glob_list = ReferenceSpectrum.read_all(
            [reference_file_glob for reference_file_glob, _ in config.items('references')]
        )

        if len(reference_spectrum_file_glob_list) == 0:
            logging.exception(
                'no reference spectrum file paths or patterns were found in section [references] of configuration file {}'.format(
                    config
                )
            )
        elif len(reference_spectrum_list) == 0:
            logging.exception('no reference spectrum files were found')
        else:
            # everything is ok
            pass

        if not config.has_option('fit', 'maximum_component_count'):
            logging.exception(
                'required parameter maximum_component_count is missing from section [fit] in configuration file {}'.format(
                    config
                )
            )
        elif not config.has_option('fit', 'minimum_component_count'):
            logging.exception(
                'required parameter minimum_component_count is missing from section [fit] in configuration file {}'.format(
                    config
                )
            )
        else:
            maximum_component_count = config.getint('fit', 'maximum_component_count')
            minimum_component_count = config.getint('fit', 'minimum_component_count')

        return reference_spectrum_list, maximum_component_count, minimum_component_count

    @classmethod
    def build(cls, config):
        log = logging.getLogger(name=str(cls))

        # read section [references]
        # support a PRM file such as
        #   prm = path/to/one.prm
        # or
        # a list of one or more file globs such as
        #   arsenic_2_reference_spectra/*.e
        #   arsenic_3_reference_spectra/*.e

        if config.has_section('references'):
            if config.has_option('references', 'prm'):
                prm_file_path = os.path.expanduser(config.get('references', 'prm'))
                reference_spectrum_list, max_cmp, min_cmp = cls.build_reference_spectrum_list_from_prm_file(prm_file_path)
            else:
                reference_spectrum_list, max_cmp, min_cmp = cls.build_reference_spectrum_list_from_config_file(config)
        elif config.has_section('reference_spectra'):
            if config.has_option('reference_spectra', 'prm'):
                prm_file_path = os.path.expanduser(config.get('reference_spectra', 'prm'))
                reference_spectrum_list, max_cmp, min_cmp = cls.build_reference_spectrum_list_from_prm_file(prm_file_path)
            else:
                raise Exception('section [reference_spectra] is missing required parameter prm')
        else:
            raise Exception('configuration file is missing section [references]')

        if 0 < min_cmp <= max_cmp:
            component_count_range = range(min_cmp, max_cmp+1)
            logging.info('component count range: {}'.format(component_count_range))
        else:
            logging.exception('minimum and maximum component counts are not valid')

        energy_range = cls.get_energy_range_from_config(config)

        # read data files
        unknown_spectrum_file_path_list = []
        for j, (unknown_spectrum_glob, _) in enumerate(config.items('data')):
            log.info('unknown spectrum glob: {}'.format(unknown_spectrum_glob))
            glob_pattern_expanded = os.path.expanduser(unknown_spectrum_glob)
            unknown_spectrum_file_path_list.extend(glob.glob(glob_pattern_expanded))
        log.info('found {} data files'.format(len(unknown_spectrum_file_path_list)))

        unknown_spectrum_list = []
        for unknown_spectrum_file_path in unknown_spectrum_file_path_list:
            log.info('reading data file {}'.format(unknown_spectrum_file_path))
            unknown_spectrum = Spectrum.read_file(unknown_spectrum_file_path)
            unknown_spectrum_list.append(unknown_spectrum)

        fit_task = cls(
            reference_spectrum_list=reference_spectrum_list,
            unknown_spectrum_list=unknown_spectrum_list,
            energy_range_builder=energy_range,
            component_count_range=component_count_range
        )

        return fit_task

    @classmethod
    def get_energy_range_from_config(cls, config):
        if config.has_option('parameters', 'fit_energy_start') and config.has_option('parameters', 'fit_energy_stop'):
            fit_energy_start = config.getfloat('parameters', 'fit_energy_start')
            fit_energy_stop = config.getfloat('parameters', 'fit_energy_stop')
            energy_range = FixedEnergyRangeBuilder(fit_energy_start, fit_energy_stop)
        elif not(config.has_option('parameters', 'fit_energy_start') or config.has_option('parameters', 'fit_energy_stop')):
            energy_range = AdaptiveEnergyRangeBuilder()
        else:
            raise Exception('only one of fit_energy_start and fit_energy_stop was specified in the configuration')

        return energy_range
