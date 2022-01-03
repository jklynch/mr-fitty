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
import itertools
import logging
from operator import attrgetter
import os.path
import time

import bokeh.io
import bokeh.models.layouts
import bokeh.plotting

import matplotlib
matplotlib.use('pdf', force=True)
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd

from mrfitty.base import InterpolatedReferenceSpectraSet, SpectrumFit


class FitFailed(Exception):
    pass


class CombinationFitResults:
    """CombinationFitResults

    """
    def __init__(self, spectrum, best_fit, component_count_fit_table):
        self.spectrum = spectrum
        self.best_fit = best_fit
        self.component_count_fit_table = component_count_fit_table


class AllCombinationFitTask:
    def __init__(self, ls, reference_spectrum_list, unknown_spectrum_list, energy_range_builder, best_fits_plot_limit, component_count_range=range(4)):
        self.ls = ls
        self.reference_spectrum_list = reference_spectrum_list
        self.unknown_spectrum_list = unknown_spectrum_list
        self.energy_range_builder = energy_range_builder
        self.best_fits_plot_limit = best_fits_plot_limit
        self.component_count_range = component_count_range

        self.fit_table = collections.OrderedDict()

    def fit_all(self, plots_pdf_dp):
        log = logging.getLogger(name='fit_all')
        for unknown_spectrum in self.unknown_spectrum_list:
            log.debug('fitting %s', unknown_spectrum.file_name)
            t0 = time.time()
            best_fit, fit_table = self.fit(unknown_spectrum)
            t1 = time.time()
            log.info('fit %s in %5.3fs', unknown_spectrum.file_name, t1-t0)

            fit_results = CombinationFitResults(
                spectrum=unknown_spectrum,
                best_fit=best_fit,
                component_count_fit_table=fit_table
            )
            self.fit_table[unknown_spectrum] = fit_results

            file_base_name, _ = os.path.splitext(os.path.basename(unknown_spectrum.file_name))
            plots_pdf_fp = os.path.join(plots_pdf_dp, file_base_name + '_fit.pdf')
            with PdfPages(plots_pdf_fp) as plot_file:
                log.info('writing plots file {}'.format(plots_pdf_dp))
                # create plot
                log.info('plotting fit for %s', unknown_spectrum.file_name)

                f = self.plot_fit(spectrum=unknown_spectrum, any_given_fit=fit_results.best_fit, title='Best Fit')
                plot_file.savefig(f)

                f = self.plot_stacked_fit(spectrum=unknown_spectrum, any_given_fit=fit_results.best_fit, title='Best Fit')
                plot_file.savefig(f)

                ordinal_list = ('1st', '2nd', '3rd', '4th', '5th', '6th', '7th', '8th', '9th', '10th')

                # plot the best n-component fit
                for n in sorted(fit_table.keys()):
                    log.info('plotting %d-component fit for %s', n, unknown_spectrum.file_name)
                    n_component_fit_results = fit_table[n]
                    # TODO: move this loop to best subset selection
                    # here only plot the best fit for each component count
                    for i, fit in enumerate(n_component_fit_results):
                        if hasattr(fit, 'median_Cp'):
                            f = self.plot_fit(
                                spectrum=unknown_spectrum,
                                any_given_fit=fit,
                                title='Best {}-Component Fit ({})'.format(n, i))
                            plot_file.savefig(f)
                        else:
                            # plot just the best fit for n component(s)
                            if i < self.best_fits_plot_limit:
                                f = self.plot_fit(
                                    spectrum=unknown_spectrum,
                                    any_given_fit=n_component_fit_results[i],
                                    title='{} Best {}-Component Fit'.format(ordinal_list[i], n))
                                plot_file.savefig(f)
                            else:
                                break

                f = self.plot_nss_path(spectrum=unknown_spectrum, fit_results=fit_results, title='NSS Path')
                plot_file.savefig(f)

            nss_path_plot_fp = os.path.join(plots_pdf_dp, file_base_name + '_nss_path.html')
            self.bokeh_nss_path(spectrum=unknown_spectrum, fit_results=fit_results, output_fp=nss_path_plot_fp)

        return self.fit_table

    def fit(self, unknown_spectrum):
        log = logging.getLogger(name=unknown_spectrum.file_name)
        log.info('fitting unknown spectrum %s', unknown_spectrum.file_name)
        interpolated_reference_spectra = InterpolatedReferenceSpectraSet(
            unknown_spectrum=unknown_spectrum,
            reference_set=self.reference_spectrum_list)
        # fit all combinations of reference_spectra
        # all_counts_spectrum_fit_table looks like this:
        #   { 1: [...list of 1-component fits sorted by NSS...],
        #     2: [...list of 2-component fits sorted by NSS...],
        #     ...
        #   }
        all_counts_spectrum_fit_table = collections.defaultdict(list)
        for reference_spectra_combination in self.reference_combination_iter(self.component_count_range):
            log.debug('fitting to reference_spectra {}'.format(reference_spectra_combination))

            try:
                spectrum_fit = self.fit_references_to_unknown(
                    interpolated_reference_spectra=interpolated_reference_spectra,
                    reference_spectra_subset=reference_spectra_combination)
                reference_count = len(reference_spectra_combination)
                spectrum_fit_list = all_counts_spectrum_fit_table[reference_count]
                spectrum_fit_list.append(spectrum_fit)
                spectrum_fit_list.sort(key=attrgetter('nss'))
                # when there are many reference spectra the list of fits can get extremely long
                # and eat up all the memory
                # keep only the top 100 fits for each component count
                if len(spectrum_fit_list) > 100:
                    spectrum_fit_list.pop()
            except FitFailed as ff:
                # this is a common occurrence when using ordinary linear regression
                # it is not an 'error' just something that happens and needs to be handled
                log.debug(
                    'failed to fit unknown "%s" to references\n\t%s',
                    unknown_spectrum.file_name,
                    '\n\t'.join([r.file_name for r in reference_spectra_combination]))
                log.debug(ff)

        """
        TODO: this should go somewhere else
        # calculate prediction error on the best fits to find indistinguishable fits
        for component_count, fit_list in all_counts_spectrum_fit_table.items():
            for f, fit in enumerate(fit_list):
                prediction_error_list = self.calculate_prediction_error_list(fit)
                fit.median_Cp = np.median(prediction_error_list)
                fit.median_Cp_ci_lo, fit.median_Cp_ci_hi = scikits.bootstrap.ci(
                    data=prediction_error_list,
                    statfunction=np.median)
                log.info(
                    'fit %d has Cp %8.5f <-- %8.5f --> %8.5f',
                    f, fit.median_Cp_ci_lo, fit.median_Cp, fit.median_Cp_ci_hi)
                # in the first iteration fit_list[0] == fit
                if fit_list[0].median_Cp_ci_hi < fit.median_Cp_ci_lo:
                    log.info('median Cp confidence interval for fit %d does not overlap that of the best fit', f)
                    break
        """
        best_fit = self.choose_best_component_count(all_counts_spectrum_fit_table)
        return best_fit, all_counts_spectrum_fit_table

    def reference_combination_iter(self, component_count_range):
        for component_count in component_count_range:
            for reference_spectra_combination in itertools.combinations(self.reference_spectrum_list, component_count):
                yield reference_spectra_combination

    def fit_references_to_unknown(self, interpolated_reference_spectra, reference_spectra_subset):
        interpolated_reference_spectra_subset_df, unknown_spectrum_df = \
            interpolated_reference_spectra.get_reference_subset_and_unknown_df(
                reference_list=reference_spectra_subset, energy_range_builder=self.energy_range_builder)

        lm = self.ls()
        lm.fit(interpolated_reference_spectra_subset_df.values, unknown_spectrum_df.norm.values)
        if any(lm.coef_ < 0.0):
            msg = 'negative coefficients while fitting:\n{}'.format(lm.coef_)
            raise FitFailed(msg)
        else:
            reference_spectra_coef_x = lm.coef_

            spectrum_fit = SpectrumFit(
                interpolant_incident_energy=interpolated_reference_spectra_subset_df.index,
                reference_spectra_A_df=interpolated_reference_spectra_subset_df,
                unknown_spectrum_b=unknown_spectrum_df,
                reference_spectra_seq=reference_spectra_subset,
                reference_spectra_coef_x=reference_spectra_coef_x
            )
            return spectrum_fit

    def choose_best_component_count(self, all_counts_spectrum_fit_table):
        """
        Choose the best fit from the best fits for each component count.
        :param all_counts_spectrum_fit_table:
          dictionary with component count keys and values list of spectrum fits in sorted order
        :return: instance of SpectrumFit
        """
        log = logging.getLogger(name=self.__class__.__name__)
        best_fit = None
        previous_nss = 1.0
        for component_count in sorted(all_counts_spectrum_fit_table.keys()):
            best_fit_for_component_count = all_counts_spectrum_fit_table[component_count][0]
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
                table_file.write('{:8.5f}\t'.format(fit_results.best_fit.nss))
                table_file.write('{:5.3f}'.format(fit_results.best_fit.residuals_contribution))
                for ref_name, ref_pct in fit_results.best_fit.reference_contribution_percent_sr.sort_values(ascending=False).items():
                    table_file.write('\t')
                    table_file.write(ref_name)
                    table_file.write('\t{:5.3f}'.format(ref_pct))
                table_file.write('\n')

    def draw_plots_matplotlib(self, plots_pdf_file_path):
        log = logging.getLogger(name=self.__class__.__name__)
        with PdfPages(plots_pdf_file_path) as plot_file:
            log.info('writing plots file {}'.format(plots_pdf_file_path))
            for spectrum, fit_results in self.fit_table.items():
                log.info('plotting fit for {}'.format(spectrum.file_name))

                f = self.plot_fit(spectrum, fit_results.best_fit)
                plot_file.savefig(f)

                # plot the best n-component fit
                #for n_fit_list in fit_results:
                # plot the best 2-component fit

    def plot_nss_path(self, spectrum, fit_results, title):
        log = logging.getLogger(name=self.__class__.__name__)

        f, ax = plt.subplots()
        f.suptitle(spectrum.file_name + '\n' + title)
        for component_count in fit_results.component_count_fit_table.keys():
            sorted_fits = fit_results.component_count_fit_table[component_count]
            ax.plot(range(len(sorted_fits)), [spectrum_fit.nss for spectrum_fit in sorted_fits])

        return f

    def plot_fit(self, spectrum, any_given_fit, title):
        log = logging.getLogger(name=self.__class__.__name__)

        f, ax = plt.subplots()
        f.suptitle(spectrum.file_name + '\n' + title + ' (NSS: {:8.5f})'.format(any_given_fit.nss))
        log.info(any_given_fit.fit_spectrum_b.shape)

        reference_contributions_percent_sr = any_given_fit.get_reference_contributions_sr()
        reference_only_contributions_percent_sr = any_given_fit.get_reference_only_contributions_sr()
        longest_name_len = max([len(name) for name in reference_contributions_percent_sr.index] + [len(spectrum.file_name)])
        # the format string should look like '{:N}{:5.2f} ({:5.2f})' where N is the length of the longest reference name
        reference_contribution_format_str = '{:' + str(longest_name_len + 4) + '}{:5.2f} ({:5.2f})'
        residuals_contribution_format_str = '{:' + str(longest_name_len + 4) + '}{:5.2f}'

        # add fits in descending order of reference contribution
        reference_line_list = []
        reference_label_list = []
        reference_contributions_percent_sr.sort_values(ascending=False, inplace=True)
        reference_only_contributions_percent_sr.sort_values(ascending=False, inplace=True)
        log.info('plotting reference components')
        log.info(reference_contributions_percent_sr.head())
        for (ref_name, ref_contrib), (ref_only_name, ref_only_contrib) \
                in zip(reference_contributions_percent_sr.items(), reference_only_contributions_percent_sr.items()):
            log.info('reference contribution {} {:5.2f}'.format(ref_name, ref_contrib))
            log.info('reference-only contribution {} {:5.2f}'.format(ref_only_name, ref_only_contrib))
            reference_label = reference_contribution_format_str.format(ref_name, ref_contrib, ref_only_contrib)
            reference_label_list.append(reference_label)

            # plot once for each reference just to build the legend
            # ax.plot returns a list
            reference_line_list.extend(
                ax.plot(
                    any_given_fit.interpolant_incident_energy,
                    any_given_fit.fit_spectrum_b,
                    label=reference_label,
                    color='w',
                    alpha=0.0))

        log.info(any_given_fit.residuals.shape)
        residuals_label = residuals_contribution_format_str.format('residuals', any_given_fit.residuals_contribution)
        residuals_line = ax.plot(
            any_given_fit.interpolant_incident_energy,
            any_given_fit.residuals,
            label=residuals_label)

        fit_line_label = 'fit'
        fit_line = ax.plot(
            any_given_fit.interpolant_incident_energy,
            any_given_fit.fit_spectrum_b,
            label=fit_line_label)

        spectrum_points = ax.plot(
            any_given_fit.interpolant_incident_energy,
            any_given_fit.unknown_spectrum_b,
            '.',
            label=spectrum.file_name,
            alpha=0.5)

        ax.set_xlabel('eV')
        ax.set_ylabel('normalized absorbance')
        # 20171029
        ax.legend(
            [*reference_line_list, *spectrum_points, *residuals_line, *fit_line],
            [*reference_label_list, spectrum.file_name, residuals_label, fit_line_label],
            prop=dict(family='Monospace', size=7))

        return f

    def plot_stacked_fit(self, spectrum, any_given_fit, title):
        log = logging.getLogger(name=self.__class__.__name__)

        f, ax = plt.subplots()
        f.suptitle(spectrum.file_name + '\n' + title + ' (NSS: {:8.5f})'.format(any_given_fit.nss))
        log.info(any_given_fit.fit_spectrum_b.shape)

        reference_contributions_percent_sr = any_given_fit.get_reference_contributions_sr()
        longest_name_len = max([len(name) for name in reference_contributions_percent_sr.index] + [len(spectrum.file_name)])
        # the format string should look like '{:N}{:5.2f}' where N is the length of the longest reference name
        contribution_format_str = '{:' + str(longest_name_len + 4) + '}{:5.2f}'

        log.info(any_given_fit.residuals.shape)
        residuals_label = contribution_format_str.format('residuals', any_given_fit.residuals_contribution)
        residuals_line = ax.plot(
            any_given_fit.interpolant_incident_energy,
            any_given_fit.residuals,
            label=residuals_label)

        spectrum_points = ax.plot(
            any_given_fit.interpolant_incident_energy,
            any_given_fit.unknown_spectrum_b,
            '.',
            label=spectrum.file_name,
            alpha=0.5)

        # add fits in descending order of reference contribution
        reference_label_list = []
        reference_contributions_percent_sr.sort_values(ascending=False, inplace=True)
        sort_ndx = reversed(any_given_fit.reference_spectra_coef_x.argsort())
        ys = any_given_fit.reference_spectra_coef_x * any_given_fit.reference_spectra_A_df
        log.info('plotting reference components')
        log.info(reference_contributions_percent_sr.head())
        reference_contributions_percent_sr.sort_values(ascending=False)
        for name, value in reference_contributions_percent_sr.items():
            log.info('reference component {} {}'.format(name, value))
            reference_label = contribution_format_str.format(name, value)
            reference_label_list.append(reference_label)

        reference_line_list = ax.stackplot(ys.index, *[ys.iloc[:, i] for i in sort_ndx], labels=reference_label_list)

        ax.set_xlabel('eV')
        ax.set_ylabel('normalized absorbance')
        ax.legend(
            # these arguments are documented but this does not seem to work
            [*spectrum_points, *reference_line_list, *residuals_line],
            [spectrum.file_name, *reference_label_list, residuals_label],
            prop=dict(family='Monospace', size=7))

        return f

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

    def bokeh_nss_path(self, spectrum, fit_results, output_fp):
        log = logging.getLogger(name=self.__class__.__name__)
        bokeh.plotting.output_file(output_fp)

        hover = bokeh.models.HoverTool(tooltips=[
            ("index", "$index"),
            ("NSS", "$y"),
            ("desc", "@desc"),
        ])
        p = bokeh.plotting.figure(
            plot_width=400, plot_height=400,
            tools=['reset', hover, 'box_zoom'],
            title=spectrum.file_name,
            x_axis_label='index',
            y_axis_label='NSS')

        for component_count in fit_results.component_count_fit_table.keys():
            sorted_fits = fit_results.component_count_fit_table[component_count][:20]
            source = bokeh.plotting.ColumnDataSource(data=dict(
                x=range(len(sorted_fits)),
                nss=[spectrum_fit.nss for spectrum_fit in sorted_fits],
                desc=[str(spectrum_fit) for spectrum_fit in sorted_fits]
            ))

            p.circle('x', 'nss', legend='{}-component'.format(component_count), size=5, source=source)

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
