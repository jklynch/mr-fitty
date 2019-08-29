"""
The MIT License (MIT)

Copyright (c) 2015-2019 Joshua Lynch, Sarah Nicholas

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
from concurrent.futures import ProcessPoolExecutor, as_completed
import datetime
import itertools
import logging
from operator import attrgetter
import os.path
import time

import bokeh.io
import bokeh.models.layouts
import bokeh.plotting

import matplotlib

matplotlib.use("pdf", warn=False, force=True)

import matplotlib.pyplot as plt

from matplotlib.backends.backend_pdf import PdfPages

import numpy as np

import pandas as pd

import scipy.cluster.hierarchy as hc
from scipy.spatial.distance import pdist

from sklearn.utils import shuffle

from mrfitty.base import (
    InterpolatedSpectrumSet,
    InterpolatedReferenceSpectraSet,
    SpectrumFit,
)


def grouper(iterable, n, fillvalue=None):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx"
    args = [iter(iterable)] * n
    return itertools.zip_longest(*args, fillvalue=fillvalue)


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
    def __init__(
        self,
        ls,
        reference_spectrum_list,
        unknown_spectrum_list,
        energy_range_builder,
        best_fits_plot_limit,
        component_count_range=range(4),
    ):
        self.ls = ls
        self.reference_spectrum_list = reference_spectrum_list
        self.unknown_spectrum_list = unknown_spectrum_list
        self.energy_range_builder = energy_range_builder
        self.best_fits_plot_limit = best_fits_plot_limit
        self.component_count_range = component_count_range

        self.fit_table = collections.OrderedDict()

    def fit_all(self, plots_pdf_dp):
        log = logging.getLogger(name="fit_all")

        os.makedirs(plots_pdf_dp, exist_ok=True)

        with ProcessPoolExecutor(max_workers=4) as executor:
            future_to_unknown_spectrum = dict()
            for unknown_spectrum in sorted(
                self.unknown_spectrum_list, key=lambda s: s.file_name
            ):

                unknown_spectrum_future = executor.submit(
                    self.fit_and_plot, unknown_spectrum, plots_pdf_dp
                )
                future_to_unknown_spectrum[unknown_spectrum_future] = unknown_spectrum

            for unknown_spectrum_future in as_completed(future_to_unknown_spectrum):
                unknown_spectrum = future_to_unknown_spectrum[unknown_spectrum_future]
                log.info("completed %s fit", unknown_spectrum.file_name)

                fit_results = unknown_spectrum_future.result()
                self.fit_table[unknown_spectrum] = fit_results

        return self.fit_table

    def fit_and_plot(self, unknown_spectrum, plots_pdf_dp):
        log = logging.getLogger(
            name="fit_and_plot:{}".format(unknown_spectrum.file_name)
        )

        log.debug("fitting %s", unknown_spectrum.file_name)
        t0 = time.time()
        best_fit, fit_table = self.fit(unknown_spectrum)
        t1 = time.time()
        log.info("fit %s in %5.3fs", unknown_spectrum.file_name, t1 - t0)
        fit_results = CombinationFitResults(
            spectrum=unknown_spectrum,
            best_fit=best_fit,
            component_count_fit_table=fit_table,
        )

        file_base_name, _ = os.path.splitext(
            os.path.basename(unknown_spectrum.file_name)
        )
        plots_pdf_fp = os.path.join(plots_pdf_dp, file_base_name + "_fit.pdf")
        with PdfPages(plots_pdf_fp) as plot_file:
            log.info("writing plots file {}".format(plots_pdf_dp))
            # create plot
            log.info("plotting fit for %s", unknown_spectrum.file_name)

            f_list = self.plot_top_fits(
                spectrum=unknown_spectrum, fit_results=fit_results
            )
            for f in f_list:
                plot_file.savefig(f)
                plt.close(f)

            f = self.plot_fit(
                spectrum=unknown_spectrum,
                any_given_fit=fit_results.best_fit,
                title="Best Fit",
            )
            plot_file.savefig(f)
            plt.close(f)

            f = self.plot_stacked_fit(
                spectrum=unknown_spectrum,
                any_given_fit=fit_results.best_fit,
                title="Best Fit",
            )
            plot_file.savefig(f)
            plt.close(f)

            clustering_parameters = {
                "linkage_method": "complete",
                "pdist_metric": "correlation",
            }

            # use these for reference tree plots
            interpolation_energy_range, _ = self.energy_range_builder.build_range(
                unknown_spectrum=unknown_spectrum,
                reference_spectrum_seq=self.reference_spectrum_list,
            )
            interpolated_reference_set_df = InterpolatedSpectrumSet.get_interpolated_spectrum_set_df(
                energy_range=interpolation_energy_range,
                spectrum_set=set(self.reference_spectrum_list),
            )

            reference_spectra_linkage, cutoff_distance = self.cluster_reference_spectra(
                interpolated_reference_set_df, **clustering_parameters
            )

            g = self.plot_reference_tree(
                linkage_distance_variable_by_sample=reference_spectra_linkage,
                reference_df=interpolated_reference_set_df,
                cutoff_distance=cutoff_distance,
                title="Best Fit",
                **clustering_parameters
            )

            reference_spectra_names = tuple(
                [r.file_name for r in fit_results.best_fit.reference_spectra_seq]
            )

            leaf_colors = plt.cm.get_cmap("Accent", 2)
            for i, leaf_label in enumerate(plt.gca().get_ymajorticklabels()):
                if leaf_label.get_text() in reference_spectra_names:
                    leaf_label.set_color(leaf_colors(1))
                else:
                    leaf_label.set_color(leaf_colors(0))

            plot_file.savefig(g)
            plt.close(g)

            ordinal_list = (
                "1st",
                "2nd",
                "3rd",
                "4th",
                "5th",
                "6th",
                "7th",
                "8th",
                "9th",
                "10th",
            )

            # plot the best n-component fit
            for n in sorted(fit_table.keys()):
                log.info(
                    "plotting %d-component fit for %s", n, unknown_spectrum.file_name
                )
                n_component_fit_results = fit_table[n]
                # here only plot the best fit for each component count

                for i, fit in enumerate(n_component_fit_results):
                    if i < self.best_fits_plot_limit:
                        title = "{} Best {}-Component Fit".format(ordinal_list[i], n)

                        f = self.plot_fit(
                            spectrum=unknown_spectrum, any_given_fit=fit, title=title
                        )
                        plot_file.savefig(f)

                        g = self.plot_reference_tree(
                            linkage_distance_variable_by_sample=reference_spectra_linkage,
                            reference_df=interpolated_reference_set_df,
                            cutoff_distance=cutoff_distance,
                            title=title,
                            **clustering_parameters
                        )

                        reference_spectra_names = tuple(
                            [r.file_name for r in fit.reference_spectra_seq]
                        )

                        leaf_colors = plt.cm.get_cmap("Accent", 2)
                        for i, leaf_label in enumerate(
                            plt.gca().get_ymajorticklabels()
                        ):
                            if leaf_label.get_text() in reference_spectra_names:
                                leaf_label.set_color(leaf_colors(1))
                            else:
                                leaf_label.set_color(leaf_colors(0))

                        plot_file.savefig(g)
                        plt.close(g)

                    else:
                        break

        return fit_results

    # tried to speed up mrfitty by distributing the work in this function
    # there was no speedup
    # apparently this is not where a lot of time is spent
    def fit(self, unknown_spectrum):
        log = logging.getLogger(name=unknown_spectrum.file_name)
        log.info("fitting unknown spectrum %s", unknown_spectrum.file_name)
        interpolated_reference_spectra = InterpolatedReferenceSpectraSet(
            unknown_spectrum=unknown_spectrum,
            reference_set=self.reference_spectrum_list,
        )
        # fit all combinations of reference_spectra
        # all_counts_spectrum_fit_table looks like this:
        #   { 1: [...list of 1-component fits sorted by NSS...],
        #     2: [...list of 2-component fits sorted by NSS...],
        #     ...
        #   }
        all_counts_spectrum_fit_table = collections.defaultdict(list)
        reference_combination_grouper = grouper(
            self.reference_combination_iter(self.component_count_range), n=1000
        )
        ##with ProcessPoolExecutor(max_workers=1) as executor:
        ##    log.info('ProcessPoolExecutor max_workers=%d', executor._max_workers)
        ##    future_to_ref_combination_group = dict()
        for reference_combination_group in reference_combination_grouper:
            log.debug(
                "fitting group of %d reference combinations",
                len(reference_combination_group),
            )
            ##reference_combination_group_future = executor.submit(
            ##    self.do_some_fits, unknown_spectrum, interpolated_reference_spectra, reference_combination_group
            ##)
            ##future_to_ref_combination_group[reference_combination_group_future] = reference_combination_group

            ##for reference_combination_group_future in as_completed(future_to_ref_combination_group):
            fits, failed_fits = self.do_some_fits(
                unknown_spectrum=unknown_spectrum,
                interpolated_reference_spectra=interpolated_reference_spectra,
                reference_spectra_combinations=reference_combination_group,
            )
            ##    fits, failed_fits = reference_combination_group_future.result()

            log.debug("%d successful fits", len(fits))
            # append new fits to the appropriate lists
            # but do not sort yet
            for fit in fits:
                reference_count = len(fit.reference_spectra_seq)
                spectrum_fit_list = all_counts_spectrum_fit_table[reference_count]
                spectrum_fit_list.append(fit)

            # now sort and trim each list to the best 100 fits
            for (
                reference_count,
                spectrum_fit_list,
            ) in all_counts_spectrum_fit_table.items():
                log.debug(
                    "sorting %d-component fit list with %d fits",
                    reference_count,
                    len(spectrum_fit_list),
                )
                spectrum_fit_list.sort(key=attrgetter("nss"))
                # when there are many reference spectra the list of fits can get extremely long
                # and eat up all of memory
                # so keep only the top 100 fits for each component count
                if len(spectrum_fit_list) > 100:
                    log.debug(
                        "trimming %d-component fit list with %d fits",
                        reference_count,
                        len(spectrum_fit_list),
                    )
                    all_counts_spectrum_fit_table[reference_count] = spectrum_fit_list[
                        :100
                    ]

            log.debug("%d failed fits", len(failed_fits))
            # for failed_fit in failed_fits:
            #    log.debug(failed_fit)

            # try:
            #     spectrum_fit = self.fit_references_to_unknown(
            #         interpolated_reference_spectra=interpolated_reference_spectra,
            #         reference_spectra_subset=reference_spectra_combination)
            #     reference_count = len(reference_spectra_combination)
            #     spectrum_fit_list = all_counts_spectrum_fit_table[reference_count]
            #     spectrum_fit_list.append(spectrum_fit)
            #     spectrum_fit_list.sort(key=attrgetter('nss'))
            #     # when there are many reference spectra the list of fits can get extremely long
            #     # and eat up all the memory
            #     # keep only the top 100 fits for each component count
            #     if len(spectrum_fit_list) > 100:
            #         spectrum_fit_list.pop()
            # except FitFailed as ff:
            #     # this is a common occurrence when using ordinary linear regression
            #     # it is not an 'error' just something that happens and needs to be handled
            #     log.debug(
            #         'failed to fit unknown "%s" to references\n\t%s',
            #         unknown_spectrum.file_name,
            #         '\n\t'.join([r.file_name for r in reference_spectra_combination]))
            #     log.debug(ff)

        best_fit = self.choose_best_component_count(all_counts_spectrum_fit_table)
        return best_fit, all_counts_spectrum_fit_table

    def do_some_fits(
        self,
        unknown_spectrum,
        interpolated_reference_spectra,
        reference_spectra_combinations,
    ):
        log = logging.getLogger(name=unknown_spectrum.file_name)

        fits = []
        failed_fits = []
        log.debug(
            "do_some_fits for %d reference combinations",
            len(reference_spectra_combinations),
        )
        for reference_spectra_combination in reference_spectra_combinations:
            log.debug("fitting to reference_spectra %s", reference_spectra_combination)

            if reference_spectra_combination is None:
                pass
            else:
                try:
                    spectrum_fit = self.fit_references_to_unknown(
                        interpolated_reference_spectra=interpolated_reference_spectra,
                        reference_spectra_subset=reference_spectra_combination,
                    )
                    fits.append(spectrum_fit)
                except FitFailed as ff:
                    # this is a common occurrence when using ordinary linear regression
                    # it is not an 'error' just something that happens and needs to be handled
                    msg = 'failed to fit unknown "{}" to references\n\t{}'.format(
                        unknown_spectrum.file_name,
                        "\n\t".join(
                            [r.file_name for r in reference_spectra_combination]
                        ),
                    )
                    failed_fits.append(msg)

        log.debug("returning %d fits, %d failed fits", len(fits), len(failed_fits))
        return fits, failed_fits

    def reference_combination_iter(self, component_count_range):
        for component_count in component_count_range:
            for reference_spectra_combination in itertools.combinations(
                self.reference_spectrum_list, component_count
            ):
                yield reference_spectra_combination

    def fit_references_to_unknown(
        self, interpolated_reference_spectra, reference_spectra_subset
    ):
        interpolated_data = interpolated_reference_spectra.get_reference_subset_and_unknown_df(
            reference_list=reference_spectra_subset,
            energy_range_builder=self.energy_range_builder,
        )

        interpolated_reference_spectra_subset_df = interpolated_data[
            "reference_subset_df"
        ]
        unknown_spectrum_df = interpolated_data["unknown_subset_df"]

        lm = self.ls()
        lm.fit(
            interpolated_reference_spectra_subset_df.values,
            unknown_spectrum_df.norm.values,
        )
        if any(lm.coef_ < 0.0):
            msg = "negative coefficients while fitting:\n{}".format(lm.coef_)
            raise FitFailed(msg)
        else:
            reference_spectra_coef_x = lm.coef_

            spectrum_fit = SpectrumFit(
                interpolant_incident_energy=interpolated_reference_spectra_subset_df.index,
                reference_spectra_A_df=interpolated_reference_spectra_subset_df,
                unknown_spectrum=interpolated_data["unknown_subset_spectrum"],
                reference_spectra_seq=reference_spectra_subset,
                reference_spectra_coef_x=reference_spectra_coef_x,
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
            best_fit_for_component_count = all_counts_spectrum_fit_table[
                component_count
            ][0]
            improvement = (
                previous_nss - best_fit_for_component_count.nss
            ) / previous_nss
            log.debug(
                "improvement: {:5.3f} for {}".format(
                    improvement, best_fit_for_component_count
                )
            )
            if improvement < 0.10:
                break
            else:
                best_fit = best_fit_for_component_count
                previous_nss = best_fit.nss
        log.debug("best fit: {}".format(best_fit))
        return best_fit

    def write_table(self, table_file_path):
        """
        sample name, residual, reference 1, fraction 1, reference 2, fraction 2, ...
        :param table_file_path:
        :return:
        """
        log = logging.getLogger(name=self.__class__.__name__)

        table_file_dir_path, _ = os.path.split(table_file_path)
        os.makedirs(table_file_dir_path, exist_ok=True)

        with open(table_file_path, "wt") as table_file:
            table_file.write(
                "spectrum\tNSS\tresidual percent\treference 1\tpercent 1\treference 2\tpercent 2\treference 3\tpercent 3\n"
            )
            for spectrum, fit_results in self.fit_table.items():
                table_file.write(spectrum.file_name)
                table_file.write("\t")
                table_file.write("{:8.5f}\t".format(fit_results.best_fit.nss))
                table_file.write(
                    "{:5.3f}".format(fit_results.best_fit.residuals_contribution)
                )
                for (
                    ref_name,
                    ref_pct,
                ) in fit_results.best_fit.reference_contribution_percent_sr.sort_values(
                    ascending=False
                ).items():
                    table_file.write("\t")
                    table_file.write(ref_name)
                    table_file.write("\t{:5.3f}".format(ref_pct))
                table_file.write("\n")

    def draw_plots_matplotlib(self, plots_pdf_file_path):
        log = logging.getLogger(name=self.__class__.__name__)
        with PdfPages(plots_pdf_file_path) as plot_file:
            log.info("writing plots file %s", plots_pdf_file_path)
            for spectrum, fit_results in self.fit_table.items():
                log.debug("plotting fit for %s", spectrum.file_name)

                f = self.plot_fit(spectrum, fit_results.best_fit)
                plot_file.savefig(f)

                # plot the best n-component fit
                # for n_fit_list in fit_results:
                # plot the best 2-component fit

    @staticmethod
    def add_date_time_footer(ax):
        ax.annotate(
            datetime.datetime.now().isoformat(),
            xy=(0.025, 0.025),
            xycoords="figure fraction",
            horizontalalignment="left",
            verticalalignment="top",
            fontsize=4,
        )

    def plot_top_fits(self, spectrum, fit_results):
        log = logging.getLogger(name=self.__class__.__name__)

        figure_list = []
        for i, component_count in enumerate(
            fit_results.component_count_fit_table.keys()
        ):
            f, ax = plt.subplots()
            f.suptitle(spectrum.file_name + "\n" + "Fit Path")

            sorted_fits = fit_results.component_count_fit_table[component_count][:10]
            ax.scatter(
                y=range(len(sorted_fits)),
                x=[spectrum_fit.nss for spectrum_fit in sorted_fits],
            )
            ax.set_title("{} component(s)".format(component_count))
            ax.set_xlabel("NSS")
            ax.set_ylabel("order")

            self.add_date_time_footer(ax)

            f.tight_layout()
            figure_list.append(f)

        return figure_list

    def get_fit_quality_score_text(self, any_given_fit):
        return ["MSE: {:8.5f}".format(any_given_fit.nss)]

    def plot_fit(self, spectrum, any_given_fit, title):
        log = logging.getLogger(name=self.__class__.__name__ + ":" + spectrum.file_name)

        f, ax = plt.subplots()
        ##ax.set_title(spectrum.file_name + '\n' + title + '\n' + self.get_fit_quality_score_text(any_given_fit))
        ax.set_title(spectrum.file_name + "\n" + title)
        ##log.info(any_given_fit.fit_spectrum_b.shape)

        reference_contributions_percent_sr = (
            any_given_fit.get_reference_contributions_sr()
        )
        reference_only_contributions_percent_sr = (
            any_given_fit.get_reference_only_contributions_sr()
        )
        longest_name_len = max(
            [len(name) for name in reference_contributions_percent_sr.index]
            + [len(spectrum.file_name)]
        )
        # the format string should look like '{:N}{:5.2f} ({:5.2f})' where N is the length of the longest reference name
        reference_contribution_format_str = (
            "{:" + str(longest_name_len + 4) + "}{:5.2f} ({:5.2f})"
        )
        residuals_contribution_format_str = (
            "{:" + str(longest_name_len + 4) + "}{:5.2f}"
        )

        # add fits in descending order of reference contribution
        reference_line_list = []
        reference_label_list = []
        reference_contributions_percent_sr.sort_values(ascending=False, inplace=True)
        reference_only_contributions_percent_sr.sort_values(
            ascending=False, inplace=True
        )
        log.debug("plotting reference components")
        log.debug(reference_contributions_percent_sr.head())
        for (ref_name, ref_contrib), (ref_only_name, ref_only_contrib) in zip(
            reference_contributions_percent_sr.items(),
            reference_only_contributions_percent_sr.items(),
        ):
            log.debug("reference contribution %s %5.2f", ref_name, ref_contrib)
            log.debug(
                "reference-only contribution %s %5.2f", ref_only_name, ref_only_contrib
            )
            reference_label = reference_contribution_format_str.format(
                ref_name, ref_contrib, ref_only_contrib
            )
            reference_label_list.append(reference_label)

            # plot once for each reference just to build the legend
            # ax.plot returns a list
            reference_line_list.extend(
                ax.plot(
                    any_given_fit.interpolant_incident_energy,
                    any_given_fit.fit_spectrum_b,
                    label=reference_label,
                    color="w",
                    alpha=0.0,
                )
            )

        ##log.info(any_given_fit.residuals.shape)
        residuals_label = residuals_contribution_format_str.format(
            "residuals", any_given_fit.residuals_contribution
        )
        residuals_line = ax.plot(
            any_given_fit.interpolant_incident_energy,
            any_given_fit.residuals,
            ".",
            label=residuals_label,
            alpha=0.5,
        )

        fit_line_label = "fit"
        fit_line = ax.plot(
            any_given_fit.interpolant_incident_energy,
            any_given_fit.fit_spectrum_b,
            label=fit_line_label,
        )

        spectrum_points = ax.plot(
            any_given_fit.interpolant_incident_energy,
            any_given_fit.unknown_spectrum_b,
            ".",
            label=spectrum.file_name,
            alpha=0.5,
        )

        # add some fake lines to create some special legend entries
        fit_quality_lines = []
        fit_quality_labels = self.get_fit_quality_score_text(
            any_given_fit=any_given_fit
        )
        for fit_quality_label in fit_quality_labels:
            fit_quality_lines.extend(
                ax.plot([], [], label=fit_quality_label, color="w", alpha=0.0)
            )

        ax.set_xlabel("eV")
        ax.set_ylabel("normalized absorbance")
        # TODO: make these configurable
        legend_location = "best"
        legend_font_size = 6
        legend = ax.legend(
            [
                *reference_line_list,
                *spectrum_points,
                *residuals_line,
                *fit_line,
                *fit_quality_lines,
            ],
            [
                *reference_label_list,
                spectrum.file_name,
                residuals_label,
                fit_line_label,
                *fit_quality_labels,
            ],
            loc=legend_location,
            prop=dict(family="Monospace", size=legend_font_size),
        )

        # ax.text(
        #     0.75,
        #     0.35,
        #     self.get_fit_quality_score_text(any_given_fit=any_given_fit),
        #     transform=ax.transAxes,
        #     fontsize=6,
        #     verticalalignment='top',
        #     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        # )

        self.add_date_time_footer(ax)

        plt.tight_layout()

        return f

    def plot_stacked_fit(self, spectrum, any_given_fit, title):
        log = logging.getLogger(name=self.__class__.__name__)

        f, ax = plt.subplots()
        ax.set_title(spectrum.file_name + "\n" + title)
        ##log.info(any_given_fit.fit_spectrum_b.shape)

        reference_contributions_percent_sr = (
            any_given_fit.get_reference_contributions_sr()
        )
        longest_name_len = max(
            [len(name) for name in reference_contributions_percent_sr.index]
            + [len(spectrum.file_name)]
        )
        # the format string should look like '{:N}{:5.2f}' where N is the length of the longest reference name
        contribution_format_str = "{:" + str(longest_name_len + 4) + "}{:5.2f}"

        ##log.info(any_given_fit.residuals.shape)
        residuals_label = contribution_format_str.format(
            "residuals", any_given_fit.residuals_contribution
        )
        residuals_line = ax.plot(
            any_given_fit.interpolant_incident_energy,
            any_given_fit.residuals,
            ".",
            label=residuals_label,
            alpha=0.5,
        )

        spectrum_points = ax.plot(
            any_given_fit.interpolant_incident_energy,
            any_given_fit.unknown_spectrum_b,
            ".",
            label=spectrum.file_name,
            alpha=0.5,
        )

        # add fits in descending order of reference contribution
        reference_label_list = []
        reference_contributions_percent_sr.sort_values(ascending=False, inplace=True)
        sort_ndx = reversed(any_given_fit.reference_spectra_coef_x.argsort())
        ys = (
            any_given_fit.reference_spectra_coef_x
            * any_given_fit.reference_spectra_A_df
        )
        log.debug("plotting reference components")
        log.debug(reference_contributions_percent_sr.head())
        reference_contributions_percent_sr.sort_values(ascending=False)
        for name, value in reference_contributions_percent_sr.items():
            log.debug("reference component {} {}".format(name, value))
            reference_label = contribution_format_str.format(name, value)
            reference_label_list.append(reference_label)

        reference_line_list = ax.stackplot(
            ys.index, *[ys.iloc[:, i] for i in sort_ndx], labels=reference_label_list
        )

        # add some fake lines to create some special legend entries
        fit_quality_lines = []
        fit_quality_labels = self.get_fit_quality_score_text(
            any_given_fit=any_given_fit
        )
        for fit_quality_label in fit_quality_labels:
            fit_quality_lines.extend(
                ax.plot([], [], label=fit_quality_label, color="w", alpha=0.0)
            )

        ax.set_xlabel("eV")
        ax.set_ylabel("normalized absorbance")
        ax.legend(
            # these arguments are documented but this does not seem to work
            [
                *spectrum_points,
                *reference_line_list,
                *residuals_line,
                *fit_quality_lines,
            ],
            [
                spectrum.file_name,
                *reference_label_list,
                residuals_label,
                *fit_quality_labels,
            ],
            prop=dict(family="Monospace", size=7),
        )

        self.add_date_time_footer(ax)

        plt.tight_layout()

        return f

    @staticmethod
    def permute_row_elements(df):
        for i in range(df.shape[0]):
            df.values[i, :] = shuffle(df.values[i, :])
        return df

    def cluster_reference_spectra(
        self, reference_df, pdist_metric="correlation", linkage_method="complete"
    ):
        log = logging.getLogger(name=self.__class__.__name__)

        distance_for_sample_pairs = pdist(
            X=np.transpose(reference_df.values), metric=pdist_metric
        )

        # plt.figure()
        # plt.title(title)
        # plt.hist(distance_for_sample_pairs)
        # plt.xlabel('{} distance'.format(pdist_metric))
        # plt.ylabel('{} pairs'.format(variable_by_sample_df.shape))
        # plt.show()

        resample_count = 1000
        expected_distance_list = []
        for i in range(resample_count):
            # permute the elements of each row of variable_by_sample_df
            p_variable_by_sample_df = self.permute_row_elements(reference_df.copy())
            p_distance_for_sample_pairs = pdist(
                X=np.transpose(p_variable_by_sample_df.values), metric=pdist_metric
            )
            p_linkage_distance_variable_by_sample = hc.linkage(
                y=p_distance_for_sample_pairs, method=linkage_method
            )
            p_dendrogram = hc.dendrogram(
                Z=p_linkage_distance_variable_by_sample, no_plot=True
            )
            expected_distance_list.extend(
                [d for (_, _, d, _) in p_dendrogram["dcoord"]]
            )

        p = 95.0
        alpha = 1.0 - p / 100.0
        cutoff_distance = np.percentile(expected_distance_list, q=p)
        ##print('cutoff distance is {}'.format(cutoff_distance))

        # plt.figure()
        # plt.hist(expected_distance_list)
        # plt.title('dendrogram distance null distribution')
        # plt.show()

        linkage_distance_variable_by_sample = hc.linkage(
            y=distance_for_sample_pairs, method=linkage_method
        )

        return linkage_distance_variable_by_sample, cutoff_distance

    def plot_reference_tree(
        self,
        linkage_distance_variable_by_sample,
        reference_df,
        cutoff_distance,
        title,
        pdist_metric,
        linkage_method,
    ):
        log = logging.getLogger(name=self.__class__.__name__)

        f, ax = plt.subplots()
        dendrogram = hc.dendrogram(
            ax=ax,
            Z=linkage_distance_variable_by_sample,
            orientation="left",
            leaf_font_size=8,
            labels=reference_df.columns,
        )

        # leaf_colors = plt.cm.get_cmap("Accent", 2)
        # for i, leaf_label in enumerate(plt.gca().get_ymajorticklabels()):
        #    leaf_label.set_color(leaf_colors(i % 2))

        icoords = tuple([i for i in itertools.chain(dendrogram["icoord"])])
        ax.vlines(cutoff_distance, ymin=np.min(icoords), ymax=np.max(icoords))
        ax.set_title("{}\n{} linkage".format(title, linkage_method))
        ax.set_xlabel("{} distance".format(pdist_metric))

        self.add_date_time_footer(ax)

        plt.tight_layout()

        return f

    def plot_prediction_errors(self, spectrum, fit):
        log = logging.getLogger(name=self.__class__.__name__)

        f, ax = plt.subplots()
        ax.set_title(spectrum.file_name + "\n" + "Prediction Errors")

        # prediction_errors = list(itertools.chain.from_iterable(fit.prediction_errors))
        ax.hist(fit.prediction_errors)
        ax.set_xlabel("Prediction Error")
        ax.set_ylabel("Count")

        self.add_date_time_footer(ax)

        return f

    def draw_plots_bokeh(self, plots_html_file_path):
        log = logging.getLogger(name=self.__class__.__name__)
        bokeh.io.output_file(plots_html_file_path)
        # with PdfPages(plots_pdf_file_path) as plot_file:
        log.info("writing plots file {}".format(plots_html_file_path))
        plot_list = []
        for spectrum, fit_results in self.fit_table.items():
            log.debug("plotting fit for {}".format(spectrum.file_name))
            # f, ax = plt.subplots()
            f = bokeh.plotting.figure(title="{}\n???".format(spectrum.file_name))
            plot_list.append(f)

            ##log.debug(fit_results.best_fit.fit_spectrum_b.shape)
            # ax.plot(fit_results.best_fit.interpolant_incident_energy, fit_results.best_fit.fit_spectrum_b)
            f.line(
                fit_results.best_fit.interpolant_incident_energy,
                fit_results.best_fit.fit_spectrum_b,
                line_width=2,
            )
            ##log.debug(fit_results.best_fit.residuals.shape)
            # ax.plot(fit_results.best_fit.interpolant_incident_energy, fit_results.best_fit.unknown_spectrum_b, '.')
            f.circle(
                fit_results.best_fit.interpolant_incident_energy,
                fit_results.best_fit.unknown_spectrum_b,
            )
            # ax.plot(fit_results.best_fit.interpolant_incident_energy, fit_results.best_fit.residuals)
            f.line(
                fit_results.best_fit.interpolant_incident_energy,
                fit_results.best_fit.residuals,
            )
            # log.info('fit_results.best_fit.interpolant_incident_energy:\n{}'.format(
            #    fit_results.best_fit.interpolant_incident_energy)
            # )
            # ax.vlines(x=[
            #        fit_results.best_fit.interpolant_incident_energy.iloc[0],
            #        fit_results.best_fit.interpolant_incident_energy.iloc[-1]
            #    ],
            #    ymin=fit_results.best_fit.unknown_spectrum_b.min(),
            #    ymax=fit_results.best_fit.unknown_spectrum_b.max()
            # )
            # plot_file.savefig(f)
        p = bokeh.models.layouts.Column(*plot_list)
        bokeh.io.save(p)

    def bokeh_nss_path(self, spectrum, fit_results, output_fp):
        log = logging.getLogger(name=self.__class__.__name__)
        bokeh.plotting.output_file(output_fp)

        hover = bokeh.models.HoverTool(
            tooltips=[("index", "$index"), ("NSS", "$y"), ("desc", "@desc")]
        )
        p = bokeh.plotting.figure(
            plot_width=400,
            plot_height=400,
            tools=["reset", hover, "box_zoom"],
            title=spectrum.file_name,
            x_axis_label="index",
            y_axis_label="NSS",
        )

        for component_count in fit_results.component_count_fit_table.keys():
            sorted_fits = fit_results.component_count_fit_table[component_count][:20]
            source = bokeh.plotting.ColumnDataSource(
                data=dict(
                    x=range(len(sorted_fits)),
                    nss=[spectrum_fit.nss for spectrum_fit in sorted_fits],
                    desc=[str(spectrum_fit) for spectrum_fit in sorted_fits],
                )
            )

            p.circle(
                "x",
                "nss",
                legend="{}-component".format(component_count),
                size=5,
                source=source,
            )

        bokeh.io.save(p)

    def write_best_fit_arrays(self, best_fit_dir_path):
        log = logging.getLogger(name=self.__class__.__name__)
        for spectrum, fit_results in self.fit_table.items():
            file_base_name, file_name_ext = os.path.splitext(spectrum.file_name)
            fit_file_path = os.path.join(best_fit_dir_path, file_base_name + "_fit.txt")
            log.info("writing best fit to {}".format(fit_file_path))

            fit_df = pd.DataFrame(
                {
                    "energy": fit_results.best_fit.interpolant_incident_energy,
                    "spectrum": fit_results.best_fit.unknown_spectrum_b,
                    "fit": fit_results.best_fit.fit_spectrum_b,
                    "residual": fit_results.best_fit.residuals,
                }
            )
            fit_df.to_csv(fit_file_path, sep="\t", float_format="%8.4f", index=False)
