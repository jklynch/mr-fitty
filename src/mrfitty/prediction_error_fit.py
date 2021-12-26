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
from collections import defaultdict
import logging

import matplotlib.pyplot as plt
import numpy as np
import scikits.bootstrap
import sklearn.model_selection

from mrfitty.base import AdaptiveEnergyRangeBuilder
from mrfitty.combination_fit import AllCombinationFitTask
from mrfitty.linear_model import NonNegativeLinearRegression
from mrfitty.plot import (
    prediction_error_box_plots,
    prediction_error_confidence_interval_plot,
    best_fit_for_component_count_box_plots,
)


class PredictionErrorFitTask(AllCombinationFitTask):
    def __init__(
        self,
        reference_spectrum_list,
        unknown_spectrum_list,
        ls=NonNegativeLinearRegression,
        energy_range_builder=AdaptiveEnergyRangeBuilder(),
        component_count_range=range(1, 4),
        best_fits_plot_limit=3,
        bootstrap_count=1000,
    ):
        super().__init__(
            ls=ls,
            reference_spectrum_list=reference_spectrum_list,
            unknown_spectrum_list=unknown_spectrum_list,
            energy_range_builder=energy_range_builder,
            best_fits_plot_limit=best_fits_plot_limit,
            component_count_range=component_count_range,
        )
        self.bootstrap_count = bootstrap_count

    def get_fit_quality_score_text(self, any_given_fit):
        return [
            "MSPE 95% ci of median: {:.5f} <-- {:.5f} --> {:.5f}".format(
                any_given_fit.median_C_p_ci_lo,
                any_given_fit.median_C_p,
                any_given_fit.median_C_p_ci_hi,
            ),
            "MSE: {:<8.5f}".format(any_given_fit.nss),
        ]

    def choose_best_component_count(self, all_counts_spectrum_fit_table):
        """
        Calculate the prediction error statistics for top fits.

        Parameters
        ----------
        all_counts_spectrum_fit_table (dict)
          dictionary with component count keys, sorted list of spectrum fit list values

        Returns
        -------
        best_fit (SpectrumFit)
            the fit having lowest 95% confidence interval of median prediction error AND lowest number
            of reference components
        """
        component_counts = list(all_counts_spectrum_fit_table)
        a_fit = all_counts_spectrum_fit_table[component_counts[0]][0]
        log = logging.getLogger(__name__ + ":" + a_fit.unknown_spectrum.file_name)

        log.debug(
            "choosing best component count from %s", all_counts_spectrum_fit_table
        )
        component_count_to_median_cp = {
            component_count: np.Inf
            for component_count in all_counts_spectrum_fit_table.keys()
        }
        component_count_to_median_cp_ci_lo_hi = {
            component_count: (np.Inf, np.Inf)
            for component_count in all_counts_spectrum_fit_table.keys()
        }

        all_counts_spectrum_fit_pe_table = defaultdict(list)
        for component_count_i in sorted(all_counts_spectrum_fit_table.keys()):
            # calculate C_p for the first ? fits with component_count_i
            log.debug(
                "calculating CI of median C_p for %d component(s)", component_count_i
            )

            sorted_fits_for_i_components = sorted(
                all_counts_spectrum_fit_table[component_count_i],
                key=lambda fit: fit.nss,
            )

            for fit_j in sorted_fits_for_i_components[:20]:
                prediction_errors, _ = self.calculate_prediction_error_list(
                    fit_j, n_splits=self.bootstrap_count
                )

                fit_j.mean_C_p = np.mean(prediction_errors)
                mean_ci_lo, mean_ci_hi = scikits.bootstrap.ci(
                    data=prediction_errors, statfunction=np.mean
                )
                fit_j.mean_C_p_ci_lo = mean_ci_lo
                fit_j.mean_C_p_ci_hi = mean_ci_hi

                fit_j.median_C_p = np.median(prediction_errors)
                median_ci_lo, median_ci_hi = scikits.bootstrap.ci(
                    data=prediction_errors, statfunction=np.median
                )
                fit_j.median_C_p_ci_lo = median_ci_lo
                fit_j.median_C_p_ci_hi = median_ci_hi

                fit_j.prediction_errors = prediction_errors

                all_counts_spectrum_fit_pe_table[component_count_i].append(fit_j)

            all_counts_spectrum_fit_pe_table[component_count_i] = sorted(
                all_counts_spectrum_fit_pe_table[component_count_i],
                key=lambda fit: (
                    fit.median_C_p,
                    fit.median_C_p_ci_lo,
                    fit.median_C_p_ci_hi,
                ),
            )
            # for each fit find all fits with overlapping ci
            # 4 cases:
            #
            #   <-- k -->                                   j.lo > k.hi (keep checking)
            #              <-- j -->
            #
            #         <-- k -->                             j.lo <= k.hi <= j.hi
            #              <-- j -->
            #
            #             <--  k  -->                       j.lo > k.lo and j.hi < k.hi
            #              <-- j -->
            #
            #               <- k ->                         j.lo <= k.lo and j.hi > k.hi
            #              <-- j -->
            #
            #                    <-- k -->                  j.lo <= k.lo <= j.hi
            #              <-- j -->
            #
            #                         <-- k -->             j.hi < k.lo (stop checking)
            #              <-- j -->
            for fit_j in all_counts_spectrum_fit_pe_table[component_count_i]:
                for fit_k in all_counts_spectrum_fit_pe_table[component_count_i]:
                    if fit_j == fit_k:
                        log.debug(
                            "* component count %d: %8.5f <-- %8.5f --> %8.5f",
                            component_count_i,
                            fit_j.median_C_p_ci_lo,
                            fit_j.median_C_p,
                            fit_j.median_C_p_ci_hi,
                        )
                    elif fit_j.median_C_p_ci_hi < fit_k.median_C_p_ci_lo:
                        # assuming later fits will not overlap with fit_j
                        break
                    elif fit_j.median_C_p_ci_lo > fit_k.median_C_p_ci_hi:
                        # assuming later fits could overlap with fit_j
                        pass
                    elif (
                        fit_j.median_C_p_ci_lo
                        <= fit_k.median_C_p_ci_lo
                        <= fit_j.median_C_p_ci_hi
                    ):
                        log.debug(
                            "  component count %d: %8.5f <-- %8.5f --> %8.5f",
                            component_count_i,
                            fit_k.median_C_p_ci_lo,
                            fit_k.median_C_p,
                            fit_k.median_C_p_ci_hi,
                        )
                    elif (
                        fit_j.median_C_p_ci_lo
                        <= fit_k.median_C_p_ci_hi
                        <= fit_j.median_C_p_ci_hi
                    ):
                        log.debug(
                            "  component count %d: %8.5f <-- %8.5f --> %8.5f",
                            component_count_i,
                            fit_k.median_C_p_ci_lo,
                            fit_k.median_C_p,
                            fit_k.median_C_p_ci_hi,
                        )
                    else:
                        log.debug(
                            "  component count %d: %8.5f <-- %8.5f --> %8.5f",
                            component_count_i,
                            fit_k.median_C_p_ci_lo,
                            fit_k.median_C_p,
                            fit_k.median_C_p_ci_hi,
                        )
                log.debug("***")

            best_fit_for_component_count = all_counts_spectrum_fit_pe_table[
                component_count_i
            ][0]

            component_count_to_median_cp[
                component_count_i
            ] = best_fit_for_component_count.median_C_p
            component_count_to_median_cp_ci_lo_hi[component_count_i] = (
                best_fit_for_component_count.median_C_p_ci_lo,
                best_fit_for_component_count.median_C_p_ci_hi,
            )

        log.debug(
            "component count to median cp: {}".format(component_count_to_median_cp)
        )
        log.debug(
            "component count to median cp confidence interval: {}".format(
                component_count_to_median_cp_ci_lo_hi
            )
        )

        best_component_count, C_p_lo, C_p_hi = self.get_best_ci_component_count(
            component_count_to_median_cp, component_count_to_median_cp_ci_lo_hi
        )
        best_fit = all_counts_spectrum_fit_table[best_component_count][0]
        log.info("best fit: {}".format(best_fit))
        return best_fit

    @staticmethod
    def get_best_ci_component_count(
        component_count_to_median_cp,
        component_count_to_median_cp_ci_lo_hi,
        logger_name_suffix="",
    ):
        """
        Use the 'best subset selection' criterion to choose the 'best' component count using
        confidence intervals for median C_p (prediction error). Choose the component count with
        the smallest median C_p. If two or more C_p confidence intervals overlap choose the lower
        component count.

        component count  lo    hi
        1                0.3   0.4
        2                0.1   0.2

        :param component_count_to_median_cp:
        :param component_count_to_median_cp_ci_lo_hi:
        :param logger_name_suffix: str
        :return: (int) best component count
        """
        log = logging.getLogger(name=__name__ + ":" + logger_name_suffix)

        best_component_count = max(component_count_to_median_cp_ci_lo_hi.keys())
        n_lo, n_hi = component_count_to_median_cp_ci_lo_hi[best_component_count]

        for n in sorted(component_count_to_median_cp_ci_lo_hi.keys())[:-1]:
            n_lo, n_hi = component_count_to_median_cp_ci_lo_hi[n]
            n_plus_1_lo, n_plus_1_hi = component_count_to_median_cp_ci_lo_hi[n + 1]
            log.info("comparing C_p ci for component counts %d and %d", n, n + 1)
            log.info(
                "  component count %d: %8.5f <-- %8.5f --> %8.5f",
                n,
                n_lo,
                component_count_to_median_cp[n],
                n_hi,
            )
            log.info(
                "  component count %d: %8.5f <-- %8.5f --> %8.5f",
                n + 1,
                n_plus_1_lo,
                component_count_to_median_cp[n + 1],
                n_plus_1_hi,
            )
            # must handle two cases:
            #   n_plus_1_hi >= n_lo -> choose n
            #   n_plus_1_hi <  n_lo -> try n+1
            if n_plus_1_hi >= n_lo:
                best_component_count = n
                break

        return best_component_count, n_lo, n_hi

    def calculate_prediction_error_list(self, fit, n_splits, test_size=0.2):
        """
        Given a fit calculate normalized prediction error on n_splits models with randomly withheld data.

        Parameters
        ----------
        fit - instance of SpectrumFit
        n_splits - number of times to calculate prediction error, 1000 is recommended
        test_size - fraction of data to withhold from training, 0.2 is recommended

        Returns
        -------
        normalized_C_p_list - list of normalized prediction errors, one per model
        model_residuals     - (fit.reference_spectra_A_df.shape[0] x n_splits) numpy array of residuals
                              for each model with NaNs at training indices
        """

        normalized_C_p_list = []
        model_residuals = np.full(
            shape=(fit.reference_spectra_A_df.shape[0], n_splits),
            fill_value=np.nan,
            dtype=np.double,
        )
        for i, (predicted_b, train_index, test_index) in enumerate(
            self.fit_and_predict(fit, n_splits=n_splits, test_size=test_size)
        ):

            model_residuals[test_index, i] = (
                fit.unknown_spectrum_b.values[test_index] - predicted_b
            )
            cp = np.sqrt(np.nansum(np.square(model_residuals[test_index, i])))
            normalized_cp = cp / len(test_index)
            normalized_C_p_list.append(normalized_cp)
        return normalized_C_p_list, model_residuals

    def fit_and_predict(self, fit, n_splits=1000, test_size=0.2):
        cv = sklearn.model_selection.ShuffleSplit(
            n_splits=n_splits, test_size=test_size
        )
        for i, (train_index, test_index) in enumerate(
            cv.split(fit.reference_spectra_A_df.values)
        ):
            lm = self.ls()
            lm.fit(
                fit.reference_spectra_A_df.values[train_index],
                fit.unknown_spectrum_b.values[train_index],
            )
            predicted_b = lm.predict(fit.reference_spectra_A_df.values[test_index])
            yield predicted_b, train_index, test_index

    def plot_top_fits(self, spectrum, fit_results):
        # log = logging.getLogger(name=self.__class__.__name__ + ":" + spectrum.file_name)

        figure_list = []

        top_fit_per_component_count = {}
        for i, component_count in enumerate(
            fit_results.component_count_fit_table.keys()
        ):

            pe_fits = [
                fit
                for fit in fit_results.component_count_fit_table[component_count]
                if hasattr(fit, "median_C_p")
            ]

            sorted_fits = sorted(pe_fits, key=lambda fit: fit.median_C_p)[:10]
            top_fit_per_component_count[component_count] = sorted_fits[0]

            f, ax = plt.subplots()
            prediction_error_box_plots(
                ax=ax,
                title=f"Best {component_count}-component Fits\n{spectrum.file_name}",
                sorted_fits=sorted_fits,
            )
            f.tight_layout()
            figure_list.append(f)

            g, ax = plt.subplots()
            prediction_error_confidence_interval_plot(
                ax=ax,
                title=f"Best {component_count}-component Fits\n{spectrum.file_name}",
                sorted_fits=sorted_fits,
            )
            g.tight_layout()
            figure_list.append(g)

        f, ax = plt.subplots()
        best_fit_for_component_count_box_plots(
            ax=ax,
            title=f"Best Fits\n{spectrum.file_name}",
            top_fit_per_component_count=top_fit_per_component_count,
        )
        f.tight_layout()
        figure_list.append(f)

        return figure_list
