"""
The MIT License (MIT)

Copyright (c) 2015-2018 Joshua Lynch, Sarah Nicholas

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
import itertools
import logging
import time

import matplotlib.pyplot as plt
import numpy as np
import scikits.bootstrap
import sklearn.model_selection

from mrfitty.base import AdaptiveEnergyRangeBuilder
from mrfitty.combination_fit import AllCombinationFitTask
from mrfitty.linear_model import NonNegativeLinearRegression


def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)


class BestSubsetSelectionFitTask(AllCombinationFitTask):
    def __init__(
        self,
        reference_spectrum_list,
        unknown_spectrum_list,
        ls=NonNegativeLinearRegression,
        energy_range_builder=AdaptiveEnergyRangeBuilder(),
        component_count_range=range(1, 4),
        best_fits_plot_limit=3
    ):
        super().__init__(
            ls, reference_spectrum_list, unknown_spectrum_list, energy_range_builder, best_fits_plot_limit, component_count_range
        )

    def get_fit_quality_score_text(self, any_given_fit):
        return ' (median C_p: {:8.5f} <-- {:8.5f} --> {:8.5f})'.format(
            any_given_fit.median_C_p_ci_lo,
            any_given_fit.median_C_p,
            any_given_fit.median_C_p_ci_hi)

    def choose_best_component_count(self, all_counts_spectrum_fit_table):
        """
        Calculate the prediction error for top fits.

        :param all_counts_spectrum_fit_table:
          dictionary with component count keys, sorted list of spectrum fit list values
        :return:
        """
        log = logging.getLogger(__name__)

        log.debug('choosing best component count from {}'.format(all_counts_spectrum_fit_table))
        component_count_to_median_cp = {
            component_count: np.Inf
            for component_count in all_counts_spectrum_fit_table.keys()
        }
        component_count_to_median_cp_ci_lo_hi = {
            component_count: (np.Inf, np.Inf)
            for component_count in all_counts_spectrum_fit_table.keys()
        }

        for component_count_i in sorted(all_counts_spectrum_fit_table.keys()):
            # calculate C_p for the first ? fits with component_count_i
            log.debug('calculating CI of median C_p for {} component(s)'.format(component_count_i))

            for fit_j in all_counts_spectrum_fit_table[component_count_i]:
                t0 = time.time()
                prediction_errors, _ = self.calculate_prediction_error_list(fit_j)
                t1 = time.time()
                median_C_p = np.median(prediction_errors)
                ci_lo, ci_hi = scikits.bootstrap.ci(
                    data=prediction_errors,
                    statfunction=np.median)
                fit_j.median_C_p = median_C_p
                fit_j.median_C_p_ci_lo = ci_lo
                fit_j.median_C_p_ci_hi = ci_hi

                fit_j.prediction_errors = prediction_errors

                #log.info('%5.2fs to calculate prediction error list', t1 - t0)

            all_counts_spectrum_fit_table[component_count_i] = sorted(
                all_counts_spectrum_fit_table[component_count_i],
                key=lambda fit: (fit.median_C_p, fit.median_C_p_ci_lo, fit.median_C_p_ci_hi))
            # for each fit find all fits with overlapping ci
            # 4 cases:
            #
            #   <-- k -->                                   j.lo > k.hi (keep checking)
            #              <-- j -->
            #
            #         <-- k -->                             j.lo <= k.hi <= j.hi
            #              <-- j -->
            #
            #             <--  k  -->
            #              <-- j -->
            #
            #               <- k ->
            #              <-- j -->
            #
            #                    <-- k -->                  j.lo <= k.lo <= j.hi
            #              <-- j -->
            #
            #                         <-- k -->             j.hi < k.lo (stop checking)
            #              <-- j -->
            for fit_j in all_counts_spectrum_fit_table[component_count_i]:
                for fit_k in all_counts_spectrum_fit_table[component_count_i]:
                    if fit_j == fit_k:
                        log.info('* component count %d: %8.5f <-- %8.5f --> %8.5f',
                            component_count_i, fit_j.median_C_p_ci_lo, fit_j.median_C_p, fit_j.median_C_p_ci_hi)
                    elif fit_j.median_C_p_ci_hi < fit_k.median_C_p_ci_lo:
                        # assuming later fits will not overlap with fit_j
                        break
                    elif fit_j.median_C_p_ci_lo > fit_k.median_C_p_ci_hi:
                        # assuming later fits could overlap with fit_j
                        pass
                    elif fit_j.median_C_p_ci_lo <= fit_k.median_C_p_ci_lo <= fit_j.median_C_p_ci_hi:
                        log.info('  component count %d: %8.5f <-- %8.5f --> %8.5f',
                                 component_count_i, fit_k.median_C_p_ci_lo, fit_k.median_C_p, fit_k.median_C_p_ci_hi)
                    elif fit_j.median_C_p_ci_lo <= fit_k.median_C_p_ci_hi <= fit_j.median_C_p_ci_hi:
                        log.info('  component count %d: %8.5f <-- %8.5f --> %8.5f',
                                 component_count_i, fit_k.median_C_p_ci_lo, fit_k.median_C_p, fit_k.median_C_p_ci_hi)
                    else:
                        log.info('  component count %d: %8.5f <-- %8.5f --> %8.5f',
                                 component_count_i, fit_k.median_C_p_ci_lo, fit_k.median_C_p, fit_k.median_C_p_ci_hi)
                log.info('***')

            best_fit_for_component_count = all_counts_spectrum_fit_table[component_count_i][0]
            t0 = time.time()
            prediction_errors, _ = self.calculate_prediction_error_list(best_fit_for_component_count)
            t1 = time.time()
            log.info('%5.2fs to calculate prediction error list', t1-t0)
            component_count_to_median_cp[component_count_i] = np.median(prediction_errors)
            component_count_to_median_cp_ci_lo_hi[component_count_i] = scikits.bootstrap.ci(
                data=prediction_errors,
                statfunction=np.median)

        log.debug('component count to median cp: {}'.format(component_count_to_median_cp))
        log.debug('component count to median cp confidence interval: {}'.format(component_count_to_median_cp_ci_lo_hi))

        best_component_count, C_p_lo, C_p_hi = self.get_best_ci_component_count(
            component_count_to_median_cp,
            component_count_to_median_cp_ci_lo_hi)
        best_fit = all_counts_spectrum_fit_table[best_component_count][0]
        log.info('best fit: {}'.format(best_fit))
        return best_fit

    @staticmethod
    def get_best_ci_component_count(component_count_to_median_cp, component_count_to_median_cp_ci_lo_hi):
        """
        Use the 'best subset selection' criterion to choose the 'best' component count using
        confidence intervals for median C_p (prediction error). Choose the component count with
        the smallest median C_p. If two C_p confidence intervals overlap choose the lower count.

        component count  lo    hi
        1                0.3   0.4
        2                0.1   0.2

        :param component_count_to_median_cp:
        :param component_count_to_median_cp_ci_lo_hi:
        :return: (int) best component count
        """
        log = logging.getLogger(name=__name__)

        best_component_count = max(component_count_to_median_cp_ci_lo_hi.keys())
        n_lo, n_hi = component_count_to_median_cp_ci_lo_hi[best_component_count]

        for n in sorted(component_count_to_median_cp_ci_lo_hi.keys())[:-1]:
            n_lo, n_hi = component_count_to_median_cp_ci_lo_hi[n]
            n_plus_1_lo, n_plus_1_hi = component_count_to_median_cp_ci_lo_hi[n+1]
            log.info('comparing C_p ci for component counts %d and %d',  n, n+1)
            log.info('  component count %d: %8.5f <-- %8.5f --> %8.5f', n, n_lo, component_count_to_median_cp[n], n_hi)
            log.info('  component count %d: %8.5f <-- %8.5f --> %8.5f', n+1, n_plus_1_lo, component_count_to_median_cp[n + 1], n_plus_1_hi)
            # must handle two cases:
            #   n_plus_1_hi >= n_lo -> choose n
            #   n_plus_1_hi <  n_lo -> try n+1
            if n_plus_1_hi >= n_lo:
                best_component_count = n
                break

        return best_component_count, n_lo, n_hi

    def calculate_prediction_error_list(self, fit, n_splits=1000, test_size=0.2):
        """
        Given a fit calculate normalized prediction error on 1000 models with randomly withheld data.

        Parameters
        ----------
        fit - instance of SpectrumFit

        Returns
        -------
        normalized_C_p_list - list of normalized prediction errors, one per model
        model_residuals     - (fit.reference_spectra_A_df.shape[0] x n_splits) numpy array of residuals
                              for each model with NaNs at training indices
        """

        normalized_C_p_list = []
        model_residuals = np.full(shape=(fit.reference_spectra_A_df.shape[0], n_splits), fill_value=np.nan, dtype=np.double)
        cv = sklearn.model_selection.ShuffleSplit(n_splits=n_splits, test_size=test_size)
        for i, (train_index, test_index) in enumerate(cv.split(fit.reference_spectra_A_df.values)):
            lm = self.ls()
            lm.fit(
                fit.reference_spectra_A_df.values[train_index],
                fit.unknown_spectrum_b.values[train_index]
            )
            predicted_b = lm.predict(fit.reference_spectra_A_df.values[test_index])
            model_residuals[test_index, i] = fit.unknown_spectrum_b.values[test_index] - predicted_b
            cp = np.sqrt(np.nansum(np.square(model_residuals[test_index, i])))
            normalized_cp = cp / len(test_index)
            normalized_C_p_list.append(normalized_cp)
        return normalized_C_p_list, model_residuals

    def plot_fit_path(self, spectrum, fit_results):
        log = logging.getLogger(name=self.__class__.__name__)

        figure_list = []
        for i, component_count in enumerate(fit_results.component_count_fit_table.keys()):
            f, ax = plt.subplots()

            sorted_fits = fit_results.component_count_fit_table[component_count][:10]
            ax.errorbar(
                x=[spectrum_fit.median_C_p for spectrum_fit in sorted_fits],
                y=range(len(sorted_fits)),
                xerr=[
                    [s.median_C_p - s.median_C_p_ci_lo for s in sorted_fits],
                    [s.median_C_p_ci_hi - s.median_C_p for s in sorted_fits]],
                fmt='o')
            ax.set_title('Fit Path' + '\n' + spectrum.file_name + '\n' + '{} component(s)'.format(component_count))
            ax.set_xlabel('normalized median_C_p')
            ax.set_ylabel('order')

            self.add_date_time_footer(ax)

            f.tight_layout()
            figure_list.append(f)

        return figure_list

