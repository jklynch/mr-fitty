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
import logging
import time

import numpy as np
import scikits.bootstrap
import sklearn.model_selection

from mrfitty.combination_fit import AllCombinationFitTask


class BestSubsetSelectionFitTask(AllCombinationFitTask):
    def __init__(
            self,
            ls,
            reference_spectrum_list,
            unknown_spectrum_list,
            energy_range_builder,
            best_fits_plot_limit,
            component_count_range
    ):
        super(type(self), self).__init__(
            ls, reference_spectrum_list, unknown_spectrum_list, energy_range_builder, best_fits_plot_limit, component_count_range
        )

    def choose_best_component_count(self, all_counts_spectrum_fit_table):
        """
        Calculate the prediction error for each subset.
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
            log.debug('calculating CI of median C_p for {} component(s)'.format(component_count_i))

            best_fit_for_component_count = all_counts_spectrum_fit_table[component_count_i][0]
            t0 = time.time()
            prediction_error_list = self.calculate_prediction_error_list(best_fit_for_component_count)
            t1 = time.time()
            log.info('%5.2fs to calculate prediction error list', t1-t0)
            component_count_to_median_cp[component_count_i] = np.median(prediction_error_list)
            component_count_to_median_cp_ci_lo_hi[component_count_i] = scikits.bootstrap.ci(
                data=prediction_error_list,
                statfunction=np.median)

        log.debug('component count to median cp: {}'.format(component_count_to_median_cp))
        log.debug('component count to median cp confidence interval: {}'.format(component_count_to_median_cp_ci_lo_hi))

        best_component_count = self.get_best_ci_component_count(component_count_to_median_cp_ci_lo_hi)
        best_fit = all_counts_spectrum_fit_table[best_component_count][0]
        log.info('best fit: {}'.format(best_fit))
        return best_fit

    @staticmethod
    def get_best_ci_component_count(component_count_to_median_cp_ci_lo_hi):
        """
        Use the 'best subset selection' criterion to choose the 'best' component count using
        confidence intervals for median C_p (prediction error). Choose the component count with
        the smallest median C_p. If two C_p confidence intervals overlap choose the lower count.

        component count  lo    hi
        1                0.3   0.4
        2                0.1   0.2

        :param component_count_to_median_cp_ci_lo_hi:
        :return: (int) best component count
        """
        log = logging.getLogger(name=__name__)
        best_component_count = max(component_count_to_median_cp_ci_lo_hi.keys())
        for n in sorted(component_count_to_median_cp_ci_lo_hi.keys())[:-1]:
            n_lo, n_hi = component_count_to_median_cp_ci_lo_hi[n]
            n_plus_1_lo, n_plus_1_hi = component_count_to_median_cp_ci_lo_hi[n+1]
            log.debug('comparing C_p ci for %d with C_p ci for %d', n, n+1)
            # must handle two cases:
            #   n_plus_1_hi >= n_lo -> choose n
            #   n_plus_1_hi <  n_lo -> try n+1
            if n_plus_1_hi >= n_lo:
                best_component_count = n
                break

        return best_component_count

    def calculate_prediction_error_list(self, fit):
        # calculate Cp and 95% confidence interval of the median
        normalized_cp_list = []
        cv = sklearn.model_selection.ShuffleSplit(n_splits=1000, test_size=0.2)
        for train_index, test_index in cv.split(fit.reference_spectra_A_df.values):
            lm = self.ls()
            lm.fit(
                fit.reference_spectra_A_df.values[train_index],
                fit.unknown_spectrum_b.values[train_index]
            )
            predicted_b = lm.predict(fit.reference_spectra_A_df.values[test_index])
            residuals = fit.unknown_spectrum_b.values[test_index] - predicted_b
            cp = np.sqrt(np.sum(np.square(residuals)))
            normalized_cp = cp / residuals.shape
            normalized_cp_list.append(normalized_cp)
        return normalized_cp_list
