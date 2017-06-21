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

import numpy as np
import scikits.bootstrap
import sklearn.model_selection
#import sklearn.linear_model

from mrfitty.combination_fit import AllCombinationFitTask


class BestSubsetSelectionFitTask(AllCombinationFitTask):
    def __init__(
            self,
            ls,
            reference_spectrum_list,
            unknown_spectrum_list,
            energy_range_builder,
            component_count_range
    ):
        super(type(self), self).__init__(
            ls, reference_spectrum_list, unknown_spectrum_list, energy_range_builder, component_count_range
        )

    def choose_best_component_count(self, best_fit_for_component_count_list):
        """
        Calculate the prediction error for each subset.
        :param best_fit_for_component_count_list:
          a list of fits for component counts 0 to N; there is no 0-component fit
        :return:
        """
        log = logging.getLogger(__name__)
        log.setLevel(logging.DEBUG)

        log.debug('choosing best component count from {}'.format(best_fit_for_component_count_list))
        component_count_to_cp_list = [np.Inf] * len(best_fit_for_component_count_list)
        component_count_to_median_cp = [np.Inf] * len(best_fit_for_component_count_list)
        component_count_to_median_cp_ci_lo_hi = [(np.Inf, np.Inf)] * len(best_fit_for_component_count_list)
        # todo: this component_count is off by 1
        for component_count_i, best_fit_for_component_count in enumerate(best_fit_for_component_count_list):
            log.debug('calculating CI of median C_p for {} component(s)'.format(component_count_i + 1))
            #log.debug(best_fit_for_component_count.reference_spectra_A_df)
            #log.debug(best_fit_for_component_count.unknown_spectrum_b)
            # calculate Cp and 95% confidence interval of the median
            normalized_cp_list = []
            component_count_to_cp_list[component_count_i] = normalized_cp_list
            cv = sklearn.model_selection.ShuffleSplit(n_splits=1000, test_size=0.2)
            for train_index, test_index in cv.split(best_fit_for_component_count.reference_spectra_A_df.values):
                lm = self.ls()
                lm.fit(
                    best_fit_for_component_count.reference_spectra_A_df.values[train_index],
                    best_fit_for_component_count.unknown_spectrum_b.values[train_index]
                )
                predicted_b = lm.predict(best_fit_for_component_count.reference_spectra_A_df.values[test_index])
                residuals = best_fit_for_component_count.unknown_spectrum_b.values[test_index] - predicted_b
                cp = np.sqrt(np.sum(np.square(residuals)))
                normalized_cp = cp / residuals.shape
                normalized_cp_list.append(normalized_cp)

            component_count_to_median_cp[component_count_i] = np.median(normalized_cp_list)
            component_count_to_median_cp_ci_lo_hi[component_count_i] = scikits.bootstrap.ci(
                data=normalized_cp_list,
                statfunction=np.median)

        log.debug('component count to median cp: {}'.format(component_count_to_median_cp))
        log.debug('component count to median cp confidence interval: {}'.format(component_count_to_median_cp_ci_lo_hi))
        #best_fit_component_count_i = np.argmin(np.asarray(component_count_to_median_cp))

        # compare ci_1_lo with ci_2_hi
        # if ci_1_lo overlaps ci_2_hi then component_count is 1
        best_fit_component_count_i = len(component_count_to_median_cp_ci_lo_hi) - 1
        prev_lo = np.Inf
        for i, (lo, hi) in enumerate(component_count_to_median_cp_ci_lo_hi):
            print('lo: {} hi: {}'.format(lo, hi))
            if prev_lo <= hi:
                best_fit_component_count_i = i
                break
            else:
                prev_lo = lo

        best_fit_component_count = 1 + best_fit_component_count_i
        log.info('best fit component count is {}'.format(best_fit_component_count))
        best_fit = best_fit_for_component_count_list[best_fit_component_count_i]

        log.info('best fit: {}'.format(best_fit))
        return best_fit
