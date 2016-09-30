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
import sklearn.cross_validation
import sklearn.linear_model

from mrfitty.combination_fit import AllCombinationFitTask


class BestSubsetSelectionFitTask(AllCombinationFitTask):
    def __init__(
            self,
            reference_spectrum_list,
            unknown_spectrum_list,
            energy_range_builder,
            component_count_range
    ):
        super(type(self), self).__init__(
            reference_spectrum_list, unknown_spectrum_list, energy_range_builder, component_count_range
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
        component_count_to_cp_list = [None] * (len(best_fit_for_component_count_list) + 1)
        component_count_to_median_cp = [None] * (len(best_fit_for_component_count_list) + 1)
        # todo: this component_count is off by 1
        for component_count, component_count_fit_list in enumerate(best_fit_for_component_count_list):
            best_fit_for_component_count = component_count_fit_list
            log.debug(best_fit_for_component_count.reference_spectra_A_df)
            log.debug(best_fit_for_component_count.unknown_spectrum_b)
            # calculate Cp and 95% confidence interval of the median
            normalized_cp_list = []
            component_count_to_cp_list[component_count] = normalized_cp_list
            cv = sklearn.cross_validation.ShuffleSplit(
                best_fit_for_component_count.reference_spectra_A_df.values.shape[0],
                test_size=0.2,
                n_iter=100
            )
            for train_index, test_index in cv:
                lm = sklearn.linear_model.LinearRegression()
                lm.fit(
                    best_fit_for_component_count.reference_spectra_A_df.values[train_index],
                    best_fit_for_component_count.unknown_spectrum_b.values[train_index]
                )
                predicted_b = lm.predict(best_fit_for_component_count.reference_spectra_A_df.values[test_index])
                residuals = best_fit_for_component_count.unknown_spectrum_b.values[test_index] - predicted_b
                cp = np.sqrt(np.sum(np.square(residuals)))
                normalized_cp = cp / residuals.shape
                normalized_cp_list.append(normalized_cp)

            # todo: calculate confidence interval
            component_count_to_median_cp[component_count] = np.median(normalized_cp_list)
        log.debug('component count to median cp: {}'.format(component_count_to_median_cp))
        best_fit_component_count = np.argmin(np.asarray(component_count_to_median_cp))
        log.info('best fit component count is {}'.format(best_fit_component_count))
        _, best_fit = best_fit_for_component_count_list[best_fit_component_count]

        log.info('best fit: {}'.format(best_fit))
        return best_fit
