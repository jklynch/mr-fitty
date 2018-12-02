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
import logging

import sklearn.model_selection
import numpy as np

import mrfitty.linear_model as linear_model


class NormalizedSumOfSquares:
    def __init__(self):
        pass

    def loss(self):
        return 0.0


class PredictionError:
    def __init__(self, test_size, n_iter):
        self.test_size = test_size
        self.n_iter = n_iter
        self.normalized_cp_list = None

    def loss(self, reference_spectra_A_df, unknown_spectrum_b):
        self.normalized_cp_list = []

        cv = sklearn.model_selection.ShuffleSplit(
            #reference_spectra_A_df.values.shape[0],
            test_size=self.test_size,
            n_splits=self.n_iter
        )
        for train_index, test_index in cv.split(X=reference_spectra_A_df.values):
            #lm = sklearn.linear_model.LinearRegression()
            lm = linear_model.NonNegativeLinearRegression()
            lm.fit(
                reference_spectra_A_df.values[train_index],
                unknown_spectrum_b.values[train_index]
            )
            predicted_b = lm.predict(reference_spectra_A_df.values[test_index])
            residuals = unknown_spectrum_b.values[test_index] - predicted_b
            cp = np.sqrt(np.sum(np.square(residuals)))
            normalized_cp = cp / residuals.shape
            self.normalized_cp_list.append(normalized_cp)

        # todo: calculate confidence interval
        return np.median(self.normalized_cp_list)
