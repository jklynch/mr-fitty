import logging

import sklearn.cross_validation
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

        cv = sklearn.cross_validation.ShuffleSplit(
            reference_spectra_A_df.values.shape[0],
            test_size=self.test_size,
            n_iter=self.n_iter
        )
        for train_index, test_index in cv:
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
