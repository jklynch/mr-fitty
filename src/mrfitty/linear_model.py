import numpy as np
from scipy.optimize import nnls
import sklearn.linear_model


class NonNegativeLinearRegression:
    def __init__(self):
        self.reference_spectra_coef_x = None
        self.residual = None

    def fit(self, X, y):
        self.reference_spectra_coef_x, self.residual, *extra = nnls(X, y)
        return self

    def predict(self, A):
        return np.dot(A, self.reference_spectra_coef_x)
