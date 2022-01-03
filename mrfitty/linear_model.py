import numpy as np
from scipy.optimize import nnls


class NonNegativeLinearRegression:
    def __init__(self):
        self.coef_ = None
        self.residual = None

    def fit(self, X, y):
        self.coef_, self.residual, *extra = nnls(X, y)
        return self

    def predict(self, A):
        return np.dot(A, self.coef_)
