import numpy as np
import statsmodels.api as sm
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


class OlsWithStats:
    """Ordinary least-squares via statsmodels, exposing coefficient statistics."""

    def __init__(self):
        self.coef_ = None
        self.residual = None
        self.std_err_ = None
        self.t_values_ = None
        self.p_values_ = None
        self.rsquared_ = None
        self._result = None

    def fit(self, X, y):
        self._result = sm.OLS(y, X).fit()
        self.coef_ = self._result.params
        self.residual = self._result.ssr
        self.std_err_ = self._result.bse
        self.t_values_ = self._result.tvalues
        self.p_values_ = self._result.pvalues
        self.rsquared_ = self._result.rsquared
        return self

    def predict(self, A):
        return np.dot(A, self.coef_)
