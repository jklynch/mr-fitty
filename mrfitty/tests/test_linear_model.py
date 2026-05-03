import numpy as np
import pytest
from mrfitty.linear_model import OlsWithStats


@pytest.fixture
def synthetic_data():
    rng = np.random.default_rng(42)
    n_samples, n_features = 100, 3
    true_coef = np.array([0.5, 0.3, 0.2])
    X = rng.random((n_samples, n_features))
    y = X @ true_coef + rng.normal(0, 0.01, n_samples)
    return X, y, true_coef


def test_ols_with_stats_fit_coef(synthetic_data):
    X, y, true_coef = synthetic_data
    model = OlsWithStats().fit(X, y)
    np.testing.assert_allclose(model.coef_, true_coef, atol=0.05)


def test_ols_with_stats_std_err(synthetic_data):
    X, y, _ = synthetic_data
    model = OlsWithStats().fit(X, y)
    assert model.std_err_ is not None
    assert len(model.std_err_) == X.shape[1]
    assert all(model.std_err_ >= 0)


def test_ols_with_stats_statistics_populated(synthetic_data):
    X, y, _ = synthetic_data
    model = OlsWithStats().fit(X, y)
    assert model.t_values_ is not None and len(model.t_values_) == X.shape[1]
    assert model.p_values_ is not None and len(model.p_values_) == X.shape[1]
    assert model.rsquared_ is not None
    assert 0.0 <= model.rsquared_ <= 1.0


def test_ols_with_stats_predict(synthetic_data):
    X, y, _ = synthetic_data
    model = OlsWithStats().fit(X, y)
    np.testing.assert_allclose(model.predict(X), X @ model.coef_)


def test_ols_with_stats_residual(synthetic_data):
    X, y, _ = synthetic_data
    model = OlsWithStats().fit(X, y)
    assert model.residual >= 0.0


def test_ols_with_stats_fit_returns_self(synthetic_data):
    X, y, _ = synthetic_data
    model = OlsWithStats()
    assert model.fit(X, y) is model
