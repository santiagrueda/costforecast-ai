"""
Tests unitarios para SARIMAXModel.

Se usa un ARIMA(1,0,0) sin componente estacional sobre datasets pequeños
para mantener los tests rápidos (< 5 s en total).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from costforecast.models.sarimax_model import SARIMAXModel


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def ar1_data() -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Serie AR(1) sintética: 60 obs de train, 5 de test."""
    rng = np.random.default_rng(42)
    n_train = 60
    idx_train = pd.bdate_range("2020-01-01", periods=n_train)
    idx_test = pd.bdate_range(idx_train[-1] + pd.offsets.BDay(1), periods=5)

    # AR(1): y_t = 0.7 * y_{t-1} + eps
    values = np.zeros(n_train + 5)
    for t in range(1, len(values)):
        values[t] = 0.7 * values[t - 1] + rng.normal()

    y_train = pd.Series(values[:n_train], index=idx_train, name="target")
    X_train = pd.DataFrame(index=idx_train)          # sin regresor — ARIMA puro
    X_test = pd.DataFrame(index=idx_test)
    return X_train, y_train, X_test


@pytest.fixture
def arx_data() -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Serie con regresor exógeno."""
    rng = np.random.default_rng(0)
    n_train, n_test = 60, 5
    idx_train = pd.bdate_range("2020-01-01", periods=n_train)
    idx_test = pd.bdate_range(idx_train[-1] + pd.offsets.BDay(1), periods=n_test)

    exog_train = rng.normal(size=n_train)
    exog_test = rng.normal(size=n_test)
    y_train = pd.Series(0.5 * exog_train + rng.normal(size=n_train), index=idx_train)

    X_train = pd.DataFrame({"reg": exog_train}, index=idx_train)
    X_test = pd.DataFrame({"reg": exog_test}, index=idx_test)
    return X_train, y_train, X_test


# ---------------------------------------------------------------------------
# Fit
# ---------------------------------------------------------------------------

class TestSARIMAXFit:
    def test_fit_returns_self(self, ar1_data) -> None:
        X_train, y_train, _ = ar1_data
        model = SARIMAXModel(order=(1, 0, 0))
        assert model.fit(X_train, y_train) is model

    def test_fit_stores_exog_cols(self, arx_data) -> None:
        X_train, y_train, _ = arx_data
        model = SARIMAXModel(order=(1, 0, 0)).fit(X_train, y_train)
        assert model._exog_cols == ["reg"]

    def test_fit_raises_on_empty_y(self, ar1_data) -> None:
        X_train, _, _ = ar1_data
        with pytest.raises(ValueError):
            SARIMAXModel(order=(1, 0, 0)).fit(X_train, pd.Series([], dtype=float))

    def test_fit_raises_on_length_mismatch(self, ar1_data) -> None:
        X_train, y_train, _ = ar1_data
        with pytest.raises(ValueError, match="mismo largo"):
            SARIMAXModel(order=(1, 0, 0)).fit(X_train.iloc[:10], y_train)


# ---------------------------------------------------------------------------
# Predict — en muestra
# ---------------------------------------------------------------------------

class TestSARIMAXPredictInSample:
    def test_returns_series(self, ar1_data) -> None:
        X_train, y_train, _ = ar1_data
        model = SARIMAXModel(order=(1, 0, 0)).fit(X_train, y_train)
        preds = model.predict(X_train)
        assert isinstance(preds, pd.Series)

    def test_length_matches(self, ar1_data) -> None:
        X_train, y_train, _ = ar1_data
        model = SARIMAXModel(order=(1, 0, 0)).fit(X_train, y_train)
        preds = model.predict(X_train)
        assert len(preds) == len(X_train)

    def test_index_preserved(self, ar1_data) -> None:
        X_train, y_train, _ = ar1_data
        model = SARIMAXModel(order=(1, 0, 0)).fit(X_train, y_train)
        preds = model.predict(X_train)
        pd.testing.assert_index_equal(preds.index, X_train.index)


# ---------------------------------------------------------------------------
# Predict — fuera de muestra
# ---------------------------------------------------------------------------

class TestSARIMAXPredictOutOfSample:
    def test_returns_series(self, ar1_data) -> None:
        X_train, y_train, X_test = ar1_data
        model = SARIMAXModel(order=(1, 0, 0)).fit(X_train, y_train)
        preds = model.predict(X_test)
        assert isinstance(preds, pd.Series)

    def test_length_matches_test(self, ar1_data) -> None:
        X_train, y_train, X_test = ar1_data
        model = SARIMAXModel(order=(1, 0, 0)).fit(X_train, y_train)
        preds = model.predict(X_test)
        assert len(preds) == len(X_test)

    def test_no_nan_in_forecast(self, ar1_data) -> None:
        X_train, y_train, X_test = ar1_data
        model = SARIMAXModel(order=(1, 0, 0)).fit(X_train, y_train)
        preds = model.predict(X_test)
        assert not preds.isna().any()

    def test_index_matches_X_test(self, ar1_data) -> None:
        X_train, y_train, X_test = ar1_data
        model = SARIMAXModel(order=(1, 0, 0)).fit(X_train, y_train)
        preds = model.predict(X_test)
        pd.testing.assert_index_equal(preds.index, X_test.index)

    def test_with_exog(self, arx_data) -> None:
        X_train, y_train, X_test = arx_data
        model = SARIMAXModel(order=(1, 0, 0)).fit(X_train, y_train)
        preds = model.predict(X_test)
        assert len(preds) == len(X_test)
        assert not preds.isna().any()


# ---------------------------------------------------------------------------
# Validaciones de error
# ---------------------------------------------------------------------------

class TestSARIMAXErrors:
    def test_predict_before_fit_raises(self, ar1_data) -> None:
        _, _, X_test = ar1_data
        with pytest.raises(RuntimeError, match="fit"):
            SARIMAXModel(order=(1, 0, 0)).predict(X_test)

    def test_predict_wrong_exog_columns_raises(self, arx_data) -> None:
        X_train, y_train, X_test = arx_data
        model = SARIMAXModel(order=(1, 0, 0)).fit(X_train, y_train)
        X_bad = X_test.rename(columns={"reg": "wrong"})
        with pytest.raises(ValueError, match="exóg"):
            model.predict(X_bad)
