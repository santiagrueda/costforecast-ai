"""
Tests unitarios para ProphetModel.

Prophet necesita al menos ~2 períodos para ajustar. Se usa un dataset
mensual pequeño (24 meses) con uncertainty_samples=0 para acelerar el test.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from costforecast.models.prophet_model import ProphetModel


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def monthly_data() -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Serie mensual sintética con 1 regresor: 24 meses train, 3 test."""
    rng = np.random.default_rng(7)
    n_train, n_test = 24, 3
    idx_train = pd.date_range("2021-01-01", periods=n_train, freq="MS")
    idx_test = pd.date_range(
        idx_train[-1] + pd.DateOffset(months=1), periods=n_test, freq="MS"
    )

    reg_train = rng.normal(size=n_train)
    reg_test = rng.normal(size=n_test)
    y_train = pd.Series(
        np.linspace(100, 150, n_train) + reg_train * 5 + rng.normal(size=n_train),
        index=idx_train,
    )

    X_train = pd.DataFrame({"reg": reg_train}, index=idx_train)
    X_test = pd.DataFrame({"reg": reg_test}, index=idx_test)
    return X_train, y_train, X_test


@pytest.fixture
def no_regressor_data() -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Serie mensual sin regresor."""
    rng = np.random.default_rng(99)
    n_train, n_test = 24, 3
    idx_train = pd.date_range("2021-01-01", periods=n_train, freq="MS")
    idx_test = pd.date_range(
        idx_train[-1] + pd.DateOffset(months=1), periods=n_test, freq="MS"
    )
    y_train = pd.Series(np.linspace(200, 250, n_train) + rng.normal(size=n_train), index=idx_train)
    X_train = pd.DataFrame(index=idx_train)
    X_test = pd.DataFrame(index=idx_test)
    return X_train, y_train, X_test


def _fast_model() -> ProphetModel:
    """ProphetModel con parámetros mínimos para tests rápidos."""
    return ProphetModel(
        yearly_seasonality=False,
        weekly_seasonality=False,
        daily_seasonality=False,
        uncertainty_samples=0,
    )


# ---------------------------------------------------------------------------
# Fit
# ---------------------------------------------------------------------------

class TestProphetFit:
    def test_fit_returns_self(self, monthly_data) -> None:
        X_train, y_train, _ = monthly_data
        model = _fast_model()
        assert model.fit(X_train, y_train) is model

    def test_fit_stores_regressor_cols(self, monthly_data) -> None:
        X_train, y_train, _ = monthly_data
        model = _fast_model().fit(X_train, y_train)
        assert model._regressor_cols == ["reg"]

    def test_fit_no_regressors(self, no_regressor_data) -> None:
        X_train, y_train, _ = no_regressor_data
        model = _fast_model().fit(X_train, y_train)
        assert model._regressor_cols == []

    def test_fit_raises_on_empty_y(self, monthly_data) -> None:
        X_train, _, _ = monthly_data
        with pytest.raises(ValueError):
            _fast_model().fit(X_train, pd.Series([], dtype=float))

    def test_fit_raises_on_length_mismatch(self, monthly_data) -> None:
        X_train, y_train, _ = monthly_data
        with pytest.raises(ValueError, match="mismo largo"):
            _fast_model().fit(X_train.iloc[:5], y_train)


# ---------------------------------------------------------------------------
# Predict
# ---------------------------------------------------------------------------

class TestProphetPredict:
    def test_returns_series(self, monthly_data) -> None:
        X_train, y_train, X_test = monthly_data
        model = _fast_model().fit(X_train, y_train)
        preds = model.predict(X_test)
        assert isinstance(preds, pd.Series)

    def test_length_matches_X_test(self, monthly_data) -> None:
        X_train, y_train, X_test = monthly_data
        model = _fast_model().fit(X_train, y_train)
        preds = model.predict(X_test)
        assert len(preds) == len(X_test)

    def test_index_preserved(self, monthly_data) -> None:
        X_train, y_train, X_test = monthly_data
        model = _fast_model().fit(X_train, y_train)
        preds = model.predict(X_test)
        pd.testing.assert_index_equal(preds.index, X_test.index)

    def test_no_nan_in_predictions(self, monthly_data) -> None:
        X_train, y_train, X_test = monthly_data
        model = _fast_model().fit(X_train, y_train)
        preds = model.predict(X_test)
        assert not preds.isna().any()

    def test_predict_no_regressors(self, no_regressor_data) -> None:
        X_train, y_train, X_test = no_regressor_data
        model = _fast_model().fit(X_train, y_train)
        preds = model.predict(X_test)
        assert len(preds) == len(X_test)
        assert not preds.isna().any()

    def test_in_sample_predict(self, monthly_data) -> None:
        """predict() sobre datos de entrenamiento también debe funcionar."""
        X_train, y_train, _ = monthly_data
        model = _fast_model().fit(X_train, y_train)
        preds = model.predict(X_train)
        assert len(preds) == len(X_train)


# ---------------------------------------------------------------------------
# Validaciones de error
# ---------------------------------------------------------------------------

class TestProphetErrors:
    def test_predict_before_fit_raises(self, monthly_data) -> None:
        _, _, X_test = monthly_data
        with pytest.raises(RuntimeError, match="fit"):
            _fast_model().predict(X_test)

    def test_predict_wrong_regressor_raises(self, monthly_data) -> None:
        X_train, y_train, X_test = monthly_data
        model = _fast_model().fit(X_train, y_train)
        X_bad = X_test.rename(columns={"reg": "otro"})
        with pytest.raises(ValueError, match="regressor"):
            model.predict(X_bad)

    def test_predict_missing_regressor_raises(self, monthly_data) -> None:
        X_train, y_train, X_test = monthly_data
        model = _fast_model().fit(X_train, y_train)
        X_bad = X_test.drop(columns=["reg"])
        with pytest.raises(ValueError, match="regressor"):
            model.predict(X_bad)
