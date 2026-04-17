"""Tests unitarios para PersistenceModel."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from costforecast.models.baseline import PersistenceModel


@pytest.fixture
def sample_data() -> tuple[pd.DataFrame, pd.Series]:
    idx = pd.date_range("2022-01-01", periods=30, freq="D")
    X = pd.DataFrame({"x1": np.linspace(1, 30, 30)}, index=idx)
    y = pd.Series(np.linspace(100, 200, 30), index=idx, name="target")
    return X, y


class TestPersistenceModelFit:
    def test_fit_returns_self(self, sample_data) -> None:
        X, y = sample_data
        model = PersistenceModel()
        result = model.fit(X, y)
        assert result is model

    def test_fit_stores_last_value(self, sample_data) -> None:
        X, y = sample_data
        model = PersistenceModel().fit(X, y)
        assert model.last_value == pytest.approx(y.iloc[-1])

    def test_fit_raises_on_empty_y(self, sample_data) -> None:
        X, _ = sample_data
        with pytest.raises(ValueError):
            PersistenceModel().fit(X, pd.Series([], dtype=float))


class TestPersistenceModelPredict:
    def test_predict_returns_series(self, sample_data) -> None:
        X, y = sample_data
        model = PersistenceModel().fit(X, y)
        preds = model.predict(X)
        assert isinstance(preds, pd.Series)

    def test_predict_length_matches_X(self, sample_data) -> None:
        X, y = sample_data
        model = PersistenceModel().fit(X, y)
        future_idx = pd.date_range("2022-02-01", periods=10, freq="D")
        X_future = pd.DataFrame({"x1": np.zeros(10)}, index=future_idx)
        preds = model.predict(X_future)
        assert len(preds) == 10

    def test_predict_all_equal_last_value(self, sample_data) -> None:
        X, y = sample_data
        model = PersistenceModel().fit(X, y)
        preds = model.predict(X)
        assert (preds == y.iloc[-1]).all()

    def test_predict_index_preserved(self, sample_data) -> None:
        X, y = sample_data
        model = PersistenceModel().fit(X, y)
        preds = model.predict(X)
        pd.testing.assert_index_equal(preds.index, X.index)

    def test_predict_before_fit_raises(self, sample_data) -> None:
        X, _ = sample_data
        with pytest.raises(RuntimeError, match="fit"):
            PersistenceModel().predict(X)

    def test_last_value_before_fit_raises(self) -> None:
        with pytest.raises(RuntimeError):
            _ = PersistenceModel().last_value

    def test_predict_constant_series(self) -> None:
        """Si todos los valores son iguales, la predicción también lo es."""
        idx = pd.date_range("2022-01-01", periods=5, freq="D")
        X = pd.DataFrame({"x": [1, 2, 3, 4, 5]}, index=idx)
        y = pd.Series([42.0] * 5, index=idx)
        model = PersistenceModel().fit(X, y)
        assert (model.predict(X) == 42.0).all()
