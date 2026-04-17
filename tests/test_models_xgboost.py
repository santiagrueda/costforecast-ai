"""Tests unitarios para XGBoostModel."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from costforecast.models.xgboost_model import XGBoostModel


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def regression_data() -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Regresión lineal simple: y = 2*f1 + 3*f2 + ruido."""
    rng = np.random.default_rng(1)
    n_train, n_test = 200, 20
    idx_train = pd.date_range("2020-01-01", periods=n_train, freq="D")
    idx_test = pd.date_range(idx_train[-1] + pd.Timedelta(days=1), periods=n_test, freq="D")

    f1_train = rng.normal(size=n_train)
    f2_train = rng.normal(size=n_train)
    f1_test = rng.normal(size=n_test)
    f2_test = rng.normal(size=n_test)

    y_train = pd.Series(2 * f1_train + 3 * f2_train + rng.normal(size=n_train) * 0.1, index=idx_train)

    X_train = pd.DataFrame({"f1": f1_train, "f2": f2_train}, index=idx_train)
    X_test = pd.DataFrame({"f1": f1_test, "f2": f2_test}, index=idx_test)
    return X_train, y_train, X_test


def _fast_model(**kwargs) -> XGBoostModel:
    return XGBoostModel(n_estimators=20, **kwargs)


# ---------------------------------------------------------------------------
# Fit
# ---------------------------------------------------------------------------

class TestXGBoostFit:
    def test_fit_returns_self(self, regression_data) -> None:
        X_train, y_train, _ = regression_data
        model = _fast_model()
        assert model.fit(X_train, y_train) is model

    def test_fit_stores_feature_names(self, regression_data) -> None:
        X_train, y_train, _ = regression_data
        model = _fast_model().fit(X_train, y_train)
        assert model._feature_names == ["f1", "f2"]

    def test_fit_raises_on_empty_X(self, regression_data) -> None:
        _, y_train, _ = regression_data
        with pytest.raises(ValueError):
            _fast_model().fit(pd.DataFrame(), y_train.iloc[:0])

    def test_fit_raises_on_length_mismatch(self, regression_data) -> None:
        X_train, y_train, _ = regression_data
        with pytest.raises(ValueError, match="mismo largo"):
            _fast_model().fit(X_train.iloc[:10], y_train)


# ---------------------------------------------------------------------------
# Predict
# ---------------------------------------------------------------------------

class TestXGBoostPredict:
    def test_returns_series(self, regression_data) -> None:
        X_train, y_train, X_test = regression_data
        model = _fast_model().fit(X_train, y_train)
        preds = model.predict(X_test)
        assert isinstance(preds, pd.Series)

    def test_length_matches_X_test(self, regression_data) -> None:
        X_train, y_train, X_test = regression_data
        model = _fast_model().fit(X_train, y_train)
        preds = model.predict(X_test)
        assert len(preds) == len(X_test)

    def test_index_preserved(self, regression_data) -> None:
        X_train, y_train, X_test = regression_data
        model = _fast_model().fit(X_train, y_train)
        preds = model.predict(X_test)
        pd.testing.assert_index_equal(preds.index, X_test.index)

    def test_no_nan(self, regression_data) -> None:
        X_train, y_train, X_test = regression_data
        model = _fast_model().fit(X_train, y_train)
        preds = model.predict(X_test)
        assert not preds.isna().any()

    def test_predictions_are_numeric(self, regression_data) -> None:
        X_train, y_train, X_test = regression_data
        model = _fast_model().fit(X_train, y_train)
        preds = model.predict(X_test)
        assert pd.api.types.is_float_dtype(preds)

    def test_learns_linear_relationship(self, regression_data) -> None:
        """Con suficientes árboles, XGBoost debe aproximar y = 2*f1 + 3*f2."""
        X_train, y_train, X_test = regression_data
        model = XGBoostModel(n_estimators=200).fit(X_train, y_train)
        preds = model.predict(X_test)
        y_true = 2 * X_test["f1"] + 3 * X_test["f2"]
        correlation = np.corrcoef(preds.values, y_true.values)[0, 1]
        assert correlation > 0.95, f"Correlación esperada > 0.95, obtenida {correlation:.4f}"

    def test_column_order_independent(self, regression_data) -> None:
        """predict() debe ser insensible al orden de columnas en X."""
        X_train, y_train, X_test = regression_data
        model = _fast_model().fit(X_train, y_train)
        preds_normal = model.predict(X_test)
        preds_reversed = model.predict(X_test[["f2", "f1"]])
        pd.testing.assert_series_equal(preds_normal, preds_reversed)


# ---------------------------------------------------------------------------
# Feature importances
# ---------------------------------------------------------------------------

class TestXGBoostFeatureImportances:
    def test_returns_series(self, regression_data) -> None:
        X_train, y_train, _ = regression_data
        model = _fast_model().fit(X_train, y_train)
        imp = model.feature_importances_
        assert isinstance(imp, pd.Series)

    def test_index_matches_feature_names(self, regression_data) -> None:
        X_train, y_train, _ = regression_data
        model = _fast_model().fit(X_train, y_train)
        assert set(model.feature_importances_.index) == {"f1", "f2"}

    def test_sorted_descending(self, regression_data) -> None:
        X_train, y_train, _ = regression_data
        model = _fast_model().fit(X_train, y_train)
        imp = model.feature_importances_
        assert imp.iloc[0] >= imp.iloc[1]

    def test_importances_sum_to_one(self, regression_data) -> None:
        X_train, y_train, _ = regression_data
        model = _fast_model().fit(X_train, y_train)
        assert abs(model.feature_importances_.sum() - 1.0) < 1e-6

    def test_raises_before_fit(self) -> None:
        with pytest.raises(RuntimeError, match="fit"):
            _ = _fast_model().feature_importances_


# ---------------------------------------------------------------------------
# Validaciones de error
# ---------------------------------------------------------------------------

class TestXGBoostErrors:
    def test_predict_before_fit_raises(self, regression_data) -> None:
        _, _, X_test = regression_data
        with pytest.raises(RuntimeError, match="fit"):
            _fast_model().predict(X_test)

    def test_predict_missing_column_raises(self, regression_data) -> None:
        X_train, y_train, X_test = regression_data
        model = _fast_model().fit(X_train, y_train)
        with pytest.raises(ValueError, match="faltantes"):
            model.predict(X_test.drop(columns=["f1"]))
