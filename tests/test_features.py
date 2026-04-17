"""Tests unitarios para src/costforecast/features/engineering.py."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from costforecast.features.engineering import (
    create_differences,
    create_feature_matrix,
    create_lags,
    create_rolling_stats,
)


# ---------------------------------------------------------------------------
# Fixture compartida
# ---------------------------------------------------------------------------

@pytest.fixture
def price_df() -> pd.DataFrame:
    """DataFrame diario con 3 series de precio durante 60 días."""
    rng = np.random.default_rng(0)
    idx = pd.date_range("2020-01-01", periods=60, freq="D")
    return pd.DataFrame(
        {
            "X": 100 + np.cumsum(rng.normal(0, 1, 60)),
            "Y": 500 + np.cumsum(rng.normal(0, 2, 60)),
            "equipo": 300 + np.cumsum(rng.normal(0, 1.5, 60)),
        },
        index=idx,
    )


# ---------------------------------------------------------------------------
# create_lags
# ---------------------------------------------------------------------------

class TestCreateLags:
    def test_adds_expected_columns(self, price_df: pd.DataFrame) -> None:
        result = create_lags(price_df, ["X", "Y"], [1, 3])
        assert "X_lag1" in result.columns
        assert "X_lag3" in result.columns
        assert "Y_lag1" in result.columns
        assert "Y_lag3" in result.columns

    def test_original_columns_preserved(self, price_df: pd.DataFrame) -> None:
        result = create_lags(price_df, ["X"], [1])
        for col in price_df.columns:
            assert col in result.columns

    def test_lag_values_are_shifted(self, price_df: pd.DataFrame) -> None:
        result = create_lags(price_df, ["X"], [1])
        # X_lag1 en fila i debe ser igual a X en fila i-1
        pd.testing.assert_series_equal(
            result["X_lag1"].iloc[1:].reset_index(drop=True),
            result["X"].iloc[:-1].reset_index(drop=True),
            check_names=False,
        )

    def test_first_rows_are_nan(self, price_df: pd.DataFrame) -> None:
        result = create_lags(price_df, ["X"], [3])
        assert result["X_lag3"].iloc[:3].isna().all()
        assert not result["X_lag3"].iloc[3:].isna().any()

    def test_empty_lags_returns_copy(self, price_df: pd.DataFrame) -> None:
        result = create_lags(price_df, ["X"], [])
        pd.testing.assert_frame_equal(result, price_df)

    def test_does_not_mutate_input(self, price_df: pd.DataFrame) -> None:
        original_cols = list(price_df.columns)
        create_lags(price_df, ["X"], [1, 2])
        assert list(price_df.columns) == original_cols

    def test_raises_on_missing_column(self, price_df: pd.DataFrame) -> None:
        with pytest.raises(ValueError, match="no encontradas"):
            create_lags(price_df, ["inexistente"], [1])

    def test_raises_on_non_positive_lag(self, price_df: pd.DataFrame) -> None:
        with pytest.raises(ValueError, match="positivos"):
            create_lags(price_df, ["X"], [0])


# ---------------------------------------------------------------------------
# create_rolling_stats
# ---------------------------------------------------------------------------

class TestCreateRollingStats:
    def test_adds_mean_and_std_columns(self, price_df: pd.DataFrame) -> None:
        result = create_rolling_stats(price_df, ["X"], [5])
        assert "X_roll5_mean" in result.columns
        assert "X_roll5_std" in result.columns

    def test_multiple_windows(self, price_df: pd.DataFrame) -> None:
        result = create_rolling_stats(price_df, ["X", "Y"], [5, 10])
        expected = ["X_roll5_mean", "X_roll5_std", "X_roll10_mean", "X_roll10_std",
                    "Y_roll5_mean", "Y_roll5_std", "Y_roll10_mean", "Y_roll10_std"]
        for col in expected:
            assert col in result.columns

    def test_nan_in_warmup_period(self, price_df: pd.DataFrame) -> None:
        result = create_rolling_stats(price_df, ["X"], [10])
        # Las primeras 9 filas deben ser NaN (min_periods=window)
        assert result["X_roll10_mean"].iloc[:9].isna().all()
        assert not result["X_roll10_mean"].iloc[9:].isna().any()

    def test_mean_value_correctness(self, price_df: pd.DataFrame) -> None:
        result = create_rolling_stats(price_df, ["X"], [5])
        # Verificar manualmente la media en fila 10 (suficientes datos)
        expected_mean = price_df["X"].iloc[6:11].mean()
        assert abs(result["X_roll5_mean"].iloc[10] - expected_mean) < 1e-10

    def test_empty_windows_returns_copy(self, price_df: pd.DataFrame) -> None:
        result = create_rolling_stats(price_df, ["X"], [])
        pd.testing.assert_frame_equal(result, price_df)

    def test_does_not_mutate_input(self, price_df: pd.DataFrame) -> None:
        original_cols = list(price_df.columns)
        create_rolling_stats(price_df, ["X"], [5])
        assert list(price_df.columns) == original_cols

    def test_raises_on_window_lte_one(self, price_df: pd.DataFrame) -> None:
        with pytest.raises(ValueError, match="> 1"):
            create_rolling_stats(price_df, ["X"], [1])

    def test_raises_on_missing_column(self, price_df: pd.DataFrame) -> None:
        with pytest.raises(ValueError, match="no encontradas"):
            create_rolling_stats(price_df, ["inexistente"], [5])


# ---------------------------------------------------------------------------
# create_differences
# ---------------------------------------------------------------------------

class TestCreateDifferences:
    def test_adds_diff1_and_diff2_columns(self, price_df: pd.DataFrame) -> None:
        result = create_differences(price_df, ["X", "Y"])
        assert "X_diff1" in result.columns
        assert "X_diff2" in result.columns
        assert "Y_diff1" in result.columns
        assert "Y_diff2" in result.columns

    def test_diff1_values_are_correct(self, price_df: pd.DataFrame) -> None:
        result = create_differences(price_df, ["X"])
        expected = price_df["X"].diff(1)
        pd.testing.assert_series_equal(result["X_diff1"], expected, check_names=False)

    def test_diff2_values_are_correct(self, price_df: pd.DataFrame) -> None:
        result = create_differences(price_df, ["X"])
        expected = price_df["X"].diff(2)
        pd.testing.assert_series_equal(result["X_diff2"], expected, check_names=False)

    def test_diff1_first_row_is_nan(self, price_df: pd.DataFrame) -> None:
        result = create_differences(price_df, ["X"])
        assert pd.isna(result["X_diff1"].iloc[0])
        assert not result["X_diff1"].iloc[1:].isna().any()

    def test_diff2_first_two_rows_are_nan(self, price_df: pd.DataFrame) -> None:
        result = create_differences(price_df, ["X"])
        assert result["X_diff2"].iloc[:2].isna().all()
        assert not result["X_diff2"].iloc[2:].isna().any()

    def test_empty_columns_returns_copy(self, price_df: pd.DataFrame) -> None:
        result = create_differences(price_df, [])
        pd.testing.assert_frame_equal(result, price_df)

    def test_does_not_mutate_input(self, price_df: pd.DataFrame) -> None:
        original_cols = list(price_df.columns)
        create_differences(price_df, ["X"])
        assert list(price_df.columns) == original_cols

    def test_raises_on_missing_column(self, price_df: pd.DataFrame) -> None:
        with pytest.raises(ValueError, match="no encontradas"):
            create_differences(price_df, ["inexistente"])


# ---------------------------------------------------------------------------
# create_feature_matrix
# ---------------------------------------------------------------------------

class TestCreateFeatureMatrix:
    def test_returns_tuple_of_dataframe_and_series(self, price_df: pd.DataFrame) -> None:
        X, y = create_feature_matrix(price_df, target="equipo")
        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.Series)

    def test_target_not_in_X(self, price_df: pd.DataFrame) -> None:
        X, _ = create_feature_matrix(price_df, target="equipo")
        assert "equipo" not in X.columns

    def test_y_equals_target_column_aligned(self, price_df: pd.DataFrame) -> None:
        X, y = create_feature_matrix(price_df, target="equipo")
        pd.testing.assert_index_equal(X.index, y.index)

    def test_no_nan_in_output(self, price_df: pd.DataFrame) -> None:
        X, y = create_feature_matrix(price_df, target="equipo")
        assert not X.isna().any().any(), "X no debe tener NaN"
        assert not y.isna().any(), "y no debe tener NaN"

    def test_row_count_reduced_by_warmup(self, price_df: pd.DataFrame) -> None:
        # Con lags=[1..5] y windows=[5,10,20], el warmup mínimo es max(5, 20) = 20 filas
        X, y = create_feature_matrix(price_df, target="equipo")
        assert len(X) < len(price_df)

    def test_feature_columns_contain_lags(self, price_df: pd.DataFrame) -> None:
        X, _ = create_feature_matrix(
            price_df, target="equipo", lag_columns=["X"], lags=[2, 4]
        )
        assert "X_lag2" in X.columns
        assert "X_lag4" in X.columns

    def test_feature_columns_contain_rolling(self, price_df: pd.DataFrame) -> None:
        X, _ = create_feature_matrix(
            price_df, target="equipo", rolling_columns=["X"], windows=[7]
        )
        assert "X_roll7_mean" in X.columns
        assert "X_roll7_std" in X.columns

    def test_feature_columns_contain_diffs(self, price_df: pd.DataFrame) -> None:
        X, _ = create_feature_matrix(
            price_df, target="equipo", diff_columns=["X"]
        )
        assert "X_diff1" in X.columns
        assert "X_diff2" in X.columns

    def test_custom_lags_and_windows(self, price_df: pd.DataFrame) -> None:
        X, y = create_feature_matrix(
            price_df,
            target="equipo",
            lag_columns=["X"],
            lags=[1],
            rolling_columns=["X"],
            windows=[3],
            diff_columns=[],
        )
        assert len(X) == len(price_df) - 2  # warmup = max(lag=1, window=3) - 1 = 2
        assert not X.isna().any().any()

    def test_raises_on_missing_target(self, price_df: pd.DataFrame) -> None:
        with pytest.raises(ValueError, match="objetivo"):
            create_feature_matrix(price_df, target="inexistente")

    def test_indices_are_datetime(self, price_df: pd.DataFrame) -> None:
        X, y = create_feature_matrix(price_df, target="equipo")
        assert isinstance(X.index, pd.DatetimeIndex)
        assert isinstance(y.index, pd.DatetimeIndex)
