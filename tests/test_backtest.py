"""
Tests para src/costforecast/evaluation/backtest.py.

Estrategia:
- Tests unitarios de funciones puras (_ape, _ae, _se, _split_positions,
  métricas del report) → rápidos, sin modelos reales.
- Tests de integración end-to-end sobre un dataset sintético pequeño con
  n_splits=2, min_train_size=40, horizons=[1,3] y modelos fast. Verifican
  que run_backtest termina y que el report tiene la forma correcta.
- Se evita testear los modelos en detalle (ya tienen sus propios tests).
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from costforecast.evaluation.backtest import (
    BacktestConfig,
    BacktestReport,
    ModelSpec,
    _ape,
    _ae,
    _se,
    _split_positions,
    default_model_specs,
    run_backtest,
    _prep_baseline,
    _prep_sarimax,
    _prep_xgboost,
)
from costforecast.models.baseline import PersistenceModel
from costforecast.models.xgboost_model import XGBoostModel


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tiny_df() -> pd.DataFrame:
    """DataFrame sintético con 100 filas diarias: 3 materias primas + 2 equipos."""
    rng = np.random.default_rng(42)
    n = 100
    idx = pd.bdate_range("2020-01-01", periods=n)
    x = 80 + np.cumsum(rng.normal(0, 0.5, n))
    y = 500 + np.cumsum(rng.normal(0, 1, n))
    z = 2000 + np.cumsum(rng.normal(0, 2, n))
    e1 = 0.4 * x + 0.3 * z + rng.normal(0, 5, n)
    e2 = 0.2 * x + 0.5 * z + rng.normal(0, 5, n)
    return pd.DataFrame(
        {"Price_X": x, "Price_Y": y, "Price_Z": z,
         "Price_Equipo1": e1, "Price_Equipo2": e2},
        index=idx,
    )


@pytest.fixture
def fast_config() -> BacktestConfig:
    """Config mínima para tests rápidos."""
    return BacktestConfig(
        horizons=[1, 3],
        n_splits=2,
        min_train_size=40,
        xgb_lags=[1, 2],
        xgb_windows=[3],
    )


@pytest.fixture
def baseline_only_spec() -> list[ModelSpec]:
    """Un único modelo baseline para tests de integración rápidos."""
    return [
        ModelSpec(
            name="Persistence",
            factory=PersistenceModel,
            prep_fn=_prep_baseline,
        )
    ]


@pytest.fixture
def two_model_specs() -> list[ModelSpec]:
    """Baseline + XGBoost fast para integración sin SARIMAX/Prophet lentos."""
    return [
        ModelSpec(name="Persistence", factory=PersistenceModel, prep_fn=_prep_baseline),
        ModelSpec(
            name="XGBoost",
            factory=lambda: XGBoostModel(n_estimators=5),
            prep_fn=_prep_xgboost,
        ),
    ]


# ---------------------------------------------------------------------------
# Tests unitarios de métricas
# ---------------------------------------------------------------------------

class TestMetricFunctions:
    def test_ape_perfect_prediction(self) -> None:
        assert _ape(100.0, 100.0) == pytest.approx(0.0)

    def test_ape_ten_percent_error(self) -> None:
        assert _ape(100.0, 90.0) == pytest.approx(10.0)

    def test_ape_nan_when_actual_zero(self) -> None:
        assert math.isnan(_ape(0.0, 5.0))

    def test_ape_nan_when_non_finite(self) -> None:
        assert math.isnan(_ape(float("nan"), 5.0))
        assert math.isnan(_ape(5.0, float("nan")))

    def test_ae_absolute_value(self) -> None:
        assert _ae(100.0, 90.0) == pytest.approx(10.0)
        assert _ae(90.0, 100.0) == pytest.approx(10.0)

    def test_ae_nan_propagation(self) -> None:
        assert math.isnan(_ae(float("nan"), 5.0))

    def test_se_squared_error(self) -> None:
        assert _se(100.0, 90.0) == pytest.approx(100.0)

    def test_se_nan_propagation(self) -> None:
        assert math.isnan(_se(float("nan"), 5.0))


# ---------------------------------------------------------------------------
# Tests de _split_positions
# ---------------------------------------------------------------------------

class TestSplitPositions:
    def test_returns_correct_count(self) -> None:
        positions = _split_positions(n=200, min_train=50, max_horizon=10, n_splits=5)
        assert len(positions) == 5

    def test_first_position_is_min_train(self) -> None:
        positions = _split_positions(n=200, min_train=50, max_horizon=10, n_splits=5)
        assert positions[0] == 50

    def test_last_position_respects_max_horizon(self) -> None:
        positions = _split_positions(n=200, min_train=50, max_horizon=10, n_splits=5)
        assert positions[-1] <= 200 - 10

    def test_positions_are_monotonic(self) -> None:
        positions = _split_positions(n=200, min_train=50, max_horizon=10, n_splits=5)
        assert all(a <= b for a, b in zip(positions, positions[1:]))

    def test_raises_when_dataset_too_short(self) -> None:
        with pytest.raises(ValueError, match="demasiado corto"):
            _split_positions(n=30, min_train=50, max_horizon=10, n_splits=3)

    def test_single_split(self) -> None:
        positions = _split_positions(n=100, min_train=50, max_horizon=5, n_splits=1)
        assert len(positions) == 1


# ---------------------------------------------------------------------------
# Tests de BacktestConfig
# ---------------------------------------------------------------------------

class TestBacktestConfig:
    def test_default_config_valid(self) -> None:
        cfg = BacktestConfig()
        assert cfg.max_horizon == max(cfg.horizons)

    def test_max_horizon_property(self) -> None:
        cfg = BacktestConfig(horizons=[1, 5, 20])
        assert cfg.max_horizon == 20

    def test_raises_empty_horizons(self) -> None:
        with pytest.raises(ValueError, match="horizons"):
            BacktestConfig(horizons=[])

    def test_raises_n_splits_zero(self) -> None:
        with pytest.raises(ValueError, match="n_splits"):
            BacktestConfig(n_splits=0)

    def test_raises_min_train_too_small(self) -> None:
        with pytest.raises(ValueError, match="min_train_size"):
            BacktestConfig(min_train_size=5)


# ---------------------------------------------------------------------------
# Tests de funciones de preparación de datos
# ---------------------------------------------------------------------------

class TestPrepFunctions:
    @pytest.fixture
    def split_dfs(self, tiny_df) -> tuple[pd.DataFrame, pd.DataFrame]:
        return tiny_df.iloc[:60], tiny_df.iloc[60:65]

    def test_prep_baseline_empty_X(self, split_dfs) -> None:
        train, test = split_dfs
        X_tr, y_tr, X_te = _prep_baseline(train, test, "Price_Equipo1", [], pd.DataFrame(), pd.Series(dtype=float))
        assert X_tr.shape[1] == 0
        assert X_te.shape[1] == 0
        assert len(y_tr) == len(train)

    def test_prep_sarimax_exog_columns(self, split_dfs) -> None:
        train, test = split_dfs
        exog = ["Price_X", "Price_Y"]
        X_tr, y_tr, X_te = _prep_sarimax(train, test, "Price_Equipo1", exog, pd.DataFrame(), pd.Series(dtype=float))
        assert list(X_tr.columns) == exog
        assert list(X_te.columns) == exog

    def test_prep_sarimax_aligned_lengths(self, split_dfs) -> None:
        train, test = split_dfs
        X_tr, y_tr, X_te = _prep_sarimax(train, test, "Price_Equipo1", ["Price_X"], pd.DataFrame(), pd.Series(dtype=float))
        assert len(X_tr) == len(y_tr)
        assert len(X_te) == len(test)

    def test_prep_xgboost_uses_feature_matrix(self, tiny_df) -> None:
        from costforecast.features.engineering import create_feature_matrix
        X_full, y_full = create_feature_matrix(tiny_df, target="Price_Equipo1", lags=[1], windows=[3])
        train = tiny_df.iloc[:60]
        test = tiny_df.iloc[60:65]
        X_tr, y_tr, X_te = _prep_xgboost(train, test, "Price_Equipo1", [], X_full, y_full)
        assert not X_tr.empty
        assert len(X_tr) == len(y_tr)
        assert set(X_tr.columns) == set(X_full.columns)


# ---------------------------------------------------------------------------
# Tests de BacktestReport
# ---------------------------------------------------------------------------

class TestBacktestReport:
    @pytest.fixture
    def sample_records(self) -> list[dict]:
        records = []
        for fold in range(3):
            for model in ["A", "B"]:
                for h in [1, 5]:
                    actual = 100.0
                    pred = actual * (0.95 if model == "A" else 0.90)
                    records.append({
                        "model": model, "target": "Equipo1",
                        "fold": fold, "train_end": "2020-01-01",
                        "horizon": h, "actual": actual, "predicted": pred,
                        "APE": _ape(actual, pred),
                        "AE": _ae(actual, pred),
                        "SE": _se(actual, pred),
                    })
        return records

    def test_raw_returns_dataframe(self, sample_records) -> None:
        report = BacktestReport(sample_records)
        assert isinstance(report.raw, pd.DataFrame)
        assert len(report.raw) == len(sample_records)

    def test_summary_has_expected_columns(self, sample_records) -> None:
        report = BacktestReport(sample_records)
        expected = {"model", "target", "horizon", "MAPE", "RMSE", "MAE", "n_folds"}
        assert expected.issubset(set(report.summary.columns))

    def test_summary_groups_correctly(self, sample_records) -> None:
        report = BacktestReport(sample_records)
        # 2 models × 1 target × 2 horizons = 4 rows
        assert len(report.summary) == 4

    def test_mape_values_correct(self, sample_records) -> None:
        report = BacktestReport(sample_records)
        row_a = report.summary[(report.summary["model"] == "A") & (report.summary["horizon"] == 1)]
        assert row_a["MAPE"].iloc[0] == pytest.approx(5.0, abs=1e-3)

    def test_rmse_is_sqrt_mean_se(self, sample_records) -> None:
        report = BacktestReport(sample_records)
        row_b = report.summary[(report.summary["model"] == "B") & (report.summary["horizon"] == 1)]
        expected_rmse = math.sqrt(100.0)  # actual=100, pred=90 → SE=100
        assert row_b["RMSE"].iloc[0] == pytest.approx(expected_rmse, abs=1e-3)

    def test_n_folds_count(self, sample_records) -> None:
        report = BacktestReport(sample_records)
        # 3 folds per model/horizon combination
        assert (report.summary["n_folds"] == 3).all()

    def test_to_markdown_contains_targets(self, sample_records) -> None:
        report = BacktestReport(sample_records)
        md = report.to_markdown()
        assert "Equipo1" in md
        assert "MAPE" in md
        assert "RMSE" in md
        assert "MAE" in md

    def test_to_markdown_contains_model_names(self, sample_records) -> None:
        report = BacktestReport(sample_records)
        md = report.to_markdown()
        assert "| A |" in md
        assert "| B |" in md

    def test_empty_report(self) -> None:
        report = BacktestReport([])
        assert report.raw.empty
        assert report.summary.empty

    def test_raw_returns_copy(self, sample_records) -> None:
        report = BacktestReport(sample_records)
        raw1 = report.raw
        raw1["new_col"] = 0
        assert "new_col" not in report.raw.columns


# ---------------------------------------------------------------------------
# Tests de integración — run_backtest (modelos rápidos)
# ---------------------------------------------------------------------------

class TestRunBacktest:
    def test_returns_report(self, tiny_df, fast_config, baseline_only_spec) -> None:
        report = run_backtest(tiny_df, targets=["Price_Equipo1"], config=fast_config, model_specs=baseline_only_spec)
        assert isinstance(report, BacktestReport)

    def test_report_not_empty(self, tiny_df, fast_config, baseline_only_spec) -> None:
        report = run_backtest(tiny_df, targets=["Price_Equipo1"], config=fast_config, model_specs=baseline_only_spec)
        assert not report.raw.empty

    def test_report_has_all_horizons(self, tiny_df, fast_config, baseline_only_spec) -> None:
        report = run_backtest(tiny_df, targets=["Price_Equipo1"], config=fast_config, model_specs=baseline_only_spec)
        assert set(report.raw["horizon"].unique()) == set(fast_config.horizons)

    def test_report_has_correct_models(self, tiny_df, fast_config, baseline_only_spec) -> None:
        report = run_backtest(tiny_df, targets=["Price_Equipo1"], config=fast_config, model_specs=baseline_only_spec)
        assert set(report.raw["model"].unique()) == {"Persistence"}

    def test_report_has_correct_targets(self, tiny_df, fast_config, baseline_only_spec) -> None:
        report = run_backtest(
            tiny_df,
            targets=["Price_Equipo1", "Price_Equipo2"],
            config=fast_config,
            model_specs=baseline_only_spec,
        )
        assert set(report.raw["target"].unique()) == {"Price_Equipo1", "Price_Equipo2"}

    def test_summary_no_nan_mape_for_baseline(self, tiny_df, fast_config, baseline_only_spec) -> None:
        report = run_backtest(tiny_df, targets=["Price_Equipo1"], config=fast_config, model_specs=baseline_only_spec)
        assert not report.summary["MAPE"].isna().any()

    def test_two_models_xgboost(self, tiny_df, fast_config, two_model_specs) -> None:
        report = run_backtest(tiny_df, targets=["Price_Equipo1"], config=fast_config, model_specs=two_model_specs)
        assert set(report.raw["model"].unique()) == {"Persistence", "XGBoost"}
        assert not report.summary.empty

    def test_default_targets_used_when_none(self, tiny_df, fast_config, baseline_only_spec) -> None:
        report = run_backtest(tiny_df, targets=None, config=fast_config, model_specs=baseline_only_spec)
        assert "Price_Equipo1" in report.raw["target"].values
        assert "Price_Equipo2" in report.raw["target"].values

    def test_n_folds_in_summary_matches_config(self, tiny_df, fast_config, baseline_only_spec) -> None:
        report = run_backtest(tiny_df, targets=["Price_Equipo1"], config=fast_config, model_specs=baseline_only_spec)
        assert (report.summary["n_folds"] == fast_config.n_splits).all()


# ---------------------------------------------------------------------------
# Tests de default_model_specs
# ---------------------------------------------------------------------------

class TestDefaultModelSpecs:
    def test_returns_four_specs(self) -> None:
        specs = default_model_specs()
        assert len(specs) == 4

    def test_all_have_names(self) -> None:
        specs = default_model_specs()
        assert all(spec.name for spec in specs)

    def test_factory_returns_fresh_instance(self) -> None:
        specs = default_model_specs()
        for spec in specs:
            m1 = spec.factory()
            m2 = spec.factory()
            assert m1 is not m2
