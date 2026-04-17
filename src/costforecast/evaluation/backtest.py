"""
Walk-forward validation para comparar modelos de forecasting.

Implementa expanding-window cross-validation para series de tiempo:
cada fold aumenta la ventana de entrenamiento en `step_size` observaciones
y evalúa predicciones en los `max(horizons)` pasos siguientes.

Decisiones de diseño:
─────────────────────
- Oracle exog: en cada fold, SARIMAX, Prophet y XGBoost reciben los valores
  reales futuros de las materias primas como predictores. Esto aisla la
  capacidad del modelo para capturar la relación target↔exog, y hace la
  comparación entre modelos equitativa. En producción se usarían forecasts
  de las materias primas.

- XGBoost features precomputadas en el dataset completo: todos los features
  (lags, rolling) son estrictamente backward-looking (solo usan valores
  pasados en cada punto t), por lo que calcularlos sobre el dataset completo
  y luego partir por fecha no introduce leakage.

- n_splits evenly-spaced: los puntos de corte se distribuyen uniformemente
  entre min_train_size y el final del dataset para cubrir todo el periodo
  de evaluación, no solo la cola.

- Métricas almacenadas por fold: se guarda APE, AE y SE por separado para
  poder calcular MAPE = mean(APE), MAE = mean(AE) y RMSE = sqrt(mean(SE))
  correctamente al agregar entre folds (RMSE ≠ mean(RMSE por fold)).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import pandas as pd

from costforecast.features.engineering import create_feature_matrix
from costforecast.logger import logger
from costforecast.models.baseline import PersistenceModel
from costforecast.models.prophet_model import ProphetModel
from costforecast.models.sarimax_model import SARIMAXModel
from costforecast.models.xgboost_model import XGBoostModel

# ---------------------------------------------------------------------------
# Constantes del dominio
# ---------------------------------------------------------------------------

EXOG_COLS: list[str] = ["Price_X", "Price_Y", "Price_Z"]
TARGETS: list[str] = ["Price_Equipo1", "Price_Equipo2"]


# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------

@dataclass
class BacktestConfig:
    """Parámetros de la validación walk-forward."""

    horizons: list[int] = field(default_factory=lambda: [1, 5, 10, 20])
    n_splits: int = 10
    min_train_size: int = 500
    exog_cols: list[str] = field(default_factory=lambda: list(EXOG_COLS))

    # Feature engineering para XGBoost
    xgb_lags: list[int] = field(default_factory=lambda: [1, 2, 3, 4, 5])
    xgb_windows: list[int] = field(default_factory=lambda: [5, 10, 20])

    def __post_init__(self) -> None:
        if not self.horizons:
            raise ValueError("horizons no puede estar vacío")
        if self.n_splits < 1:
            raise ValueError("n_splits debe ser >= 1")
        if self.min_train_size < 10:
            raise ValueError("min_train_size debe ser >= 10")

    @property
    def max_horizon(self) -> int:
        return max(self.horizons)


# ---------------------------------------------------------------------------
# Especificación de modelos y funciones de preparación de datos
# ---------------------------------------------------------------------------

@dataclass
class ModelSpec:
    """Asocia un nombre, una fábrica de modelos y una función de preparación."""

    name: str
    # Devuelve una instancia fresca del modelo en cada fold
    factory: Callable[[], Any]
    # (df_train, df_test, target, exog_cols, X_full, y_full) → (X_train, y_train, X_test)
    prep_fn: Callable[
        [pd.DataFrame, pd.DataFrame, str, list[str], pd.DataFrame, pd.Series],
        tuple[pd.DataFrame, pd.Series, pd.DataFrame],
    ]


def _prep_baseline(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    target: str,
    exog_cols: list[str],
    X_full: pd.DataFrame,
    y_full: pd.Series,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    X_tr = pd.DataFrame(index=df_train.index)
    X_te = pd.DataFrame(index=df_test.index)
    return X_tr, df_train[target], X_te


def _prep_sarimax(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    target: str,
    exog_cols: list[str],
    X_full: pd.DataFrame,
    y_full: pd.Series,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    X_tr = df_train[exog_cols] if exog_cols else pd.DataFrame(index=df_train.index)
    X_te = df_test[exog_cols] if exog_cols else pd.DataFrame(index=df_test.index)
    return X_tr, df_train[target], X_te


def _prep_prophet(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    target: str,
    exog_cols: list[str],
    X_full: pd.DataFrame,
    y_full: pd.Series,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    X_tr = df_train[exog_cols] if exog_cols else pd.DataFrame(index=df_train.index)
    X_te = df_test[exog_cols] if exog_cols else pd.DataFrame(index=df_test.index)
    return X_tr, df_train[target], X_te


def _prep_xgboost(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    target: str,
    exog_cols: list[str],
    X_full: pd.DataFrame,
    y_full: pd.Series,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    train_idx = X_full.index.intersection(df_train.index)
    test_idx = X_full.index.intersection(df_test.index)
    return X_full.loc[train_idx], y_full.loc[train_idx], X_full.loc[test_idx]


def default_model_specs() -> list[ModelSpec]:
    """Devuelve las especificaciones de los 4 modelos con hiperparámetros por defecto."""
    return [
        ModelSpec(
            name="Persistence",
            factory=PersistenceModel,
            prep_fn=_prep_baseline,
        ),
        ModelSpec(
            name="SARIMAX(1,1,1)",
            factory=lambda: SARIMAXModel(order=(1, 1, 1)),
            prep_fn=_prep_sarimax,
        ),
        ModelSpec(
            name="Prophet",
            factory=lambda: ProphetModel(
                yearly_seasonality=True,
                weekly_seasonality=False,
                daily_seasonality=False,
                uncertainty_samples=0,
            ),
            prep_fn=_prep_prophet,
        ),
        ModelSpec(
            name="XGBoost",
            factory=lambda: XGBoostModel(n_estimators=300),
            prep_fn=_prep_xgboost,
        ),
    ]


# ---------------------------------------------------------------------------
# Métricas
# ---------------------------------------------------------------------------

def _ape(actual: float, pred: float) -> float:
    """Absolute percentage error (un solo par observado/predicho)."""
    if not math.isfinite(actual) or not math.isfinite(pred) or abs(actual) < 1e-10:
        return float("nan")
    return abs(actual - pred) / abs(actual) * 100.0


def _ae(actual: float, pred: float) -> float:
    return abs(actual - pred) if math.isfinite(actual) and math.isfinite(pred) else float("nan")


def _se(actual: float, pred: float) -> float:
    return (actual - pred) ** 2 if math.isfinite(actual) and math.isfinite(pred) else float("nan")


# ---------------------------------------------------------------------------
# Lógica central de walk-forward
# ---------------------------------------------------------------------------

def _split_positions(n: int, min_train: int, max_horizon: int, n_splits: int) -> list[int]:
    """
    Genera `n_splits` posiciones de corte uniformemente distribuidas entre
    [min_train, n - max_horizon]. Devuelve lista de enteros únicos.
    """
    lo = min_train
    hi = n - max_horizon
    if lo > hi:
        raise ValueError(
            f"Dataset demasiado corto: necesita al menos {lo + max_horizon} filas, tiene {n}"
        )
    positions = np.linspace(lo, hi, n_splits, dtype=int)
    return list(dict.fromkeys(positions.tolist()))  # dedup preservando orden


def _run_fold(
    fold_idx: int,
    split_pos: int,
    valid_df: pd.DataFrame,
    target: str,
    exog_cols: list[str],
    model_specs: list[ModelSpec],
    horizons: list[int],
    max_horizon: int,
    X_full: pd.DataFrame,
    y_full: pd.Series,
) -> list[dict]:
    """Ejecuta todos los modelos para un único fold y devuelve registros de resultados."""
    train_idx = valid_df.index[:split_pos]
    test_idx = valid_df.index[split_pos: split_pos + max_horizon]

    df_train = valid_df.loc[train_idx]
    df_test = valid_df.loc[test_idx]

    records: list[dict] = []

    for spec in model_specs:
        model_records: list[dict] = []
        try:
            model = spec.factory()
            X_tr, y_tr, X_te = spec.prep_fn(df_train, df_test, target, exog_cols, X_full, y_full)

            if len(X_te) == 0:
                raise ValueError("X_te vacío — horizonte mayor que datos disponibles en el fold")

            model.fit(X_tr, y_tr)
            preds = model.predict(X_te)
            actuals = df_test[target]

            for h in horizons:
                if h <= len(actuals) and h <= len(preds):
                    actual_h = float(actuals.iloc[h - 1])
                    pred_h = float(preds.iloc[h - 1])
                else:
                    actual_h = float("nan")
                    pred_h = float("nan")

                model_records.append({
                    "model": spec.name,
                    "target": target,
                    "fold": fold_idx,
                    "train_end": str(df_train.index[-1].date()),
                    "horizon": h,
                    "actual": actual_h,
                    "predicted": pred_h,
                    "APE": _ape(actual_h, pred_h),
                    "AE": _ae(actual_h, pred_h),
                    "SE": _se(actual_h, pred_h),
                })

        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Fold {} — {}/{} falló: {}", fold_idx, spec.name, target, exc
            )
            for h in horizons:
                model_records.append({
                    "model": spec.name, "target": target,
                    "fold": fold_idx, "train_end": "",
                    "horizon": h,
                    "actual": float("nan"), "predicted": float("nan"),
                    "APE": float("nan"), "AE": float("nan"), "SE": float("nan"),
                })

        records.extend(model_records)

    return records


# ---------------------------------------------------------------------------
# BacktestReport
# ---------------------------------------------------------------------------

class BacktestReport:
    """
    Contenedor de resultados del backtest.

    Atributos:
        raw:     DataFrame con un registro por (fold, modelo, target, horizonte).
        summary: Métricas agregadas (MAPE, RMSE, MAE) por (modelo, target, horizonte).
    """

    def __init__(self, records: list[dict]) -> None:
        self._raw = pd.DataFrame(records)

    @property
    def raw(self) -> pd.DataFrame:
        return self._raw.copy()

    @property
    def summary(self) -> pd.DataFrame:
        """
        DataFrame con MAPE (%), RMSE y MAE por (model, target, horizon).
        RMSE se calcula correctamente como sqrt(mean(SE)) en lugar de mean(sqrt(SE)).
        """
        if self._raw.empty:
            return pd.DataFrame(columns=["model", "target", "horizon", "MAPE", "RMSE", "MAE", "n_folds"])

        agg = (
            self._raw
            .groupby(["model", "target", "horizon"], sort=False)
            .agg(
                MAPE=("APE", "mean"),
                _mean_SE=("SE", "mean"),
                MAE=("AE", "mean"),
                n_folds=("fold", "count"),
            )
            .reset_index()
        )
        agg["RMSE"] = np.sqrt(agg["_mean_SE"])
        agg = agg.drop(columns=["_mean_SE"])

        col_order = ["model", "target", "horizon", "MAPE", "RMSE", "MAE", "n_folds"]
        return agg[col_order].round({"MAPE": 3, "RMSE": 4, "MAE": 4})

    def to_markdown(self) -> str:
        """
        Genera un reporte en Markdown con tablas de MAPE, RMSE y MAE
        por modelo, separadas por target y horizonte.
        """
        lines: list[str] = [
            "# Backtest — Walk-Forward Validation\n",
            f"- **Folds**: {self._raw['fold'].nunique()}",
            f"- **Horizontes evaluados**: {sorted(self._raw['horizon'].unique().tolist())} días",
            f"- **Modelos**: {', '.join(self._raw['model'].unique().tolist())}\n",
        ]

        summary = self.summary
        horizons = sorted(summary["horizon"].unique())

        for target in summary["target"].unique():
            lines.append(f"\n## {target}\n")
            sub = summary[summary["target"] == target].set_index("model")

            for metric, label in [("MAPE", "MAPE (%)"), ("RMSE", "RMSE"), ("MAE", "MAE")]:
                lines.append(f"### {label}\n")
                header = "| Modelo | " + " | ".join(f"h={h}" for h in horizons) + " |"
                separator = "|---|" + "|".join(["---"] * len(horizons)) + "|"
                lines.append(header)
                lines.append(separator)

                for model_name in self._raw["model"].unique():
                    row_vals: list[str] = []
                    for h in horizons:
                        mask = (summary["model"] == model_name) & \
                               (summary["target"] == target) & \
                               (summary["horizon"] == h)
                        val = summary.loc[mask, metric]
                        row_vals.append(f"{val.iloc[0]:.3f}" if not val.empty and not val.isna().all() else "—")
                    lines.append(f"| {model_name} | " + " | ".join(row_vals) + " |")
                lines.append("")

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Punto de entrada principal
# ---------------------------------------------------------------------------

def run_backtest(
    df: pd.DataFrame,
    targets: list[str] | None = None,
    config: BacktestConfig | None = None,
    model_specs: list[ModelSpec] | None = None,
) -> BacktestReport:
    """
    Ejecuta la validación walk-forward y devuelve un BacktestReport.

    Args:
        df:          DataFrame con DatetimeIndex y columnas de precios
                     (materias primas + equipos).
        targets:     Columnas objetivo. Por defecto: ["Price_Equipo1", "Price_Equipo2"].
        config:      Configuración del backtest. Por defecto: BacktestConfig().
        model_specs: Lista de ModelSpec a comparar.
                     Por defecto: default_model_specs() (los 4 modelos del proyecto).

    Returns:
        BacktestReport con resultados crudos y métricas agregadas.
    """
    cfg = config or BacktestConfig()
    specs = model_specs or default_model_specs()
    eval_targets = targets or TARGETS

    # Precomputar feature matrix de XGBoost (backward-looking, sin leakage)
    xgb_matrices: dict[str, tuple[pd.DataFrame, pd.Series]] = {}
    for target in eval_targets:
        X_full, y_full = create_feature_matrix(
            df,
            target=target,
            lags=cfg.xgb_lags,
            windows=cfg.xgb_windows,
        )
        xgb_matrices[target] = (X_full, y_full)

    # valid_df: filas con features XGBoost disponibles (post-warmup)
    first_valid_idx = next(iter(xgb_matrices.values()))[0].index[0]
    valid_df = df.loc[first_valid_idx:]

    n = len(valid_df)
    split_positions = _split_positions(n, cfg.min_train_size, cfg.max_horizon, cfg.n_splits)

    logger.info(
        "Backtest: {} targets × {} modelos × {} folds × {} horizontes",
        len(eval_targets), len(specs), len(split_positions), len(cfg.horizons),
    )

    all_records: list[dict] = []

    for target in eval_targets:
        X_full, y_full = xgb_matrices[target]

        for fold_idx, split_pos in enumerate(split_positions):
            logger.debug("  fold {}/{} — target={}", fold_idx + 1, len(split_positions), target)
            records = _run_fold(
                fold_idx=fold_idx,
                split_pos=split_pos,
                valid_df=valid_df,
                target=target,
                exog_cols=cfg.exog_cols,
                model_specs=specs,
                horizons=cfg.horizons,
                max_horizon=cfg.max_horizon,
                X_full=X_full,
                y_full=y_full,
            )
            all_records.extend(records)

    logger.info("Backtest completado: {} registros totales", len(all_records))
    return BacktestReport(all_records)
