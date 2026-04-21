"""
Entrenamiento de todos los modelos sobre el dataset completo.
Invocado por `make train`.

Entrena Persistence, SARIMAX(1,1,1), Prophet y XGBoost para cada equipo,
persiste los modelos como archivos joblib en data/processed/models/,
y muestra un resumen de los modelos entrenados.
"""

from __future__ import annotations

import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import joblib
import pandas as pd

from costforecast.config import settings
from costforecast.data.consolidator import build_consolidated_dataset
from costforecast.features.engineering import create_feature_matrix
from costforecast.logger import logger
from costforecast.models.baseline import PersistenceModel
from costforecast.models.prophet_model import ProphetModel
from costforecast.models.sarimax_model import SARIMAXModel
from costforecast.models.xgboost_model import XGBoostModel

EXOG_COLS: list[str] = ["Price_X", "Price_Y", "Price_Z"]
TARGETS: list[str] = ["Price_Equipo1", "Price_Equipo2"]
MODELS_DIR: Path = settings.processed_data_dir / "models"

XGB_LAGS: list[int] = [1, 2, 3, 4, 5]
XGB_WINDOWS: list[int] = [5, 10, 20]


def _load_dataset() -> pd.DataFrame:
    """Carga el dataset procesado si existe; si no, consolida desde raw."""
    processed = settings.processed_dataset_path
    if processed.exists():
        logger.info("Cargando dataset procesado desde {}", processed)
        return pd.read_parquet(processed)
    logger.info("Dataset procesado no encontrado — cargando desde raw")
    raw = settings.raw_data_dir
    return build_consolidated_dataset(
        historico_path=raw / "historico_equipos.csv",
        x_path=raw / "X.csv",
        y_path=raw / "Y.csv",
        z_path=raw / "Z.csv",
    )


def train_and_save(df: pd.DataFrame) -> dict[str, dict[str, Path]]:
    """
    Entrena los 4 modelos para cada target y los persiste.

    Returns:
        Diccionario {target: {model_name: path}} con rutas de los modelos guardados.
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    saved: dict[str, dict[str, Path]] = {}

    for target in TARGETS:
        logger.info("─── Entrenando modelos para {} ───", target)
        saved[target] = {}

        # ── Baseline (Persistence) ──────────────────────────────────────────
        baseline = PersistenceModel()
        baseline.fit(pd.DataFrame(index=df.index), df[target])
        path = MODELS_DIR / f"baseline_{target}.pkl"
        joblib.dump(baseline, path)
        saved[target]["Persistence"] = path
        logger.info("  ✓ Baseline (último valor: {:.2f})", baseline.last_value)

        # ── SARIMAX(1,1,1) ──────────────────────────────────────────────────
        sarimax = SARIMAXModel(order=(1, 1, 1))
        sarimax.fit(df[EXOG_COLS], df[target])
        path = MODELS_DIR / f"sarimax_{target}.pkl"
        joblib.dump(sarimax, path)
        saved[target]["SARIMAX"] = path
        logger.info("  ✓ SARIMAX guardado")

        # ── Prophet ─────────────────────────────────────────────────────────
        prophet = ProphetModel(
            yearly_seasonality=True,
            weekly_seasonality=False,
            daily_seasonality=False,
            uncertainty_samples=0,
        )
        prophet.fit(df[EXOG_COLS], df[target])
        path = MODELS_DIR / f"prophet_{target}.pkl"
        joblib.dump(prophet, path)
        saved[target]["Prophet"] = path
        logger.info("  ✓ Prophet guardado")

        # ── XGBoost ─────────────────────────────────────────────────────────
        X, y = create_feature_matrix(
            df, target=target, lags=XGB_LAGS, windows=XGB_WINDOWS
        )
        xgb = XGBoostModel(n_estimators=500)
        xgb.fit(X, y)
        path = MODELS_DIR / f"xgboost_{target}.pkl"
        joblib.dump(xgb, path)
        saved[target]["XGBoost"] = path
        logger.info(
            "  ✓ XGBoost guardado ({} features, top: {})",
            len(X.columns),
            xgb.feature_importances_.index[0],
        )

    return saved


def print_summary(saved: dict[str, dict[str, Path]]) -> None:
    print("\n=== Modelos entrenados y guardados ===\n")
    for target, models in saved.items():
        print(f"  {target}:")
        for name, path in models.items():
            size_kb = path.stat().st_size / 1024
            print(f"    {name:12s} → {path.name} ({size_kb:.1f} KB)")
    print()


def main() -> None:
    logger.info("=== CostForecast AI — Entrenamiento de modelos ===")
    df = _load_dataset()
    logger.info("Dataset: {} filas × {} columnas", *df.shape)
    saved = train_and_save(df)
    print_summary(saved)
    logger.info("=== Entrenamiento completado ===")


if __name__ == "__main__":
    main()
