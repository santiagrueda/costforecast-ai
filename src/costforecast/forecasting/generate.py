"""
Generación de pronósticos con intervalos de confianza. Invocado por `make forecast`.

Carga los modelos SARIMAX entrenados (o re-entrena si no existen), genera un
pronóstico de HORIZON_DAYS días hábiles hacia adelante, calcula intervalos de
confianza vía simulación Monte Carlo y persiste los resultados como CSV y Parquet.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from costforecast.config import settings
from costforecast.data.consolidator import build_consolidated_dataset
from costforecast.logger import logger
from costforecast.models.sarimax_model import SARIMAXModel

EXOG_COLS: list[str] = ["Price_X", "Price_Y", "Price_Z"]
TARGETS: list[str] = ["Price_Equipo1", "Price_Equipo2"]
MODELS_DIR: Path = settings.processed_data_dir / "models"
HORIZON_DAYS: int = 20
N_SIMULATIONS: int = 1000


def _load_dataset() -> pd.DataFrame:
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


def _load_or_fit_sarimax(target: str, df: pd.DataFrame) -> SARIMAXModel:
    model_path = MODELS_DIR / f"sarimax_{target}.pkl"
    if model_path.exists():
        logger.info("Cargando SARIMAX para {} desde {}", target, model_path.name)
        return joblib.load(model_path)
    logger.info("SARIMAX no encontrado para {} — entrenando ahora", target)
    model = SARIMAXModel(order=(1, 1, 1))
    model.fit(df[EXOG_COLS], df[target])
    return model


def _monte_carlo_intervals(
    point_forecast: np.ndarray,
    residual_std: float,
    n_simulations: int = 1000,
    alpha: float = 0.05,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Intervalo de confianza (1-alpha) vía simulación Monte Carlo.

    Supone errores normales con desviación estándar `residual_std` que crece
    proporcionalmente a la raíz del horizonte (random walk uncertainty).
    """
    h = len(point_forecast)
    rng = np.random.default_rng(settings.random_seed)
    noise_scale = residual_std * np.sqrt(np.arange(1, h + 1))

    sims = point_forecast[np.newaxis, :] + rng.normal(
        0, noise_scale, size=(n_simulations, h)
    )
    lower = np.percentile(sims, 100 * alpha / 2, axis=0)
    upper = np.percentile(sims, 100 * (1 - alpha / 2), axis=0)
    return lower, upper


def generate_forecasts(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """
    Genera pronósticos de HORIZON_DAYS días hábiles para todos los targets.

    Args:
        df: Dataset histórico consolidado con DatetimeIndex.

    Returns:
        Diccionario {target: DataFrame} donde cada DataFrame contiene las columnas:
        forecast, lower_95, upper_95, last_actual.
    """
    last_date = df.index[-1]
    future_dates = pd.bdate_range(
        start=last_date + pd.offsets.BDay(1), periods=HORIZON_DAYS
    )
    last_exog = df[EXOG_COLS].iloc[-1]
    X_future = pd.DataFrame(
        np.tile(last_exog.values, (HORIZON_DAYS, 1)),
        columns=EXOG_COLS,
        index=future_dates,
    )

    results: dict[str, pd.DataFrame] = {}

    for target in TARGETS:
        logger.info("Generando pronóstico para {}", target)

        model = _load_or_fit_sarimax(target, df)
        point_series = model.predict(X_future)
        point = point_series.values

        # Residuos in-sample para estimar dispersión
        X_in = df[EXOG_COLS]
        fitted = model.predict(X_in)
        residuals = df[target].values - fitted.values
        residual_std = float(np.nanstd(residuals))

        lower, upper = _monte_carlo_intervals(point, residual_std, N_SIMULATIONS)

        last_actual = float(df[target].iloc[-1])

        fc_df = pd.DataFrame(
            {
                "target": target,
                "forecast": point,
                "lower_95": lower,
                "upper_95": upper,
                "last_actual": last_actual,
            },
            index=future_dates,
        )
        fc_df.index.name = "date"
        results[target] = fc_df

        logger.info(
            "  {} días proyectados | media={:.2f} | IC=[{:.2f}, {:.2f}]",
            HORIZON_DAYS,
            float(np.mean(point)),
            float(np.mean(lower)),
            float(np.mean(upper)),
        )

    return results


def save_forecasts(results: dict[str, pd.DataFrame]) -> None:
    """Persiste todos los pronósticos como CSV y Parquet en data/forecasts/."""
    out_dir = settings.forecasts_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    all_fc = pd.concat(results.values())
    all_fc.index.name = "date"

    csv_path = out_dir / "forecasts.csv"
    parquet_path = out_dir / "forecasts.parquet"

    all_fc.to_csv(csv_path)
    all_fc.to_parquet(parquet_path, compression="snappy")

    logger.info("Pronósticos guardados:")
    logger.info("  CSV    → {}", csv_path)
    logger.info("  Parquet → {}", parquet_path)


def main() -> None:
    logger.info("=== CostForecast AI — Generación de pronósticos ===")
    df = _load_dataset()

    results = generate_forecasts(df)
    save_forecasts(results)

    print("\n=== Pronósticos generados ===\n")
    for target, fc in results.items():
        print(f"\n### {target}")
        print(fc.round(2).to_string())

    logger.info("=== Forecasting completado ===")


if __name__ == "__main__":
    main()
