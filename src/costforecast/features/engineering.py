"""
Feature engineering para forecasting de precios de equipos.

Las cuatro funciones públicas generan features compatibles con XGBoost/LightGBM:
- create_lags:          rezagos de 1..n días sobre columnas seleccionadas
- create_rolling_stats: media y desvío estándar con ventanas configurables
- create_differences:   primera y segunda diferencia (series I(1) → estacionarias)
- create_feature_matrix: combina todo y devuelve X, y listos para entrenar

Decisión de diseño: todas las funciones reciben y devuelven DataFrames con
DatetimeIndex, sin modificar el original (copy-on-write). Las filas con NaN
introducidas por lags/rolling se eliminan solo en `create_feature_matrix`,
para que las funciones individuales sean componibles sin pérdida silenciosa de datos.
"""

from __future__ import annotations

import pandas as pd

from costforecast.logger import logger


def create_lags(
    df: pd.DataFrame,
    columns: list[str],
    lags: list[int],
) -> pd.DataFrame:
    """
    Agrega columnas con rezagos temporales.

    Args:
        df:      DataFrame con DatetimeIndex ordenado.
        columns: Columnas a las que aplicar los rezagos.
        lags:    Lista de enteros positivos (ej. [1, 2, 5]).

    Returns:
        DataFrame original más las columnas `<col>_lag<n>` para cada combinación.
    """
    if not lags:
        return df.copy()

    _validate_columns(df, columns, "create_lags")

    result = df.copy()
    for col in columns:
        for lag in lags:
            if lag <= 0:
                raise ValueError(f"Los lags deben ser enteros positivos, se recibió: {lag}")
            result[f"{col}_lag{lag}"] = result[col].shift(lag)

    n_new = len(columns) * len(lags)
    logger.debug("create_lags: {} columnas nuevas generadas", n_new)
    return result


def create_rolling_stats(
    df: pd.DataFrame,
    columns: list[str],
    windows: list[int],
) -> pd.DataFrame:
    """
    Agrega media y desvío estándar móviles.

    Args:
        df:      DataFrame con DatetimeIndex ordenado.
        columns: Columnas sobre las que calcular las estadísticas.
        windows: Tamaños de ventana en días (ej. [5, 10, 20]).

    Returns:
        DataFrame original más columnas `<col>_roll<w>_mean` y `<col>_roll<w>_std`.

    Note:
        Se usa `min_periods=window` para evitar estadísticas sobre ventanas parciales,
        lo que genera NaN en las primeras `window-1` filas de cada columna nueva.
    """
    if not windows:
        return df.copy()

    _validate_columns(df, columns, "create_rolling_stats")

    result = df.copy()
    for col in columns:
        for window in windows:
            if window <= 1:
                raise ValueError(f"Las ventanas deben ser > 1, se recibió: {window}")
            roller = result[col].rolling(window=window, min_periods=window)
            result[f"{col}_roll{window}_mean"] = roller.mean()
            result[f"{col}_roll{window}_std"] = roller.std()

    n_new = len(columns) * len(windows) * 2
    logger.debug("create_rolling_stats: {} columnas nuevas generadas", n_new)
    return result


def create_differences(
    df: pd.DataFrame,
    columns: list[str],
) -> pd.DataFrame:
    """
    Agrega primera y segunda diferencia de las columnas indicadas.

    La primera diferencia (Δ) convierte series I(1) a estacionarias.
    La segunda diferencia (Δ²) se incluye para capturar aceleración del precio,
    útil como feature de momentum en modelos de árbol.

    Args:
        df:      DataFrame con DatetimeIndex ordenado.
        columns: Columnas a diferenciar.

    Returns:
        DataFrame original más columnas `<col>_diff1` y `<col>_diff2`.
    """
    if not columns:
        return df.copy()

    _validate_columns(df, columns, "create_differences")

    result = df.copy()
    for col in columns:
        result[f"{col}_diff1"] = result[col].diff(1)
        result[f"{col}_diff2"] = result[col].diff(2)

    logger.debug("create_differences: {} columnas nuevas generadas", len(columns) * 2)
    return result


def create_feature_matrix(
    df: pd.DataFrame,
    target: str,
    lag_columns: list[str] | None = None,
    lags: list[int] | None = None,
    rolling_columns: list[str] | None = None,
    windows: list[int] | None = None,
    diff_columns: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Construye la matriz de features X y el vector objetivo y para XGBoost.

    Aplica en orden: lags → rolling stats → diferencias, luego elimina
    las filas con NaN (introducidas por los lookback de lags y rolling).

    Args:
        df:              DataFrame de entrada con DatetimeIndex.
        target:          Nombre de la columna objetivo.
        lag_columns:     Columnas para rezagos (default: todas menos `target`).
        lags:            Rezagos a aplicar (default: [1, 2, 3, 4, 5]).
        rolling_columns: Columnas para rolling stats (default: igual a lag_columns).
        windows:         Ventanas para rolling stats (default: [5, 10, 20]).
        diff_columns:    Columnas a diferenciar (default: igual a lag_columns).

    Returns:
        (X, y) donde X es el DataFrame de features sin NaN e y es la Serie objetivo
        alineada con el mismo índice.

    Raises:
        ValueError: si `target` no existe en `df`.
    """
    if target not in df.columns:
        raise ValueError(f"Columna objetivo '{target}' no encontrada en el DataFrame")

    predictors = [c for c in df.columns if c != target]

    lag_cols = lag_columns if lag_columns is not None else predictors
    lag_list = lags if lags is not None else [1, 2, 3, 4, 5]
    roll_cols = rolling_columns if rolling_columns is not None else lag_cols
    win_list = windows if windows is not None else [5, 10, 20]
    diff_cols = diff_columns if diff_columns is not None else lag_cols

    logger.info(
        "create_feature_matrix: target='{}', {} predictores base, lags={}, windows={}",
        target,
        len(predictors),
        lag_list,
        win_list,
    )

    enriched = create_lags(df, lag_cols, lag_list)
    enriched = create_rolling_stats(enriched, roll_cols, win_list)
    enriched = create_differences(enriched, diff_cols)

    # Eliminar filas con NaN (warmup de lags y rolling)
    n_before = len(enriched)
    enriched = enriched.dropna()
    n_dropped = n_before - len(enriched)
    if n_dropped:
        logger.debug(
            "create_feature_matrix: {} filas eliminadas por NaN (warmup de lags/rolling)",
            n_dropped,
        )

    feature_cols = [c for c in enriched.columns if c != target]
    X = enriched[feature_cols]
    y = enriched[target]

    logger.info(
        "Feature matrix lista: {} filas × {} features",
        len(X),
        len(X.columns),
    )
    return X, y


# ---------------------------------------------------------------------------
# Helpers internos
# ---------------------------------------------------------------------------

def _validate_columns(df: pd.DataFrame, columns: list[str], caller: str) -> None:
    missing = set(columns) - set(df.columns)
    if missing:
        raise ValueError(f"{caller}: columnas no encontradas en el DataFrame: {missing}")
