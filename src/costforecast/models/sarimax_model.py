"""
SARIMAX con variables exógenas.

Envuelve statsmodels.SARIMAX en la interfaz fit/predict estándar del proyecto.

Decisiones de diseño:
- fit(X, y): X es la matriz de regresores con DatetimeIndex; y es la serie objetivo.
  Las columnas de X se almacenan para validar coherencia en predict().
- predict(X): detecta automáticamente si la petición es dentro de la muestra
  (usa fittedvalues) o fuera (usa get_forecast). Esto permite usar el mismo
  método en evaluación sobre test set y en producción.
- Se permite X vacío (DataFrame sin columnas) para ajustar ARIMA puro.
- disp=False en fit() suprime la salida iterativa de optimización de statsmodels.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

from costforecast.logger import logger


class SARIMAXModel:
    """
    SARIMAX con interfaz fit/predict compatible con el resto del proyecto.

    Args:
        order:          (p, d, q) — parte no estacional.
        seasonal_order: (P, D, Q, s) — parte estacional.
        fit_kwargs:     Argumentos adicionales para SARIMAXResults.fit().
    """

    def __init__(
        self,
        order: tuple[int, int, int] = (1, 1, 1),
        seasonal_order: tuple[int, int, int, int] = (0, 0, 0, 0),
        **fit_kwargs: Any,
    ) -> None:
        self.order = order
        self.seasonal_order = seasonal_order
        self.fit_kwargs = {"disp": False, **fit_kwargs}

        self._result: Any = None
        self._exog_cols: list[str] = []
        self._train_index: pd.DatetimeIndex | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "SARIMAXModel":
        """
        Ajusta el modelo SARIMAX.

        Args:
            X: Variables exógenas con DatetimeIndex. Puede ser un DataFrame
               sin columnas para ajustar ARIMA puro.
            y: Serie objetivo con DatetimeIndex alineado con X.
        """
        if len(y) == 0:
            raise ValueError("y no puede estar vacío")
        if len(X) != len(y):
            raise ValueError(f"X ({len(X)} filas) e y ({len(y)} filas) deben tener el mismo largo")

        self._exog_cols = list(X.columns)
        self._train_index = pd.DatetimeIndex(y.index)

        exog = X.values if self._exog_cols else None

        logger.info(
            "SARIMAXModel.fit: {} obs, order={}, seasonal_order={}, {} regresor(es)",
            len(y),
            self.order,
            self.seasonal_order,
            len(self._exog_cols),
        )

        model = SARIMAX(
            y,
            exog=exog,
            order=self.order,
            seasonal_order=self.seasonal_order,
        )
        self._result = model.fit(**self.fit_kwargs)
        logger.info("SARIMAXModel entrenado (AIC={:.2f})", self._result.aic)
        return self

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """
        Genera predicciones en-muestra o fuera de muestra.

        - Si el índice de X está dentro del rango de entrenamiento, devuelve
          los valores ajustados (fittedvalues) para esas fechas.
        - Si el índice de X comienza después del último dato de entrenamiento,
          produce un forecast de `len(X)` pasos hacia adelante.

        Args:
            X: DataFrame con DatetimeIndex. Las columnas deben coincidir con
               las usadas en fit(). Puede estar vacío (ARIMA puro).
        """
        self._check_fitted()
        self._validate_exog_columns(X)

        last_train = self._train_index[-1]  # type: ignore[index]
        first_pred = X.index[0]

        exog = X[self._exog_cols].values if self._exog_cols else None

        if first_pred > last_train:
            # Fuera de muestra
            steps = len(X)
            logger.debug("SARIMAXModel.predict: {} pasos fuera de muestra", steps)
            forecast = self._result.get_forecast(steps=steps, exog=exog)
            values = forecast.predicted_mean.values
        else:
            # En muestra: devolver fittedvalues para las fechas solicitadas
            logger.debug("SARIMAXModel.predict: predicción en muestra")
            fitted = self._result.fittedvalues
            # Reindexar al rango pedido, rellenar con NaN si hay fechas no vistas
            values = fitted.reindex(X.index).values

        return pd.Series(values, index=X.index, name="prediction")

    # ------------------------------------------------------------------
    # Helpers privados
    # ------------------------------------------------------------------

    def _check_fitted(self) -> None:
        if self._result is None:
            raise RuntimeError("Llama a fit() antes de predict()")

    def _validate_exog_columns(self, X: pd.DataFrame) -> None:
        provided = set(X.columns)
        expected = set(self._exog_cols)
        if provided != expected:
            raise ValueError(
                f"Columnas exógenas incorrectas. Esperadas: {expected}, recibidas: {provided}"
            )
