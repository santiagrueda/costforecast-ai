"""
Prophet con regressors externos.

Envuelve facebook/prophet en la interfaz fit/predict del proyecto.

Decisiones de diseño:
- fit(X, y): X tiene DatetimeIndex y columnas que se agregan como regressors.
  y es la serie objetivo. Internamente se construye el DataFrame {ds, y, ...regs}.
- predict(X): X tiene DatetimeIndex y las mismas columnas de regressors usadas
  en fit(). Se construye el DataFrame {ds, ...regs} que Prophet espera.
- Se suprimen los logs de cmdstanpy/prophet para no contaminar la consola del
  proyecto (Prophet usa Stan internamente y es muy verboso por defecto).
"""

from __future__ import annotations

import logging

import pandas as pd

from costforecast.logger import logger as project_logger

# Prophet y Stan son muy verbosos; silenciarlos aquí para no contaminar
# el logger del proyecto (loguru). Se hace a nivel de módulo, una sola vez.
logging.getLogger("prophet").setLevel(logging.WARNING)
logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
logging.getLogger("pystan").setLevel(logging.WARNING)


class ProphetModel:
    """
    Prophet con soporte para regressors exógenos.

    Args:
        yearly_seasonality:  Activar estacionalidad anual (bool o int para términos Fourier).
        weekly_seasonality:  Activar estacionalidad semanal.
        daily_seasonality:   Activar estacionalidad diaria.
        prophet_kwargs:      Parámetros adicionales para el constructor de Prophet.

    Uso:
        model = ProphetModel(yearly_seasonality=True)
        model.fit(X_train, y_train)     # X_train: DataFrame con DatetimeIndex
        y_pred = model.predict(X_test)
    """

    def __init__(
        self,
        yearly_seasonality: bool | int = "auto",
        weekly_seasonality: bool | int = "auto",
        daily_seasonality: bool | int = "auto",
        **prophet_kwargs,
    ) -> None:
        self.yearly_seasonality = yearly_seasonality
        self.weekly_seasonality = weekly_seasonality
        self.daily_seasonality = daily_seasonality
        self.prophet_kwargs = prophet_kwargs

        self._model = None
        self._regressor_cols: list[str] = []

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "ProphetModel":
        """
        Ajusta Prophet.

        Args:
            X: DataFrame con DatetimeIndex. Cada columna se agrega como regressor.
               Puede tener 0 columnas (Prophet puro sin regressors).
            y: Serie objetivo alineada con X.
        """
        if len(y) == 0:
            raise ValueError("y no puede estar vacío")
        if len(X) != len(y):
            raise ValueError(f"X ({len(X)}) e y ({len(y)}) deben tener el mismo largo")

        # Import diferido para evitar tiempo de carga cuando no se usa este modelo
        from prophet import Prophet  # noqa: PLC0415

        self._regressor_cols = list(X.columns)

        self._model = Prophet(
            yearly_seasonality=self.yearly_seasonality,
            weekly_seasonality=self.weekly_seasonality,
            daily_seasonality=self.daily_seasonality,
            **self.prophet_kwargs,
        )

        for col in self._regressor_cols:
            self._model.add_regressor(col)

        train_df = pd.DataFrame({"ds": X.index, "y": y.values})
        for col in self._regressor_cols:
            train_df[col] = X[col].values

        project_logger.info(
            "ProphetModel.fit: {} obs, {} regressor(es): {}",
            len(train_df),
            len(self._regressor_cols),
            self._regressor_cols,
        )

        self._model.fit(train_df)
        project_logger.info("ProphetModel entrenado")
        return self

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """
        Genera predicciones para las fechas en X.

        Args:
            X: DataFrame con DatetimeIndex y las mismas columnas de regressors
               usadas en fit(). Las filas pueden ser en-muestra o fuera de muestra.

        Returns:
            Serie con las predicciones (yhat de Prophet).
        """
        self._check_fitted()
        self._validate_regressor_columns(X)

        pred_df = pd.DataFrame({"ds": X.index})
        for col in self._regressor_cols:
            pred_df[col] = X[col].values

        project_logger.debug("ProphetModel.predict: {} pasos", len(pred_df))
        forecast = self._model.predict(pred_df)
        return pd.Series(forecast["yhat"].values, index=X.index, name="prediction")

    # ------------------------------------------------------------------
    # Helpers privados
    # ------------------------------------------------------------------

    def _check_fitted(self) -> None:
        if self._model is None:
            raise RuntimeError("Llama a fit() antes de predict()")

    def _validate_regressor_columns(self, X: pd.DataFrame) -> None:
        provided = set(X.columns)
        expected = set(self._regressor_cols)
        if provided != expected:
            raise ValueError(
                f"Columnas de regressors incorrectas. Esperadas: {expected}, recibidas: {provided}"
            )
