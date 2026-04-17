"""
XGBoost para forecasting de precios de equipos.

Envuelve XGBRegressor de xgboost en la interfaz fit/predict del proyecto.
Está pensado para recibir directamente la salida de create_feature_matrix()
del módulo features.engineering.

Decisiones de diseño:
- Expone feature_importances_ tras el fit para uso con SHAP.
- Acepta early_stopping_rounds opcionalmente; si se pasa eval_set en
  fit_kwargs, activa el early stopping de XGBoost de forma transparente.
- Los parámetros por defecto están orientados a series de tiempo:
  subsample < 1 y colsample_bytree < 1 para regularización ligera.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from costforecast.logger import logger


_DEFAULTS: dict[str, Any] = {
    "n_estimators": 500,
    "learning_rate": 0.05,
    "max_depth": 4,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
    "verbosity": 0,
}


class XGBoostModel:
    """
    XGBRegressor con interfaz fit/predict estándar del proyecto.

    Args:
        early_stopping_rounds: Paradas tempranas si se provee eval_set en fit_kwargs.
            Si es None, se desactiva early stopping.
        fit_kwargs:             Argumentos adicionales para XGBRegressor.fit()
                                (ej. eval_set, verbose).
        **xgb_kwargs:           Parámetros del modelo (sobreescriben defaults).

    Uso:
        model = XGBoostModel(n_estimators=300, max_depth=3)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        importances = model.feature_importances_  # pd.Series
    """

    def __init__(
        self,
        early_stopping_rounds: int | None = None,
        fit_kwargs: dict[str, Any] | None = None,
        **xgb_kwargs: Any,
    ) -> None:
        self.early_stopping_rounds = early_stopping_rounds
        self.fit_kwargs = fit_kwargs or {}

        params = {**_DEFAULTS, **xgb_kwargs}
        if early_stopping_rounds is not None:
            params["early_stopping_rounds"] = early_stopping_rounds

        self._model = XGBRegressor(**params)
        self._feature_names: list[str] = []

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "XGBoostModel":
        """
        Entrena el modelo.

        Args:
            X: Matriz de features (salida de create_feature_matrix).
            y: Serie objetivo alineada con X.
        """
        if len(X) == 0:
            raise ValueError("X no puede estar vacío")
        if len(X) != len(y):
            raise ValueError(f"X ({len(X)} filas) e y ({len(y)} filas) deben tener el mismo largo")

        self._feature_names = list(X.columns)

        logger.info(
            "XGBoostModel.fit: {} obs, {} features",
            len(X),
            len(self._feature_names),
        )

        self._model.fit(X, y, **self.fit_kwargs)
        try:
            iteration = self._model.best_iteration
        except AttributeError:
            iteration = self._model.n_estimators
        logger.info("XGBoostModel entrenado (n_estimators={})", iteration)
        return self

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """
        Genera predicciones.

        Args:
            X: DataFrame con las mismas columnas que se usaron en fit().

        Returns:
            Serie de predicciones con el mismo índice que X.
        """
        self._check_fitted()
        self._validate_feature_columns(X)

        logger.debug("XGBoostModel.predict: {} filas", len(X))
        preds = self._model.predict(X[self._feature_names])
        return pd.Series(preds, index=X.index, name="prediction")

    @property
    def feature_importances_(self) -> pd.Series:
        """Importancia de features (gain) ordenada de mayor a menor."""
        self._check_fitted()
        importances = self._model.feature_importances_
        return (
            pd.Series(importances, index=self._feature_names, name="importance")
            .sort_values(ascending=False)
        )

    @property
    def booster(self):
        """Acceso al booster nativo de XGBoost (útil para SHAP)."""
        self._check_fitted()
        return self._model.get_booster()

    # ------------------------------------------------------------------
    # Helpers privados
    # ------------------------------------------------------------------

    def _check_fitted(self) -> None:
        if not self._feature_names:
            raise RuntimeError("Llama a fit() antes de predict()")

    def _validate_feature_columns(self, X: pd.DataFrame) -> None:
        provided = set(X.columns)
        expected = set(self._feature_names)
        missing = expected - provided
        if missing:
            raise ValueError(f"Columnas faltantes en X: {missing}")
