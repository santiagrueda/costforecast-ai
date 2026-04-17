"""
Modelo de persistencia (baseline).

Predice el último valor observado durante el entrenamiento para todos los
horizontes. Es el benchmark mínimo que cualquier modelo real debe superar.

Si un modelo no mejora la persistencia, no tiene valor predictivo.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from costforecast.logger import logger


class PersistenceModel:
    """
    Predice siempre el último valor visto en entrenamiento.

    Uso:
        model = PersistenceModel()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
    """

    def __init__(self) -> None:
        self._last_value: float | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "PersistenceModel":
        """Memoriza el último valor de la serie objetivo."""
        if len(y) == 0:
            raise ValueError("y no puede estar vacío")
        self._last_value = float(y.iloc[-1])
        logger.info("PersistenceModel entrenado: último valor = {:.4f}", self._last_value)
        return self

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """Devuelve el último valor entrenado para cada fila de X."""
        if self._last_value is None:
            raise RuntimeError("Llama a fit() antes de predict()")
        preds = np.full(len(X), self._last_value)
        return pd.Series(preds, index=X.index, name="prediction")

    @property
    def last_value(self) -> float:
        if self._last_value is None:
            raise RuntimeError("Modelo no entrenado")
        return self._last_value
