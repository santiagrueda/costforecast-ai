"""
SHAP wrapper para los modelos XGBoost del proyecto.

Encapsula la lógica de cálculo de valores de Shapley, generación de
resúmenes tabulares y exportación de resultados.

Uso típico:
    from costforecast.explainability import ShapExplainer
    explainer = ShapExplainer(model.booster, X_train)
    summary = explainer.summary(X_test, n_top=10)
    print(summary.to_markdown())
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from costforecast.logger import logger


@dataclass
class ShapSummary:
    """Resultado de un análisis SHAP sobre un conjunto de observaciones."""

    feature_importance: pd.DataFrame
    """DataFrame con columnas: feature, mean_abs_shap, mean_shap, direction."""

    base_value: float
    """Valor base del explainer (predicción media del modelo)."""

    n_samples: int
    """Número de observaciones usadas para el cálculo."""

    def to_markdown(self, n_top: int | None = None) -> str:
        """Genera una tabla Markdown con el ranking de importancia."""
        df = self.feature_importance
        if n_top is not None:
            df = df.head(n_top)

        lines = [
            f"## Importancia de features (SHAP)",
            f"_n_samples = {self.n_samples} | base_value = {self.base_value:.4f}_\n",
            "| # | Feature | |SHAP| medio | SHAP medio | Dirección |",
            "|---|---------|-----------|------------|-----------|",
        ]
        for rank, (_, row) in enumerate(df.iterrows(), 1):
            direction = "↑ Sube precio" if row["mean_shap"] > 0 else "↓ Baja precio"
            lines.append(
                f"| {rank} | {row['feature']} | "
                f"{row['mean_abs_shap']:.4f} | "
                f"{row['mean_shap']:+.4f} | "
                f"{direction} |"
            )
        return "\n".join(lines)

    def top_features(self, n: int = 10) -> list[str]:
        """Retorna los nombres de las N features más importantes."""
        return self.feature_importance.head(n)["feature"].tolist()


class ShapExplainer:
    """
    Wrapper de SHAP TreeExplainer para modelos XGBoost.

    Args:
        booster:       Booster nativo de XGBoost (model.booster de XGBoostModel).
        X_background:  DataFrame de referencia para el explainer (típicamente X_train).
    """

    def __init__(self, booster, X_background: pd.DataFrame) -> None:
        import shap  # lazy import — SHAP es pesado

        self._explainer = shap.TreeExplainer(booster)
        self._feature_names = list(X_background.columns)
        self.base_value = float(self._explainer.expected_value)
        logger.info(
            "ShapExplainer inicializado | base_value={:.4f} | features={}",
            self.base_value,
            len(self._feature_names),
        )

    def explain(self, X: pd.DataFrame) -> np.ndarray:
        """
        Calcula valores SHAP para todas las observaciones en X.

        Args:
            X: DataFrame con las mismas columnas que X_background.

        Returns:
            Array (n_samples, n_features) de valores SHAP.
        """
        logger.debug("ShapExplainer.explain: {} observaciones", len(X))
        return self._explainer.shap_values(X[self._feature_names])

    def summary(self, X: pd.DataFrame, n_top: int = 10) -> ShapSummary:
        """
        Calcula un resumen de importancia de features sobre X.

        Args:
            X:     DataFrame de observaciones a explicar.
            n_top: Número de features a incluir en el resumen.

        Returns:
            ShapSummary con tabla de importancia ordenada por |SHAP| medio.
        """
        shap_values = self.explain(X)

        mean_abs = np.abs(shap_values).mean(axis=0)
        mean_signed = shap_values.mean(axis=0)

        importance_df = (
            pd.DataFrame(
                {
                    "feature": self._feature_names,
                    "mean_abs_shap": mean_abs,
                    "mean_shap": mean_signed,
                }
            )
            .sort_values("mean_abs_shap", ascending=False)
            .reset_index(drop=True)
        )

        logger.info(
            "ShapExplainer.summary: top feature = {} (|SHAP|={:.4f})",
            importance_df["feature"].iloc[0],
            importance_df["mean_abs_shap"].iloc[0],
        )

        return ShapSummary(
            feature_importance=importance_df,
            base_value=self.base_value,
            n_samples=len(X),
        )
