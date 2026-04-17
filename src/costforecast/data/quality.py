"""
Auditoría de calidad de datos.

Genera un reporte cuantitativo sobre:
- Completitud (nulos por columna)
- Rango temporal y gaps de fechas
- Outliers detectados con IQR y z-score
- Estadísticas descriptivas
- Estacionariedad (test ADF por columna)

Este reporte es el primer entregable del EDA y sirve como sección
"Auditoría de integridad del dataset" en el informe ejecutivo.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tsa.stattools import adfuller

from costforecast.logger import logger


@dataclass
class DataQualityReport:
    """Resultado estructurado de la auditoría de calidad."""

    n_rows: int
    n_cols: int
    date_range: tuple[pd.Timestamp, pd.Timestamp]
    frequency: str | None
    missing_values: dict[str, int]
    missing_pct: dict[str, float]
    duplicated_dates: int
    date_gaps: int
    outliers_iqr: dict[str, int]
    outliers_zscore: dict[str, int]
    descriptive_stats: pd.DataFrame
    stationarity: dict[str, dict[str, Any]] = field(default_factory=dict)

    def to_markdown(self) -> str:
        """Genera un reporte en Markdown listo para el informe."""
        lines: list[str] = []
        lines.append("# Auditoría de calidad del dataset\n")

        lines.append("## Dimensiones\n")
        lines.append(f"- **Filas**: {self.n_rows:,}")
        lines.append(f"- **Columnas**: {self.n_cols}")
        lines.append(f"- **Rango temporal**: {self.date_range[0].date()} → {self.date_range[1].date()}")
        lines.append(f"- **Frecuencia detectada**: {self.frequency}")
        lines.append(f"- **Fechas duplicadas**: {self.duplicated_dates}")
        lines.append(f"- **Gaps en serie temporal**: {self.date_gaps}\n")

        lines.append("## Valores faltantes\n")
        lines.append("| Columna | Nulos | % |")
        lines.append("|---|---|---|")
        for col, n in self.missing_values.items():
            lines.append(f"| {col} | {n} | {self.missing_pct[col]:.2f}% |")

        lines.append("\n## Outliers detectados\n")
        lines.append("| Columna | IQR (1.5×) | Z-score (>3σ) |")
        lines.append("|---|---|---|")
        for col in self.outliers_iqr:
            lines.append(f"| {col} | {self.outliers_iqr[col]} | {self.outliers_zscore[col]} |")

        if self.stationarity:
            lines.append("\n## Estacionariedad (Augmented Dickey-Fuller)\n")
            lines.append("| Columna | ADF statistic | p-value | Estacionaria? |")
            lines.append("|---|---|---|---|")
            for col, result in self.stationarity.items():
                is_stat = "✓ Sí" if result["is_stationary"] else "✗ No"
                lines.append(
                    f"| {col} | {result['adf_stat']:.4f} | {result['p_value']:.4f} | {is_stat} |"
                )

        lines.append("\n## Estadísticas descriptivas\n")
        lines.append(self.descriptive_stats.to_markdown())

        return "\n".join(lines)


def _count_outliers_iqr(series: pd.Series, factor: float = 1.5) -> int:
    """Cuenta outliers usando el método de rango intercuartílico."""
    clean = series.dropna()
    if len(clean) < 4:
        return 0
    q1 = clean.quantile(0.25)
    q3 = clean.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - factor * iqr
    upper = q3 + factor * iqr
    return int(((clean < lower) | (clean > upper)).sum())


def _count_outliers_zscore(series: pd.Series, threshold: float = 3.0) -> int:
    """Cuenta outliers usando z-score estándar."""
    clean = series.dropna()
    if len(clean) < 3 or clean.std() == 0:
        return 0
    z = np.abs(stats.zscore(clean))
    return int((z > threshold).sum())


def _detect_date_gaps(index: pd.DatetimeIndex, expected_freq: str | None) -> int:
    """Detecta fechas faltantes según la frecuencia esperada."""
    if expected_freq is None or expected_freq == "unknown":
        return 0
    try:
        freq_code = expected_freq.split()[0]  # "M (monthly)" → "M"
        expected = pd.date_range(start=index.min(), end=index.max(), freq=freq_code)
        return len(expected) - len(index)
    except Exception:
        return 0


def _test_stationarity(series: pd.Series) -> dict[str, Any]:
    """Test de Dickey-Fuller aumentado."""
    clean = series.dropna()
    if len(clean) < 10:
        return {"adf_stat": np.nan, "p_value": np.nan, "is_stationary": False}
    try:
        adf_stat, p_value, *_ = adfuller(clean, autolag="AIC")
        return {
            "adf_stat": float(adf_stat),
            "p_value": float(p_value),
            "is_stationary": bool(p_value < 0.05),
        }
    except Exception as e:
        logger.warning("ADF falló para una serie: {}", e)
        return {"adf_stat": np.nan, "p_value": np.nan, "is_stationary": False}


def assess_quality(
    df: pd.DataFrame,
    value_columns: list[str],
    frequency: str | None = None,
    run_stationarity: bool = True,
) -> DataQualityReport:
    """
    Ejecuta la auditoría completa y retorna un reporte estructurado.

    Args:
        df: DataFrame con índice temporal.
        value_columns: Columnas numéricas a analizar.
        frequency: Frecuencia esperada para detectar gaps.
        run_stationarity: Si correr el test ADF (costoso para muchas columnas).

    Returns:
        DataQualityReport con todos los hallazgos.
    """
    logger.info("Ejecutando auditoría de calidad sobre {} columnas", len(value_columns))

    # Missing values
    missing = df[value_columns].isna().sum().to_dict()
    missing_pct = {
        col: (n / len(df)) * 100 for col, n in missing.items()
    }

    # Duplicated dates
    duplicated = int(df.index.duplicated().sum())

    # Date gaps
    gaps = _detect_date_gaps(df.index, frequency)  # type: ignore[arg-type]

    # Outliers
    outliers_iqr = {col: _count_outliers_iqr(df[col]) for col in value_columns}
    outliers_z = {col: _count_outliers_zscore(df[col]) for col in value_columns}

    # Estadísticas descriptivas
    desc = df[value_columns].describe().T.round(2)

    # Estacionariedad
    stationarity = {}
    if run_stationarity:
        for col in value_columns:
            stationarity[col] = _test_stationarity(df[col])

    report = DataQualityReport(
        n_rows=len(df),
        n_cols=len(df.columns),
        date_range=(df.index.min(), df.index.max()),
        frequency=frequency,
        missing_values=missing,
        missing_pct=missing_pct,
        duplicated_dates=duplicated,
        date_gaps=gaps,
        outliers_iqr=outliers_iqr,
        outliers_zscore=outliers_z,
        descriptive_stats=desc,
        stationarity=stationarity,
    )
    logger.info("Auditoría completada")
    return report
