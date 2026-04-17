"""
Carga y validación de datasets de precios.

El loader es intencionalmente genérico: recibe un schema que describe las
columnas esperadas (fecha, materias primas, equipos) y valida que el dataset
cumpla con ese schema antes de retornarlo.

Esto permite adaptarse a cualquier estructura de CSV/Excel que nos entreguen,
simplemente definiendo el schema apropiado.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import pandas as pd
from pydantic import BaseModel, Field, model_validator

from costforecast.logger import logger


class DatasetSchema(BaseModel):
    """
    Schema esperado del dataset.

    Una vez que conozcamos el CSV real, instanciamos esto con las columnas
    correspondientes y el loader valida automáticamente.
    """

    date_column: str = Field(..., description="Nombre de la columna de fecha")
    raw_material_columns: list[str] = Field(
        ..., description="Columnas de precios de materias primas (predictores)"
    )
    equipment_columns: list[str] = Field(
        ..., description="Columnas de precios de equipos (variables a predecir)"
    )
    date_format: str | None = Field(
        default=None,
        description="Formato de la fecha (ej. '%Y-%m-%d'). Si es None, se infiere.",
    )
    expected_frequency: Literal["D", "W", "M", "Q", "Y"] | None = Field(
        default=None,
        description="Frecuencia temporal esperada. Si es None, se infiere.",
    )

    @model_validator(mode="after")
    def _validate_non_empty(self) -> DatasetSchema:
        if not self.raw_material_columns:
            raise ValueError("Debe haber al menos una columna de materia prima")
        if not self.equipment_columns:
            raise ValueError("Debe haber al menos una columna de equipo")
        return self

    @property
    def all_value_columns(self) -> list[str]:
        """Todas las columnas numéricas (materias primas + equipos)."""
        return self.raw_material_columns + self.equipment_columns


@dataclass
class DataLoader:
    """
    Cargador de datasets con validación y normalización.

    Uso:
        schema = DatasetSchema(
            date_column="fecha",
            raw_material_columns=["cobre", "acero", "aluminio"],
            equipment_columns=["equipo_1", "equipo_2"],
        )
        loader = DataLoader(schema=schema)
        df = loader.load("data/raw/dataset.csv")
    """

    schema: DatasetSchema
    _inferred_frequency: str | None = field(default=None, init=False, repr=False)

    def load(self, path: str | Path) -> pd.DataFrame:
        """Carga y valida un dataset desde disco (CSV o Excel)."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Dataset no encontrado: {path}")

        logger.info("Cargando dataset desde: {}", path)

        # Detectar formato
        if path.suffix.lower() in {".csv", ".txt"}:
            df = pd.read_csv(path)
        elif path.suffix.lower() in {".xlsx", ".xls"}:
            df = pd.read_excel(path)
        elif path.suffix.lower() == ".parquet":
            df = pd.read_parquet(path)
        else:
            raise ValueError(f"Formato no soportado: {path.suffix}")

        logger.info("Dataset cargado: {} filas x {} columnas", *df.shape)

        # Validar schema
        self._validate_columns(df)

        # Normalizar: parsear fechas, ordenar, establecer índice
        df = self._normalize(df)

        # Inferir frecuencia
        self._inferred_frequency = self._infer_frequency(df)
        logger.info("Frecuencia temporal detectada: {}", self._inferred_frequency)

        return df

    def _validate_columns(self, df: pd.DataFrame) -> None:
        """Verifica que todas las columnas del schema existan en el DataFrame."""
        required = {self.schema.date_column, *self.schema.all_value_columns}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(
                f"Columnas faltantes en el dataset: {missing}. "
                f"Columnas disponibles: {list(df.columns)}"
            )
        logger.debug("Validación de columnas OK")

    def _normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        """Parsea fechas, ordena cronológicamente, valida tipos numéricos."""
        df = df.copy()

        # Parsear fecha
        try:
            df[self.schema.date_column] = pd.to_datetime(
                df[self.schema.date_column], format=self.schema.date_format
            )
        except Exception as e:
            raise ValueError(
                f"No se pudo parsear la columna de fecha '{self.schema.date_column}': {e}"
            ) from e

        # Ordenar cronológicamente
        df = df.sort_values(self.schema.date_column).reset_index(drop=True)

        # Validar que las columnas de valor sean numéricas
        for col in self.schema.all_value_columns:
            if not pd.api.types.is_numeric_dtype(df[col]):
                logger.warning("Columna '{}' no es numérica, intentando coerción", col)
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # Establecer fecha como índice
        df = df.set_index(self.schema.date_column)

        return df

    def _infer_frequency(self, df: pd.DataFrame) -> str:
        """Infiere la frecuencia temporal del dataset."""
        inferred = pd.infer_freq(df.index)
        if inferred:
            return inferred

        # Fallback: calcular la diferencia modal entre fechas consecutivas
        diffs = df.index.to_series().diff().dropna()
        if len(diffs) == 0:
            return "unknown"
        modal_diff = diffs.mode().iloc[0]
        days = modal_diff.days

        if days == 1:
            return "D (daily)"
        elif 6 <= days <= 8:
            return "W (weekly)"
        elif 28 <= days <= 32:
            return "M (monthly)"
        elif 88 <= days <= 93:
            return "Q (quarterly)"
        elif 360 <= days <= 370:
            return "Y (yearly)"
        else:
            return f"~{days} days"

    @property
    def inferred_frequency(self) -> str | None:
        """Frecuencia detectada tras la carga."""
        return self._inferred_frequency
