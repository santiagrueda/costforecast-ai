"""Tests del módulo de carga de datos."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from costforecast.data import DataLoader, DatasetSchema


@pytest.fixture
def schema() -> DatasetSchema:
    return DatasetSchema(
        date_column="fecha",
        raw_material_columns=["cobre", "acero", "aluminio"],
        equipment_columns=["equipo_1", "equipo_2"],
    )


def test_schema_rejects_empty_columns() -> None:
    with pytest.raises(ValueError, match="materia prima"):
        DatasetSchema(
            date_column="fecha",
            raw_material_columns=[],
            equipment_columns=["equipo_1"],
        )


def test_loader_loads_valid_csv(synthetic_dataset: Path, schema: DatasetSchema) -> None:
    loader = DataLoader(schema=schema)
    df = loader.load(synthetic_dataset)

    assert len(df) == 60
    assert isinstance(df.index, pd.DatetimeIndex)
    assert set(schema.all_value_columns).issubset(df.columns)


def test_loader_raises_on_missing_file(schema: DatasetSchema) -> None:
    loader = DataLoader(schema=schema)
    with pytest.raises(FileNotFoundError):
        loader.load("nonexistent.csv")


def test_loader_detects_monthly_frequency(
    synthetic_dataset: Path, schema: DatasetSchema
) -> None:
    loader = DataLoader(schema=schema)
    _ = loader.load(synthetic_dataset)
    assert loader.inferred_frequency is not None
    assert "M" in loader.inferred_frequency or "monthly" in loader.inferred_frequency.lower()


def test_loader_validates_columns(tmp_path: Path, schema: DatasetSchema) -> None:
    # CSV con columnas incorrectas
    bad_df = pd.DataFrame({"fecha": ["2024-01-01"], "otra_cosa": [100]})
    bad_path = tmp_path / "bad.csv"
    bad_df.to_csv(bad_path, index=False)

    loader = DataLoader(schema=schema)
    with pytest.raises(ValueError, match="Columnas faltantes"):
        loader.load(bad_path)
