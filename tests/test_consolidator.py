"""Tests de integración del consolidador con los datasets reales de data/raw/."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from costforecast.data.consolidator import (
    build_consolidated_dataset,
    load_historico_equipos,
    load_materia_prima_x,
    load_materia_prima_y,
    load_materia_prima_z,
)

RAW_DIR = Path(__file__).parent.parent / "data" / "raw"

HISTORICO_PATH = RAW_DIR / "historico_equipos.csv"
X_PATH = RAW_DIR / "X.csv"
Y_PATH = RAW_DIR / "Y.csv"
Z_PATH = RAW_DIR / "Z.csv"


# ---------------------------------------------------------------------------
# load_historico_equipos
# ---------------------------------------------------------------------------

def test_historico_index_is_datetime() -> None:
    df = load_historico_equipos(HISTORICO_PATH)
    assert isinstance(df.index, pd.DatetimeIndex), "El índice debe ser DatetimeIndex"


def test_historico_has_expected_columns() -> None:
    df = load_historico_equipos(HISTORICO_PATH)
    expected = {"Price_X", "Price_Y", "Price_Z", "Price_Equipo1", "Price_Equipo2"}
    assert expected.issubset(df.columns), f"Columnas faltantes: {expected - set(df.columns)}"


def test_historico_is_sorted() -> None:
    df = load_historico_equipos(HISTORICO_PATH)
    assert df.index.is_monotonic_increasing, "El índice debe estar ordenado cronológicamente"


def test_historico_no_nulls() -> None:
    df = load_historico_equipos(HISTORICO_PATH)
    assert df.isnull().sum().sum() == 0, "No se esperan nulos en historico_equipos"


def test_historico_positive_prices() -> None:
    df = load_historico_equipos(HISTORICO_PATH)
    assert (df > 0).all().all(), "Todos los precios deben ser positivos"


# ---------------------------------------------------------------------------
# load_materia_prima_x
# ---------------------------------------------------------------------------

def test_x_index_is_datetime() -> None:
    df = load_materia_prima_x(X_PATH)
    assert isinstance(df.index, pd.DatetimeIndex)


def test_x_column_renamed() -> None:
    df = load_materia_prima_x(X_PATH)
    assert "Price_X" in df.columns, "La columna Price debe renombrarse a Price_X"


def test_x_sorted() -> None:
    df = load_materia_prima_x(X_PATH)
    assert df.index.is_monotonic_increasing


def test_x_long_history() -> None:
    """X.csv tiene histórico extendido (1988-2024)."""
    df = load_materia_prima_x(X_PATH)
    assert df.index.min().year <= 2024
    assert len(df) > 0


# ---------------------------------------------------------------------------
# load_materia_prima_y  (formato europeo: sep=;, decimal=,, D/M/YYYY, BOM)
# ---------------------------------------------------------------------------

def test_y_index_is_datetime() -> None:
    df = load_materia_prima_y(Y_PATH)
    assert isinstance(df.index, pd.DatetimeIndex)


def test_y_column_renamed() -> None:
    df = load_materia_prima_y(Y_PATH)
    assert "Price_Y" in df.columns


def test_y_prices_are_numeric() -> None:
    df = load_materia_prima_y(Y_PATH)
    assert pd.api.types.is_numeric_dtype(df["Price_Y"]), (
        "Los precios de Y deben ser numéricos (verificar decimal=',')"
    )


def test_y_sorted() -> None:
    df = load_materia_prima_y(Y_PATH)
    assert df.index.is_monotonic_increasing


# ---------------------------------------------------------------------------
# load_materia_prima_z  (columnas invertidas: Price, Date)
# ---------------------------------------------------------------------------

def test_z_index_is_datetime() -> None:
    df = load_materia_prima_z(Z_PATH)
    assert isinstance(df.index, pd.DatetimeIndex)


def test_z_column_renamed() -> None:
    df = load_materia_prima_z(Z_PATH)
    assert "Price_Z" in df.columns


def test_z_only_price_column() -> None:
    df = load_materia_prima_z(Z_PATH)
    assert list(df.columns) == ["Price_Z"], (
        "Z debe tener solo Price_Z tras el reordenamiento de columnas invertidas"
    )


def test_z_sorted() -> None:
    df = load_materia_prima_z(Z_PATH)
    assert df.index.is_monotonic_increasing


# ---------------------------------------------------------------------------
# build_consolidated_dataset — modo default (historico como fuente de verdad)
# ---------------------------------------------------------------------------

def test_build_default_returns_historico() -> None:
    df = build_consolidated_dataset(HISTORICO_PATH, X_PATH, Y_PATH, Z_PATH)
    expected_cols = {"Price_X", "Price_Y", "Price_Z", "Price_Equipo1", "Price_Equipo2"}
    assert expected_cols.issubset(df.columns)


def test_build_default_index_is_datetime() -> None:
    df = build_consolidated_dataset(HISTORICO_PATH, X_PATH, Y_PATH, Z_PATH)
    assert isinstance(df.index, pd.DatetimeIndex)


def test_build_default_no_empty() -> None:
    df = build_consolidated_dataset(HISTORICO_PATH, X_PATH, Y_PATH, Z_PATH)
    assert len(df) > 0


# ---------------------------------------------------------------------------
# build_consolidated_dataset — modo prefer_raw_series=True
# ---------------------------------------------------------------------------

def test_build_raw_series_columns() -> None:
    df = build_consolidated_dataset(
        HISTORICO_PATH, X_PATH, Y_PATH, Z_PATH, prefer_raw_series=True
    )
    expected = {"Price_X", "Price_Y", "Price_Z", "Price_Equipo1", "Price_Equipo2"}
    assert set(df.columns) == expected


def test_build_raw_series_inner_join_non_empty() -> None:
    """El inner join de las 4 fuentes debe producir al menos un registro."""
    df = build_consolidated_dataset(
        HISTORICO_PATH, X_PATH, Y_PATH, Z_PATH, prefer_raw_series=True
    )
    assert len(df) > 0, (
        "El inner join de X, Y, Z y equipos no produjo registros — revisar solapamiento de fechas"
    )


def test_build_raw_series_no_nulls() -> None:
    df = build_consolidated_dataset(
        HISTORICO_PATH, X_PATH, Y_PATH, Z_PATH, prefer_raw_series=True
    )
    assert df.isnull().sum().sum() == 0
