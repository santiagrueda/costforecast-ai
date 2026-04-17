"""
Consolidador de fuentes heterogéneas.

Los 4 archivos recibidos tienen formatos distintos:
- historico_equipos.csv: ya consolidado, formato estándar.
- X.csv: formato estándar (Date, Price).
- Y.csv: formato europeo (separador `;`, coma decimal, fechas D/M/YYYY).
- Z.csv: columnas invertidas (Price, Date).

Este módulo los unifica en un único DataFrame limpio con índice temporal
y lo persiste como parquet.

Decisión de diseño: usar `historico_equipos.csv` como fuente de verdad
para el rango de entrenamiento (2010-2023), y usar X.csv extendido
(1988-2024) solo para validación futura o análisis de tendencias de largo plazo.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from costforecast.logger import logger


def load_historico_equipos(path: Path) -> pd.DataFrame:
    """Carga el dataset consolidado (dataset principal de entrenamiento)."""
    logger.info("Cargando historico_equipos desde {}", path)
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index("Date").sort_index()
    logger.info("  → {} filas, {} columnas", *df.shape)
    return df


def load_materia_prima_x(path: Path) -> pd.DataFrame:
    """Carga X.csv (formato estándar, histórico extendido 1988-2024)."""
    logger.info("Cargando materia prima X desde {}", path)
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.rename(columns={"Price": "Price_X"})
    df = df.set_index("Date").sort_index()
    logger.info("  → {} filas, rango {} a {}", len(df), df.index.min().date(), df.index.max().date())
    return df


def load_materia_prima_y(path: Path) -> pd.DataFrame:
    """
    Carga Y.csv (formato europeo).

    Características:
    - Separador: `;`
    - Decimal: `,`
    - Fechas: `D/M/YYYY`
    - BOM UTF-8 al inicio del archivo
    """
    logger.info("Cargando materia prima Y desde {}", path)
    df = pd.read_csv(path, sep=";", decimal=",", encoding="utf-8-sig")
    df["Date"] = pd.to_datetime(df["Date"], format="%d/%m/%Y", errors="coerce")
    n_invalid = df["Date"].isna().sum()
    if n_invalid > 0:
        logger.warning("  → {} fechas inválidas descartadas", n_invalid)
        df = df.dropna(subset=["Date"])
    df = df.rename(columns={"Price": "Price_Y"})
    df = df.set_index("Date").sort_index()
    logger.info("  → {} filas, rango {} a {}", len(df), df.index.min().date(), df.index.max().date())
    return df


def load_materia_prima_z(path: Path) -> pd.DataFrame:
    """Carga Z.csv (columnas invertidas: Price, Date)."""
    logger.info("Cargando materia prima Z desde {}", path)
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.rename(columns={"Price": "Price_Z"})
    df = df[["Date", "Price_Z"]].set_index("Date").sort_index()
    logger.info("  → {} filas, rango {} a {}", len(df), df.index.min().date(), df.index.max().date())
    return df


def build_consolidated_dataset(
    historico_path: Path,
    x_path: Path,
    y_path: Path,
    z_path: Path,
    prefer_raw_series: bool = False,
) -> pd.DataFrame:
    """
    Construye el dataset consolidado.

    Args:
        historico_path: ruta a historico_equipos.csv
        x_path, y_path, z_path: rutas a los CSVs de cada materia prima
        prefer_raw_series: Si es True, usa las series individuales X/Y/Z.
            Si es False (default), usa los precios ya consolidados en historico_equipos.

    Returns:
        DataFrame con índice Date y columnas:
        [Price_X, Price_Y, Price_Z, Price_Equipo1, Price_Equipo2]
    """
    historico = load_historico_equipos(historico_path)

    if not prefer_raw_series:
        logger.info("Usando precios consolidados de historico_equipos (modo default)")
        return historico

    # Modo alternativo: reconstruir desde las fuentes individuales
    logger.info("Reconstruyendo desde fuentes individuales X, Y, Z")
    x = load_materia_prima_x(x_path)
    y = load_materia_prima_y(y_path)
    z = load_materia_prima_z(z_path)

    # Tomar solo los equipos de historico_equipos
    equipos = historico[["Price_Equipo1", "Price_Equipo2"]]

    # Inner join por fecha común a todas las fuentes
    consolidated = x.join(y, how="inner").join(z, how="inner").join(equipos, how="inner")

    # Reordenar columnas
    consolidated = consolidated[
        ["Price_X", "Price_Y", "Price_Z", "Price_Equipo1", "Price_Equipo2"]
    ]

    logger.info(
        "Consolidación completa: {} filas, rango {} a {}",
        len(consolidated),
        consolidated.index.min().date(),
        consolidated.index.max().date(),
    )
    return consolidated


def save_processed(df: pd.DataFrame, output_path: Path) -> None:
    """Guarda el dataset consolidado como parquet."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, compression="snappy")
    logger.info("Dataset guardado en {}", output_path)
