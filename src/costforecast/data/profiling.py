"""
ETL + auditoría de calidad del dataset. Invocado por `make eda`.

Consolida las 4 fuentes de datos, ejecuta la auditoría de calidad completa,
imprime el reporte en Markdown y persiste el dataset limpio como parquet.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Forzar UTF-8 en stdout para evitar UnicodeEncodeError en Windows (cp1252)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from costforecast.config import settings
from costforecast.data.consolidator import build_consolidated_dataset
from costforecast.data.quality import assess_quality
from costforecast.logger import logger


def run_profiling() -> None:
    """Ejecuta el pipeline ETL + auditoría y persiste el dataset procesado."""
    raw = settings.raw_data_dir
    logger.info("=== CostForecast AI — ETL + Auditoría de calidad ===")

    df = build_consolidated_dataset(
        historico_path=raw / "historico_equipos.csv",
        x_path=raw / "X.csv",
        y_path=raw / "Y.csv",
        z_path=raw / "Z.csv",
    )
    logger.info("Dataset consolidado: {} filas × {} columnas", *df.shape)

    value_cols = list(df.columns)
    report = assess_quality(
        df,
        value_columns=value_cols,
        frequency="B",  # business-day frequency
        run_stationarity=True,
    )

    print("\n" + report.to_markdown())

    out = settings.processed_dataset_path
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, compression="snappy")
    logger.info("Dataset guardado en {}", out)
    logger.info("=== Auditoría completada ===")


def main() -> None:
    run_profiling()


if __name__ == "__main__":
    main()
