"""Fixtures compartidas para todos los tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def synthetic_dataset(tmp_path: Path) -> Path:
    """
    Genera un dataset sintético con la estructura típica del caso:
    - Fecha mensual
    - 3 materias primas (cobre, acero, aluminio)
    - 2 equipos (equipo_1, equipo_2)

    Útil para probar el pipeline sin necesitar el CSV real.
    """
    rng = np.random.default_rng(42)
    n_months = 60

    dates = pd.date_range(start="2019-01-01", periods=n_months, freq="MS")

    # Materias primas con tendencia + ruido
    cobre = 3000 + np.linspace(0, 1500, n_months) + rng.normal(0, 100, n_months)
    acero = 500 + np.linspace(0, 200, n_months) + rng.normal(0, 30, n_months)
    aluminio = 1800 + np.linspace(0, 500, n_months) + rng.normal(0, 80, n_months)

    # Equipo 1: depende principalmente de cobre y aluminio
    equipo_1 = 0.5 * cobre + 0.3 * aluminio + rng.normal(0, 50, n_months)
    # Equipo 2: depende principalmente de acero
    equipo_2 = 2.0 * acero + 0.1 * cobre + rng.normal(0, 40, n_months)

    df = pd.DataFrame(
        {
            "fecha": dates,
            "cobre": cobre.round(2),
            "acero": acero.round(2),
            "aluminio": aluminio.round(2),
            "equipo_1": equipo_1.round(2),
            "equipo_2": equipo_2.round(2),
        }
    )

    path = tmp_path / "synthetic.csv"
    df.to_csv(path, index=False)
    return path
