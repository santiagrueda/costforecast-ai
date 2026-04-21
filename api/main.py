"""
API REST de CostForecast AI — FastAPI.
Invocado por `make api` (uvicorn api.main:app --reload --port 8000).

Endpoints:
  GET  /health                     — Estado del servicio
  GET  /forecast/{equipment}       — Pronóstico futuro (SARIMAX + Monte Carlo)
  GET  /historical                 — Estadísticas del dataset histórico
  POST /scenario                   — Simulación what-if de materias primas
  GET  /shap/{equipment}           — Importancia de features (SHAP)
  GET  /models                     — Lista de modelos disponibles
"""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from costforecast.agent.tools import (
    clear_cache,
    get_forecast,
    get_historical_data,
    get_shap_explanation,
    simulate_scenario,
)
from costforecast.config import settings
from costforecast.logger import logger

# ---------------------------------------------------------------------------
# Aplicación FastAPI
# ---------------------------------------------------------------------------

app = FastAPI(
    title="CostForecast AI",
    description=(
        "API REST para pronóstico de costos de equipos de construcción. "
        "Usa SARIMAX con variables exógenas de materias primas (X, Y, Z)."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Schemas Pydantic
# ---------------------------------------------------------------------------

VALID_EQUIPMENT = {"equipo1", "equipo2"}


class HealthResponse(BaseModel):
    status: str
    version: str
    model: str


class ForecastResponse(BaseModel):
    equipment: str
    horizon_days: int
    result: str


class HistoricalResponse(BaseModel):
    result: str


class ScenarioRequest(BaseModel):
    equipment: str = Field(description="'equipo1' o 'equipo2'")
    price_x_change_pct: float = Field(default=0.0, ge=-50.0, le=100.0)
    price_y_change_pct: float = Field(default=0.0, ge=-50.0, le=100.0)
    price_z_change_pct: float = Field(default=0.0, ge=-50.0, le=100.0)
    horizon_days: int = Field(default=10, ge=1, le=60)


class ScenarioResponse(BaseModel):
    equipment: str
    result: str


class ShapResponse(BaseModel):
    equipment: str
    n_top_features: int
    result: str


class ModelsResponse(BaseModel):
    models: list[str]
    targets: list[str]
    exog_cols: list[str]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _validate_equipment(equipment: str) -> str:
    key = equipment.strip().lower()
    if key not in VALID_EQUIPMENT:
        raise HTTPException(
            status_code=422,
            detail=f"Equipo '{equipment}' no válido. Opciones: {sorted(VALID_EQUIPMENT)}",
        )
    return key


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse, tags=["Sistema"])
def health() -> dict[str, Any]:
    """Verifica que el servicio está activo."""
    return {
        "status": "ok",
        "version": "1.0.0",
        "model": settings.claude_model,
    }


@app.get("/forecast/{equipment}", response_model=ForecastResponse, tags=["Pronóstico"])
def forecast(
    equipment: str,
    horizon_days: int = Query(default=10, ge=1, le=60, description="Días hábiles a proyectar"),
) -> dict[str, Any]:
    """
    Genera un pronóstico del precio de un equipo usando SARIMAX(1,1,1).

    - **equipment**: `equipo1` o `equipo2`
    - **horizon_days**: días hábiles a proyectar (1–60, default 10)
    """
    _validate_equipment(equipment)
    logger.info("GET /forecast/{} horizon={}", equipment, horizon_days)

    result = get_forecast.invoke({"equipment": equipment, "horizon_days": horizon_days})
    return {"equipment": equipment, "horizon_days": horizon_days, "result": result}


@app.get("/historical", response_model=HistoricalResponse, tags=["Datos"])
def historical(
    columns: list[str] = Query(
        default=[],
        description="Columnas a consultar. Dejar vacío para todas.",
    ),
    start_date: str | None = Query(default=None, description="Fecha inicio YYYY-MM-DD"),
    end_date: str | None = Query(default=None, description="Fecha fin YYYY-MM-DD"),
    last_n: int | None = Query(default=None, ge=1, le=500, description="Últimas N observaciones"),
) -> dict[str, Any]:
    """
    Consulta estadísticas del dataset histórico de precios.

    Devuelve estadísticas descriptivas y la tendencia reciente de las series.
    """
    logger.info("GET /historical cols={} start={} end={} last_n={}", columns, start_date, end_date, last_n)
    result = get_historical_data.invoke(
        {
            "columns": columns,
            "start_date": start_date,
            "end_date": end_date,
            "last_n": last_n,
        }
    )
    return {"result": result}


@app.post("/scenario", response_model=ScenarioResponse, tags=["Simulación"])
def scenario(body: ScenarioRequest) -> dict[str, Any]:
    """
    Simula el impacto en el precio de un equipo ante cambios hipotéticos
    en las materias primas (análisis what-if).

    Compara el pronóstico baseline contra el escenario con las variaciones indicadas.
    """
    _validate_equipment(body.equipment)
    logger.info(
        "POST /scenario equipment={} X={:+.1f}% Y={:+.1f}% Z={:+.1f}%",
        body.equipment, body.price_x_change_pct, body.price_y_change_pct, body.price_z_change_pct,
    )

    result = simulate_scenario.invoke(
        {
            "equipment": body.equipment,
            "price_x_change_pct": body.price_x_change_pct,
            "price_y_change_pct": body.price_y_change_pct,
            "price_z_change_pct": body.price_z_change_pct,
            "horizon_days": body.horizon_days,
        }
    )
    return {"equipment": body.equipment, "result": result}


@app.get("/shap/{equipment}", response_model=ShapResponse, tags=["Explicabilidad"])
def shap(
    equipment: str,
    n_top_features: int = Query(default=10, ge=3, le=30, description="Número de features a mostrar"),
    n_samples: int = Query(default=200, ge=20, le=1000, description="Observaciones para SHAP"),
) -> dict[str, Any]:
    """
    Muestra las features más influyentes en las predicciones del modelo XGBoost
    usando valores de Shapley.
    """
    _validate_equipment(equipment)
    logger.info("GET /shap/{} n_top={} n_samples={}", equipment, n_top_features, n_samples)

    result = get_shap_explanation.invoke(
        {"equipment": equipment, "n_top_features": n_top_features, "n_samples": n_samples}
    )
    return {"equipment": equipment, "n_top_features": n_top_features, "result": result}


@app.get("/models", response_model=ModelsResponse, tags=["Sistema"])
def list_models() -> dict[str, Any]:
    """Lista los modelos disponibles en el proyecto."""
    return {
        "models": ["Persistence", "SARIMAX(1,1,1)", "Prophet", "XGBoost"],
        "targets": ["Price_Equipo1", "Price_Equipo2"],
        "exog_cols": ["Price_X", "Price_Y", "Price_Z"],
    }


@app.delete("/cache", tags=["Sistema"])
def reset_cache() -> dict[str, str]:
    """Limpia la caché de modelos en memoria (fuerza re-entrenamiento en la próxima petición)."""
    clear_cache()
    logger.info("Caché de modelos limpiada vía API")
    return {"status": "cache cleared"}
