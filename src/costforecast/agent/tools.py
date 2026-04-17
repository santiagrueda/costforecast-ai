"""
Cinco tools del agente CostForecast AI.

Cada tool encapsula una capacidad analítica del proyecto y devuelve texto
que el LLM puede interpretar directamente en el loop ReAct.

Tools disponibles:
──────────────────
1. get_forecast            — Proyección futura con SARIMAX
2. get_historical_data     — Consulta del dataset histórico con estadísticas
3. web_search_market_news  — Búsqueda de noticias de mercado vía Tavily
4. simulate_scenario       — Simulación "what-if" de cambios en materias primas
5. get_shap_explanation    — Explicación de importancia de features con SHAP

Decisiones de diseño:
─────────────────────
- Los modelos y el dataset se cachean en memoria la primera vez que se cargan
  para evitar I/O y entrenamiento repetido dentro de una sesión de agente.
- Todos los errores quedan capturados y se devuelven como texto descriptivo:
  el agente nunca ve una excepción Python, lo que evita que el loop ReAct
  se interrumpa por un fallo de tool.
- Se expone clear_cache() para facilitar el testing con datos controlados.
- web_search_market_news degrada graciosamente si TAVILY_API_KEY no está
  configurada, devolviendo un mensaje explicativo en lugar de lanzar error.
"""

from __future__ import annotations

import traceback
from typing import Any

import numpy as np
import pandas as pd
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from costforecast.config import settings
from costforecast.data.consolidator import build_consolidated_dataset
from costforecast.features.engineering import create_feature_matrix
from costforecast.logger import logger
from costforecast.models.sarimax_model import SARIMAXModel
from costforecast.models.xgboost_model import XGBoostModel

# ---------------------------------------------------------------------------
# Constantes del dominio
# ---------------------------------------------------------------------------

EXOG_COLS: list[str] = ["Price_X", "Price_Y", "Price_Z"]
VALID_TARGETS: dict[str, str] = {
    "equipo1": "Price_Equipo1",
    "equipo2": "Price_Equipo2",
    "price_equipo1": "Price_Equipo1",
    "price_equipo2": "Price_Equipo2",
}

# ---------------------------------------------------------------------------
# Cache de sesión (dataset + modelos entrenados)
# ---------------------------------------------------------------------------

_cache: dict[str, Any] = {}


def clear_cache() -> None:
    """Limpia todos los objetos cacheados. Útil para tests."""
    _cache.clear()


def _load_dataset() -> pd.DataFrame:
    """Carga y cachea el dataset consolidado."""
    if "dataset" not in _cache:
        raw = settings.raw_data_dir
        df = build_consolidated_dataset(
            historico_path=raw / "historico_equipos.csv",
            x_path=raw / "X.csv",
            y_path=raw / "Y.csv",
            z_path=raw / "Z.csv",
        )
        _cache["dataset"] = df
    return _cache["dataset"]


def _resolve_target(equipment: str) -> str:
    """Normaliza el nombre del equipo a la columna del DataFrame."""
    key = equipment.strip().lower()
    col = VALID_TARGETS.get(key)
    if col is None:
        raise ValueError(
            f"Equipo '{equipment}' no reconocido. "
            f"Opciones válidas: {list(VALID_TARGETS.keys())}"
        )
    return col


def _get_or_fit_sarimax(target_col: str) -> SARIMAXModel:
    """Ajusta y cachea SARIMAX(1,1,1) para el target dado."""
    cache_key = f"sarimax_{target_col}"
    if cache_key not in _cache:
        df = _load_dataset()
        model = SARIMAXModel(order=(1, 1, 1))
        model.fit(df[EXOG_COLS], df[target_col])
        _cache[cache_key] = model
        logger.debug("SARIMAX cacheado para {}", target_col)
    return _cache[cache_key]


def _get_or_fit_xgboost(
    target_col: str,
) -> tuple[XGBoostModel, pd.DataFrame, pd.Series]:
    """Ajusta y cachea XGBoostModel para el target dado. Retorna (model, X, y)."""
    cache_key = f"xgb_{target_col}"
    if cache_key not in _cache:
        df = _load_dataset()
        X, y = create_feature_matrix(df, target=target_col, lags=[1, 2, 3, 4, 5], windows=[5, 10, 20])
        model = XGBoostModel(n_estimators=300)
        model.fit(X, y)
        _cache[cache_key] = (model, X, y)
        logger.debug("XGBoost cacheado para {}", target_col)
    return _cache[cache_key]  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Tool 1 — get_forecast
# ---------------------------------------------------------------------------

class GetForecastInput(BaseModel):
    equipment: str = Field(
        description="Equipo a proyectar. Valores aceptados: 'equipo1', 'equipo2'."
    )
    horizon_days: int = Field(
        default=10,
        ge=1,
        le=60,
        description="Número de días hábiles a proyectar (1–60).",
    )


@tool(args_schema=GetForecastInput)
def get_forecast(equipment: str, horizon_days: int = 10) -> str:
    """
    Genera una proyección futura del precio de un equipo usando SARIMAX(1,1,1)
    con las tres materias primas (X, Y, Z) como variables exógenas.
    El pronóstico asume que las materias primas se mantienen en su último valor
    conocido (supuesto de persistencia).

    Devuelve una tabla con las fechas proyectadas, los valores estimados y
    estadísticas resumidas (media, mín, máx).
    """
    try:
        target_col = _resolve_target(equipment)
        df = _load_dataset()
        model = _get_or_fit_sarimax(target_col)

        # Construir exog futura: persistencia del último valor conocido
        last_exog = df[EXOG_COLS].iloc[-1]
        last_date = df.index[-1]
        future_dates = pd.bdate_range(
            start=last_date + pd.offsets.BDay(1), periods=horizon_days
        )
        X_future = pd.DataFrame(
            np.tile(last_exog.values, (horizon_days, 1)),
            columns=EXOG_COLS,
            index=future_dates,
        )

        preds = model.predict(X_future)
        last_actual = float(df[target_col].iloc[-1])

        lines = [
            f"## Pronóstico — {target_col} ({horizon_days} días hábiles)",
            f"Último valor real: {last_actual:.2f} (al {last_date.date()})",
            "",
            "| Fecha        | Pronóstico |",
            "|--------------|-----------|",
        ]
        for date, val in zip(future_dates, preds):
            lines.append(f"| {date.date()} | {val:.2f} |")

        lines += [
            "",
            f"**Media**: {preds.mean():.2f}",
            f"**Mín**: {preds.min():.2f} | **Máx**: {preds.max():.2f}",
            f"**Variación esperada vs actual**: {(preds.mean() - last_actual) / last_actual * 100:+.2f}%",
        ]
        return "\n".join(lines)

    except Exception:
        return f"Error al generar pronóstico: {traceback.format_exc(limit=3)}"


# ---------------------------------------------------------------------------
# Tool 2 — get_historical_data
# ---------------------------------------------------------------------------

class GetHistoricalDataInput(BaseModel):
    columns: list[str] = Field(
        default_factory=list,
        description=(
            "Columnas a consultar. Dejar vacío para obtener todas. "
            "Opciones: Price_X, Price_Y, Price_Z, Price_Equipo1, Price_Equipo2."
        ),
    )
    start_date: str | None = Field(
        default=None,
        description="Fecha inicial en formato YYYY-MM-DD (opcional).",
    )
    end_date: str | None = Field(
        default=None,
        description="Fecha final en formato YYYY-MM-DD (opcional).",
    )
    last_n: int | None = Field(
        default=None,
        ge=1,
        le=500,
        description="Mostrar las últimas N observaciones en lugar de un rango de fechas.",
    )


@tool(args_schema=GetHistoricalDataInput)
def get_historical_data(
    columns: list[str] | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    last_n: int | None = None,
) -> str:
    """
    Consulta el dataset histórico de precios (materias primas + equipos).
    Devuelve estadísticas descriptivas (media, desviación, percentiles,
    tendencia reciente) y las últimas observaciones del periodo solicitado.
    """
    try:
        df = _load_dataset()

        # Filtrar columnas
        cols = [c for c in (columns or []) if c in df.columns] or list(df.columns)
        sub = df[cols]

        # Filtrar fechas
        if start_date:
            sub = sub.loc[sub.index >= start_date]
        if end_date:
            sub = sub.loc[sub.index <= end_date]
        if last_n:
            sub = sub.iloc[-last_n:]

        if sub.empty:
            return "No se encontraron datos para el rango solicitado."

        desc = sub.describe().round(2)
        trend_30d = (
            (sub.iloc[-1] / sub.iloc[-min(30, len(sub))] - 1) * 100
        ).round(2)

        lines = [
            f"## Datos históricos — {', '.join(cols)}",
            f"Periodo: {sub.index[0].date()} → {sub.index[-1].date()} "
            f"({len(sub):,} observaciones)",
            "",
            "### Estadísticas descriptivas",
            desc.to_string(),
            "",
            "### Tendencia últimos 30 días (%)",
        ]
        for col_name, val in trend_30d.items():
            direction = "↑" if val > 0 else ("↓" if val < 0 else "→")
            lines.append(f"  {col_name}: {val:+.2f}% {direction}")

        lines += ["", "### Últimas 5 observaciones", sub.tail(5).to_string()]
        return "\n".join(lines)

    except Exception:
        return f"Error al consultar datos históricos: {traceback.format_exc(limit=3)}"


# ---------------------------------------------------------------------------
# Tool 3 — web_search_market_news
# ---------------------------------------------------------------------------

class WebSearchInput(BaseModel):
    query: str = Field(
        description=(
            "Consulta de búsqueda en inglés o español. "
            "Ejemplos: 'steel prices outlook 2024', "
            "'precios acero tendencia mercado'."
        )
    )
    max_results: int = Field(
        default=5,
        ge=1,
        le=10,
        description="Número máximo de resultados a devolver.",
    )


@tool(args_schema=WebSearchInput)
def web_search_market_news(query: str, max_results: int = 5) -> str:
    """
    Busca noticias y análisis recientes sobre materias primas, precios de equipos
    de construcción y factores macroeconómicos relevantes usando la API de Tavily.
    Útil para contextualizar los pronósticos cuantitativos con información actual
    del mercado.
    """
    try:
        api_key = settings.tavily_api_key
        if not api_key:
            return (
                "⚠️  TAVILY_API_KEY no está configurada. "
                "Para habilitar búsqueda web, agrega la clave en el archivo .env: "
                "TAVILY_API_KEY=tvly-..."
            )

        from tavily import TavilyClient  # noqa: PLC0415

        client = TavilyClient(api_key=api_key)
        response = client.search(
            query=query,
            max_results=max_results,
            search_depth="basic",
        )

        results = response.get("results", [])
        if not results:
            return f"No se encontraron resultados para: '{query}'"

        lines = [
            f"## Resultados de búsqueda: '{query}'",
            f"({len(results)} resultado(s))",
            "",
        ]
        for i, r in enumerate(results, 1):
            lines.append(f"### {i}. {r.get('title', 'Sin título')}")
            lines.append(f"**URL**: {r.get('url', '')}")
            lines.append(f"**Resumen**: {r.get('content', 'Sin contenido')[:400]}")
            lines.append("")

        return "\n".join(lines)

    except Exception:
        return f"Error en búsqueda web: {traceback.format_exc(limit=3)}"


# ---------------------------------------------------------------------------
# Tool 4 — simulate_scenario
# ---------------------------------------------------------------------------

class SimulateScenarioInput(BaseModel):
    equipment: str = Field(
        description="Equipo a simular: 'equipo1' o 'equipo2'."
    )
    price_x_change_pct: float = Field(
        default=0.0,
        ge=-50.0,
        le=100.0,
        description="Cambio porcentual en Price_X respecto al último valor conocido.",
    )
    price_y_change_pct: float = Field(
        default=0.0,
        ge=-50.0,
        le=100.0,
        description="Cambio porcentual en Price_Y respecto al último valor conocido.",
    )
    price_z_change_pct: float = Field(
        default=0.0,
        ge=-50.0,
        le=100.0,
        description="Cambio porcentual en Price_Z respecto al último valor conocido.",
    )
    horizon_days: int = Field(
        default=10,
        ge=1,
        le=60,
        description="Horizonte de simulación en días hábiles.",
    )


@tool(args_schema=SimulateScenarioInput)
def simulate_scenario(
    equipment: str,
    price_x_change_pct: float = 0.0,
    price_y_change_pct: float = 0.0,
    price_z_change_pct: float = 0.0,
    horizon_days: int = 10,
) -> str:
    """
    Simula el impacto en el precio de un equipo ante cambios hipotéticos
    en las materias primas (análisis what-if). Compara el pronóstico baseline
    (materias primas constantes en su último valor) contra el escenario
    con las variaciones indicadas. Usa SARIMAX(1,1,1) para ambas proyecciones.
    """
    try:
        target_col = _resolve_target(equipment)
        df = _load_dataset()
        model = _get_or_fit_sarimax(target_col)

        last_exog = df[EXOG_COLS].iloc[-1].copy()
        last_date = df.index[-1]
        future_dates = pd.bdate_range(
            start=last_date + pd.offsets.BDay(1), periods=horizon_days
        )

        # Baseline: persistencia sin cambios
        X_baseline = pd.DataFrame(
            np.tile(last_exog.values, (horizon_days, 1)),
            columns=EXOG_COLS,
            index=future_dates,
        )

        # Escenario: aplicar cambios porcentuales
        changes = {
            "Price_X": price_x_change_pct / 100.0,
            "Price_Y": price_y_change_pct / 100.0,
            "Price_Z": price_z_change_pct / 100.0,
        }
        scenario_exog = last_exog.copy()
        for col, pct in changes.items():
            scenario_exog[col] = last_exog[col] * (1.0 + pct)

        X_scenario = pd.DataFrame(
            np.tile(scenario_exog.values, (horizon_days, 1)),
            columns=EXOG_COLS,
            index=future_dates,
        )

        baseline_preds = model.predict(X_baseline)
        scenario_preds = model.predict(X_scenario)
        delta = scenario_preds - baseline_preds
        delta_pct = (delta / baseline_preds.abs()) * 100

        last_actual = float(df[target_col].iloc[-1])

        lines = [
            f"## Simulación de escenario — {target_col}",
            f"Último valor real: {last_actual:.2f} (al {last_date.date()})",
            "",
            "### Supuestos del escenario",
            f"  - Price_X: {price_x_change_pct:+.1f}% "
            f"({last_exog['Price_X']:.2f} → {scenario_exog['Price_X']:.2f})",
            f"  - Price_Y: {price_y_change_pct:+.1f}% "
            f"({last_exog['Price_Y']:.2f} → {scenario_exog['Price_Y']:.2f})",
            f"  - Price_Z: {price_z_change_pct:+.1f}% "
            f"({last_exog['Price_Z']:.2f} → {scenario_exog['Price_Z']:.2f})",
            "",
            "### Comparación de pronósticos",
            f"| Métrica     | Baseline | Escenario | Δ abs | Δ % |",
            f"|-------------|----------|-----------|-------|-----|",
            f"| Media       | {baseline_preds.mean():.2f} | {scenario_preds.mean():.2f} | "
            f"{delta.mean():+.2f} | {delta_pct.mean():+.2f}% |",
            f"| Mín         | {baseline_preds.min():.2f} | {scenario_preds.min():.2f} | "
            f"{(scenario_preds.min() - baseline_preds.min()):+.2f} | — |",
            f"| Máx         | {baseline_preds.max():.2f} | {scenario_preds.max():.2f} | "
            f"{(scenario_preds.max() - baseline_preds.max()):+.2f} | — |",
        ]
        return "\n".join(lines)

    except Exception:
        return f"Error al simular escenario: {traceback.format_exc(limit=3)}"


# ---------------------------------------------------------------------------
# Tool 5 — get_shap_explanation
# ---------------------------------------------------------------------------

class GetShapExplanationInput(BaseModel):
    equipment: str = Field(
        description="Equipo a explicar: 'equipo1' o 'equipo2'."
    )
    n_top_features: int = Field(
        default=10,
        ge=3,
        le=30,
        description="Número de features más importantes a mostrar.",
    )
    n_samples: int = Field(
        default=200,
        ge=20,
        le=1000,
        description="Número de observaciones recientes para calcular SHAP.",
    )


@tool(args_schema=GetShapExplanationInput)
def get_shap_explanation(
    equipment: str,
    n_top_features: int = 10,
    n_samples: int = 200,
) -> str:
    """
    Explica qué features (rezagos y estadísticos de materias primas) tienen
    mayor impacto en las predicciones del modelo XGBoost para un equipo.
    Usa SHAP TreeExplainer para calcular valores de Shapley sobre las últimas
    `n_samples` observaciones y devuelve el ranking de importancia con el
    signo medio (positivo = sube el precio, negativo = lo baja).
    """
    try:
        import shap  # noqa: PLC0415

        target_col = _resolve_target(equipment)
        model, X_full, y_full = _get_or_fit_xgboost(target_col)

        X_sample = X_full.iloc[-n_samples:]

        explainer = shap.TreeExplainer(model.booster)
        shap_values = explainer.shap_values(X_sample)  # (n_samples, n_features)

        mean_abs = np.abs(shap_values).mean(axis=0)
        mean_signed = shap_values.mean(axis=0)

        importance_df = pd.DataFrame(
            {"feature": X_sample.columns, "mean_abs_shap": mean_abs, "mean_shap": mean_signed}
        ).sort_values("mean_abs_shap", ascending=False).head(n_top_features)

        last_actual = float(y_full.iloc[-1])
        lines = [
            f"## Explicación SHAP — {target_col}",
            f"Modelo: XGBoost | Últimas {len(X_sample):,} observaciones analizadas",
            f"Precio actual del equipo: {last_actual:.2f}",
            "",
            f"### Top {n_top_features} features por importancia (|SHAP| medio)",
            "| # | Feature | |SHAP| medio | SHAP medio (signo) | Dirección |",
            "|---|---------|------------|-------------------|-----------|",
        ]
        for rank, (_, row) in enumerate(importance_df.iterrows(), 1):
            direction = "↑ Sube precio" if row["mean_shap"] > 0 else "↓ Baja precio"
            lines.append(
                f"| {rank} | {row['feature']} | {row['mean_abs_shap']:.4f} | "
                f"{row['mean_shap']:+.4f} | {direction} |"
            )

        lines += [
            "",
            "**Interpretación**: el |SHAP| medio indica cuánto contribuye en promedio "
            "cada feature a la predicción del modelo, independientemente del signo.",
        ]
        return "\n".join(lines)

    except Exception:
        return f"Error al calcular SHAP: {traceback.format_exc(limit=3)}"


# ---------------------------------------------------------------------------
# Exportación del conjunto de tools
# ---------------------------------------------------------------------------

TOOLS = [
    get_forecast,
    get_historical_data,
    web_search_market_news,
    simulate_scenario,
    get_shap_explanation,
]
