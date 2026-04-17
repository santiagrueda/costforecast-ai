"""
CostForecast AI - Forecasting de costos de equipos basado en materias primas.

Este paquete provee:
- Pipeline de datos (carga, validación, limpieza)
- Análisis estadístico y selección de features
- Modelos de forecasting (SARIMAX, Prophet, XGBoost)
- Explainability con SHAP
- Agente conversacional con LangGraph + Claude
- Simulación Monte Carlo para escenarios

Autor: Santiago Rueda
"""

__version__ = "0.1.0"
__author__ = "Santiago Rueda"

from costforecast.config import settings
from costforecast.logger import logger

__all__ = ["settings", "logger", "__version__"]
