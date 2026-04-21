"""Módulo de forecasting: generación de pronósticos con intervalos de confianza."""

from __future__ import annotations

__all__ = ["generate_forecasts", "save_forecasts"]


def __getattr__(name: str):
    if name in ("generate_forecasts", "save_forecasts"):
        from costforecast.forecasting.generate import generate_forecasts, save_forecasts  # noqa: PLC0415
        return {"generate_forecasts": generate_forecasts, "save_forecasts": save_forecasts}[name]
    raise AttributeError(name)
