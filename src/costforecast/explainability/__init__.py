"""Módulo de explicabilidad: wrappers SHAP para los modelos de forecasting."""

from __future__ import annotations

__all__ = ["ShapExplainer"]


def __getattr__(name: str):
    if name == "ShapExplainer":
        from costforecast.explainability.shap_wrapper import ShapExplainer  # noqa: PLC0415
        return ShapExplainer
    raise AttributeError(name)
