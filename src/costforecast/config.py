"""
Configuración centralizada del proyecto.

Usa pydantic-settings para cargar variables de entorno con validación de tipos.
Todas las rutas y parámetros del proyecto se acceden vía `settings`.

Ejemplo:
    from costforecast.config import settings
    print(settings.raw_data_dir)
"""

from __future__ import annotations

from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Configuración global del proyecto, cargada desde .env."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ---------------- LLM ----------------
    anthropic_api_key: str = Field(default="", description="API key de Anthropic")
    tavily_api_key: str = Field(default="", description="API key de Tavily para búsqueda web")
    claude_model: str = Field(default="claude-sonnet-4-5-20250929")
    claude_max_tokens: int = Field(default=2048, ge=1, le=8192)
    claude_temperature: float = Field(default=0.2, ge=0.0, le=1.0)

    # ---------------- Paths ----------------
    data_dir: Path = Field(default=Path("./data"))
    raw_data_dir: Path = Field(default=Path("./data/raw"))
    processed_data_dir: Path = Field(default=Path("./data/processed"))
    forecasts_dir: Path = Field(default=Path("./data/forecasts"))
    dataset_filename: str = Field(default="dataset_precios.csv")

    # ---------------- AWS ----------------
    aws_region: str = Field(default="us-east-1")
    aws_s3_bucket: str = Field(default="costforecast-ai-dataknow")

    # ---------------- Logging ----------------
    log_level: str = Field(default="INFO")
    log_file: Path = Field(default=Path("./logs/costforecast.log"))

    # ---------------- Modelado ----------------
    random_seed: int = Field(default=42)
    test_size: float = Field(default=0.2, gt=0.0, lt=1.0)
    forecast_horizon_months: int = Field(default=6, ge=1, le=24)

    # ---------------- Validators ----------------
    @field_validator(
        "data_dir", "raw_data_dir", "processed_data_dir", "forecasts_dir", mode="before"
    )
    @classmethod
    def _ensure_path(cls, v: str | Path) -> Path:
        return Path(v).resolve()

    @property
    def raw_dataset_path(self) -> Path:
        """Ruta completa al dataset original."""
        return self.raw_data_dir / self.dataset_filename

    @property
    def processed_dataset_path(self) -> Path:
        """Ruta al dataset procesado (parquet)."""
        return self.processed_data_dir / "dataset_clean.parquet"


# Singleton accesible en todo el proyecto
settings = Settings()
