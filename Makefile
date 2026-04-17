# =============================================================================
# CostForecast AI - Makefile
# Comandos estandarizados para desarrollo, testing y deployment
# =============================================================================

.PHONY: help install install-dev clean lint format test coverage eda train forecast app api docs

PYTHON := python
PIP := pip
PROJECT_NAME := costforecast-ai

help:  ## Mostrar esta ayuda
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install:  ## Instalar dependencias de producción
	$(PIP) install -e .

install-dev:  ## Instalar dependencias de desarrollo
	$(PIP) install -e ".[dev]"
	pre-commit install

clean:  ## Limpiar archivos temporales
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	rm -rf build/ dist/ *.egg-info/ htmlcov/ .coverage

lint:  ## Ejecutar linters (ruff + mypy)
	ruff check src/ tests/
	mypy src/

format:  ## Formatear código (ruff + black)
	ruff check --fix src/ tests/
	black src/ tests/

test:  ## Ejecutar tests
	pytest tests/ -v

coverage:  ## Ejecutar tests con reporte de cobertura
	pytest tests/ --cov=src/costforecast --cov-report=html --cov-report=term

eda:  ## Ejecutar EDA automático
	$(PYTHON) -m costforecast.data.profiling

train:  ## Entrenar todos los modelos
	$(PYTHON) -m costforecast.models.train_all

forecast:  ## Generar pronósticos
	$(PYTHON) -m costforecast.forecasting.generate

app:  ## Iniciar app Streamlit (agente conversacional)
	streamlit run app/streamlit_app.py

api:  ## Iniciar API FastAPI
	uvicorn api.main:app --reload --port 8000

docs:  ## Construir documentación
	@echo "Documentación disponible en docs/"
