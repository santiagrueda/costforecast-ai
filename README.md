# CostForecast AI

> Forecasting de costos de equipos de construcción basado en precios de materias primas, con un agente conversacional que combina predicciones cuantitativas con contexto de mercado en tiempo real.

![Python](https://img.shields.io/badge/python-3.11%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Tests](https://img.shields.io/badge/tests-pytest-brightgreen)
![Code Style](https://img.shields.io/badge/code%20style-black-000000)
![Linting](https://img.shields.io/badge/linting-ruff-red)

---

## 📋 Tabla de contenidos

1. [Contexto del problema](#-contexto-del-problema)
2. [Solución propuesta](#-solución-propuesta)
3. [Arquitectura](#-arquitectura)
4. [Estructura del proyecto](#-estructura-del-proyecto)
5. [Quickstart](#-quickstart)
6. [Metodología](#-metodología)
7. [Resultados](#-resultados)
8. [El agente de IA](#-el-agente-de-ia)
9. [Próximos pasos](#-próximos-pasos)

---

## 🎯 Contexto del problema

Una empresa constructora debe gestionar el suministro de dos tipos de equipos críticos durante un proyecto con ventana definida. Históricamente, los precios de estos equipos han mostrado volatilidad que genera desviaciones presupuestales recurrentes. La hipótesis gerencial es que los precios dependen de ciertas materias primas del mercado, pero no existe un modelo formal que respalde cuáles son determinantes ni cómo proyectar costos futuros.

**Objetivos**:

- Identificar qué materias primas explican el comportamiento de los precios de cada equipo.
- Proyectar costos futuros con intervalos de confianza.
- Ofrecer un agente conversacional que enriquezca las predicciones con contexto de mercado actualizado.

## 💡 Solución propuesta

| Capa | Componente | Tecnología |
|---|---|---|
| **Datos** | Ingesta, validación, auditoría de calidad | pandas, pydantic |
| **Análisis** | EDA estadístico, selección de features | statsmodels, scipy |
| **Modelado** | Competencia entre SARIMAX, Prophet, XGBoost | statsmodels, prophet, xgboost |
| **Explainability** | Atribución de importancia por variable | SHAP |
| **Pronóstico** | Predicción + intervalos de confianza + Monte Carlo | numpy, scipy |
| **Agente IA** | Conversacional con tool-use y búsqueda web | LangGraph, Claude, Tavily |
| **UI** | App de demostración para el evaluador | Streamlit |
| **API** | Endpoint de serving del modelo | FastAPI |
| **Cloud** | Arquitectura documentada | AWS (S3, Lambda, SageMaker, Bedrock) |
| **BI** | Dashboard ejecutivo | Power BI |

## 🏛️ Arquitectura

La arquitectura propuesta en AWS se documenta en [`infra/architecture.png`](infra/architecture.png). Incluye:

- **S3** para data lake (raw + processed).
- **AWS Glue** para ETL.
- **SageMaker** para entrenamiento reproducible.
- **Lambda + API Gateway** para serving del modelo.
- **App Runner** para la UI del agente conversacional.
- **Bedrock** para acceder a Claude como LLM del agente.
- **Secrets Manager, IAM, CloudWatch** para seguridad y observabilidad.

## 📁 Estructura del proyecto

```
costforecast-ai/
├── data/              # Datos (gitignored)
│   ├── raw/           # CSV original
│   ├── processed/     # Parquet limpio
│   └── forecasts/     # Predicciones
├── notebooks/         # EDA, modelado, storytelling analítico
├── src/costforecast/  # Código fuente
│   ├── data/          # Carga, validación, calidad
│   ├── features/      # Feature engineering
│   ├── models/        # SARIMAX, Prophet, XGBoost
│   ├── evaluation/    # Backtesting, métricas
│   ├── forecasting/   # Pronóstico + Monte Carlo
│   ├── explainability/# SHAP
│   └── agent/         # LangGraph agent
├── app/               # Streamlit
├── api/               # FastAPI
├── infra/             # Arquitectura AWS + Terraform
├── tests/             # Pytest
├── docs/              # Informe, explicaciones
└── powerbi/           # Dashboard BI
```

## 🚀 Quickstart

### Requisitos

- Python 3.11+
- Claves de API para [Anthropic](https://console.anthropic.com/) y [Tavily](https://tavily.com/) (para el agente)

### Instalación

```bash
# Clonar repo
git clone https://github.com/santiagrueda/costforecast-ai.git
cd costforecast-ai

# Crear entorno virtual
python -m venv .venv
source .venv/bin/activate  # En Windows: .venv\Scripts\activate

# Instalar dependencias
pip install -e ".[dev]"

# Configurar variables de entorno
cp .env.example .env
# Editar .env con tus API keys

# Ejecutar tests
make test
```

### Ejecución del pipeline

```bash
# 1. Colocar el dataset en data/raw/
# 2. Correr EDA
make eda

# 3. Entrenar modelos
make train

# 4. Generar pronósticos
make forecast

# 5. Lanzar el agente conversacional
make app
# Abre http://localhost:8501

# 6. Lanzar la API REST
make api
# Docs interactivas en http://localhost:8000/docs
```

## 📊 Metodología

El pipeline sigue un flujo iterativo:

1. **Auditoría de calidad** — Completitud, outliers, estacionariedad, gaps temporales.
2. **EDA dirigido** — Correlaciones (Pearson + Spearman), análisis de lags, visualización.
3. **Selección de features** — Granger causality + Lasso para identificar materias primas relevantes.
4. **Modelado competitivo** — Entrenamiento de varios modelos con walk-forward validation.
5. **Selección del ganador** — Por MAPE de backtesting en cada equipo.
6. **Pronóstico con incertidumbre** — Intervalos de confianza + simulación Monte Carlo.
7. **Explainability** — SHAP para entender qué factor pesa más en cada predicción.

## 📈 Resultados

*Esta sección se completará una vez ejecutado el pipeline sobre el dataset real.*

## 🤖 El agente de IA

El agente implementa un ciclo **ReAct (Reason + Act)** con las siguientes herramientas:

| Tool | Descripción |
|---|---|
| `get_forecast(equipo, meses)` | Consulta al modelo de forecasting |
| `get_historical_data(fecha_inicio, fecha_fin)` | Accede a datos históricos |
| `web_search_market_news(query)` | Busca noticias del sector (Tavily) |
| `simulate_scenario(materia_prima, shock_pct)` | What-if análisis |
| `get_shap_explanation(prediccion_id)` | Explicación de una predicción |

**Diferencia vs IA convencional**: un modelo tradicional recibe input y produce output. Un agente percibe, razona, decide qué herramienta usar, ejecuta acciones y ajusta su siguiente paso según el resultado. Tiene **autonomía, uso de herramientas, memoria y capacidad de acción**.

Ejemplo de pregunta que solo un agente puede responder:

> *"Si el cobre sube 15% el próximo mes por la huelga en Chile, ¿cómo afecta el presupuesto de Q2?"*

El agente buscará noticias para confirmar la huelga, simulará el escenario, consultará SHAP para cuantificar el impacto por equipo, y responderá con números respaldados.

## 🔮 Próximos pasos

Documentados en [`docs/informe_ejecutivo.pdf`](docs/informe_ejecutivo.pdf), sección "Futuros ajustes o mejoras".

## 👤 Autor

**Santiago Rueda** — Data Scientist / ML Engineer /
[LinkedIn](https://linkedin.com/in/santiagorueda) · [GitHub](https://github.com/santiagrueda)

## 📄 Licencia

MIT
