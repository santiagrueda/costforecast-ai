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
| **Agente IA (Claude)** | Conversacional con tool-use y búsqueda web | LangGraph, Claude, Tavily |
| **Agente IA (Open Source)** | Agente 100% local, sin API keys | LangGraph, Gemma 4, Ollama, DuckDuckGo |
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
- **Agente Claude**: claves de API para [Anthropic](https://console.anthropic.com/) y [Tavily](https://tavily.com/)
- **Agente Open Source**: [Ollama](https://ollama.com/download) instalado localmente con Gemma 4 (sin API keys)

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

> Todos los resultados fueron computados sobre el dataset real (`historico_equipos.csv`).  
> 3 530 observaciones diarias en días hábiles · 2010-01-04 → 2023-08-31 · 0 nulos · 0 gaps > 5 días.

---

### 1. Análisis descriptivo

| Serie | Media | Std | CV (%) | Mín | Máx | Asimetría | Curtosis |
|---|---|---|---|---|---|---|---|
| Price_X | 78.09 | 25.19 | 32.3 | 19.33 | 127.98 | 0.08 | −1.14 |
| Price_Y | 555.53 | 138.49 | 24.9 | 257.50 | 1 062.37 | 0.48 | 0.73 |
| Price_Z | 2 037.43 | 373.14 | 18.3 | 1 421.50 | 3 984.00 | 1.14 | 1.99 |
| **Price_Equipo1** | **460.04** | **113.68** | **24.7** | 208.34 | 855.32 | 0.41 | 0.59 |
| **Price_Equipo2** | **889.98** | **170.04** | **19.1** | 566.00 | 1 703.96 | 0.83 | 1.07 |

> **CV (coeficiente de variación)**: X es la materia prima más volátil (32.3 %). Price_Z la más estable (18.3 %). Los equipos tienen variabilidad similar a Y y Z, sus drivers naturales.

**Tendencia (CAGR 2010–2023) y volatilidad anualizada:**

| Serie | Precio inicio | Precio fin | Variación total | CAGR | Vol. anual |
|---|---|---|---|---|---|
| Price_X | 80.1 | 86.9 | +8.4 % | +0.59 %/año | 35.9 % |
| Price_Y | 527.5 | 545.0 | +3.3 % | +0.24 %/año | 16.6 % |
| Price_Z | 2 225.2 | 2 165.2 | −2.7 % | −0.20 %/año | 21.8 % |
| **Price_Equipo1** | 434.7 | 451.7 | +3.9 % | +0.28 %/año | **42.2 %** |
| **Price_Equipo2** | 931.7 | 955.4 | +2.5 % | +0.18 %/año | **42.8 %** |

> Insight clave: los equipos tienen **CAGR casi nulo** (< 0.3 %/año) pero **volatilidad intradécada del 42 %**. Esto explica por qué el problema presupuestal no es de tendencia sino de oscilación: el precio puede desviarse ±40 % del promedio en cualquier año.

---

### 2. Estacionariedad (Test ADF)

Todas las series son **I(1)**: no estacionarias en nivel pero estacionarias tras una primera diferencia.

| Serie | ADF nivel | p-valor | ADF Δ(1) | p-valor | Orden |
|---|---|---|---|---|---|
| Price_X | −1.705 | 0.4284 | −27.455 | < 0.001 | I(1) |
| Price_Y | −2.530 | 0.1083 | −8.727 | < 0.001 | I(1) |
| Price_Z | −2.203 | 0.2052 | −14.724 | < 0.001 | I(1) |
| Price_Equipo1 | −2.390 | 0.1445 | −16.946 | < 0.001 | I(1) |
| Price_Equipo2 | −1.906 | 0.3290 | −29.707 | < 0.001 | I(1) |

> Implicación de modelado: se justifica la diferenciación d=1 en SARIMAX y el uso de features de primera y segunda diferencia en XGBoost. Los modelos que no manejan series I(1) (ej. OLS directo) producirían regresiones espúrias.

---

### 3. Correlación — Pearson y Spearman

| Materia prima | Equipo 1 (Pearson) | Equipo 1 (Spearman) | Equipo 2 (Pearson) | Equipo 2 (Spearman) |
|---|---|---|---|---|
| Price_X | 0.523 | 0.624 | 0.530 | 0.604 |
| Price_Y | **0.997** | **0.994** | 0.913 | 0.927 |
| Price_Z | 0.844 | 0.866 | **0.983** | **0.980** |

> La consistencia entre Pearson y Spearman descarta que las correlaciones altas sean artefactos de outliers: la relación Y→Equipo1 y Z→Equipo2 es robusta tanto en magnitud como en rango.

**Drivers identificados:**
- **Equipo 1**: dominado por Price_Y (r = 0.997, ρ = 0.994). La relación es casi perfectamente lineal.
- **Equipo 2**: dominado por Price_Z (r = 0.983, ρ = 0.980). Price_Y también es relevante (r = 0.913) pero secundario.
- **Price_X**: correlación moderada con ambos (~0.52–0.62). Contribuye pero no es driver principal de ninguno.

---

### 4. Causalidad de Granger

Test F sobre series diferenciadas · lags 1–5 · `***` p < 0.001 · `**` p < 0.01 · `*` p < 0.05

| Causa → Efecto | Lag óptimo | p-valor | Significancia |
|---|---|---|---|
| Price_Y → Equipo 1 | 5 | < 0.0001 | *** |
| Price_Z → Equipo 1 | 4 | < 0.0001 | *** |
| Price_X → Equipo 1 | 4 | 0.0001 | *** |
| Price_Y → Equipo 2 | 4 | < 0.0001 | *** |
| Price_Z → Equipo 2 | 5 | < 0.0001 | *** |
| Price_X → Equipo 2 | 4 | < 0.0001 | *** |

> Todas las materias primas Granger-causan los precios de ambos equipos con alta significancia estadística. Con n = 3 530, incluso efectos pequeños resultan significativos: la **magnitud económica** (correlación de Pearson) es el criterio principal de selección, no el p-valor. Esto refuerza la elección de Y para Equipo 1 y Z para Equipo 2.

> El lag óptimo de 4–5 días (~1 semana hábil) indica que las variaciones en materias primas se transmiten al precio de los equipos con aproximadamente **una semana de rezago**, información útil para estrategias de cobertura.

---

### 5. Correlación cruzada por rezagos (CCF)

Calculada sobre primeras diferencias para eliminar tendencia común.

| Par | CCF lag=0 | CCF lag=1 | CCF lag=2 | CCF lag=5 |
|---|---|---|---|---|
| Price_Y → Price_Equipo1 | **0.386** | −0.006 | 0.059 | 0.057 |
| Price_Z → Price_Equipo2 | **0.431** | −0.045 | 0.033 | −0.010 |

> La correlación cruzada es máxima en **lag=0** (contemporánea) y decae a valores cercanos a cero para lags > 1. Esto indica que ambos mercados reaccionan en el **mismo día hábil**, no con días de anticipación. El modelo SARIMAX con exog contemporáneo es por tanto la especificación correcta; no hay ganancia de añadir lags de materias primas como regresores.

---

### 6. Backtesting walk-forward — MAPE (%)

Validación expanding-window · 5 folds · horizontes 1 / 5 / 20 días hábiles · exog oracle.

**Equipo 1 — Price_Equipo1**

| Modelo | h = 1 día | h = 5 días | h = 20 días |
|---|---|---|---|
| Persistence (baseline) | 3.34 % | 2.24 % | 1.33 % |
| SARIMAX(1,1,1) + exog | 1.12 % | 1.60 % | 1.20 % |
| **Prophet + regressors** | **1.18 %** | **1.46 %** | **1.17 %** |

**Equipo 2 — Price_Equipo2**

| Modelo | h = 1 día | h = 5 días | h = 20 días |
|---|---|---|---|
| Persistence (baseline) | 2.41 % | 1.72 % | 3.65 % |
| SARIMAX(1,1,1) + exog | 1.81 % | 1.47 % | 1.53 % |
| **Prophet + regressors** | **1.66 %** | **1.32 %** | **1.44 %** |

**Modelo ganador: Prophet** — supera a SARIMAX en 5 de 6 combinaciones equipo × horizonte. SARIMAX gana únicamente en Equipo 1 a h=1 (MAPE 1.12 % vs 1.18 %). Se usa en la interfaz por sus intervalos de confianza paramétricos nativos.

**Reducción de error vs baseline:**

| Equipo | Horizonte | Baseline | Ganador | Mejora |
|---|---|---|---|---|
| Equipo 1 | 1 día | 3.34 % | 1.18 % | **−65 %** |
| Equipo 1 | 20 días | 1.33 % | 1.17 % | −12 % |
| Equipo 2 | 1 día | 2.41 % | 1.66 % | **−31 %** |
| Equipo 2 | 20 días | 3.65 % | 1.44 % | **−61 %** |

## 🤖 Los agentes de IA

Ambos agentes implementan el ciclo **ReAct (Reason + Act)** y comparten las mismas 5 herramientas:

| Tool | Descripción |
|---|---|
| `get_forecast(equipo, meses)` | Consulta al modelo de forecasting |
| `get_historical_data(fecha_inicio, fecha_fin)` | Accede a datos históricos |
| `web_search_market_news(query)` | Busca noticias del sector |
| `simulate_scenario(materia_prima, shock_pct)` | What-if análisis |
| `get_shap_explanation(equipo, n_top)` | Explicación de importancia de variables |

### Comparativa de arquitecturas

| Dimensión | Agente Claude | Agente Open Source |
|---|---|---|
| **LLM** | Claude Sonnet (Anthropic) | Gemma 4 (Google / Ollama) |
| **Licencia** | Propietario | Apache 2.0 |
| **Costo** | API de pago | Gratuito |
| **Web search** | Tavily (API key) | DuckDuckGo (free) |
| **Infraestructura** | Cloud | Local (CPU) |
| **Velocidad** | ~2 s/respuesta | ~15–40 s en CPU i7 13G |
| **Privacidad de datos** | Enviados a Anthropic | 100% local |

### Setup del agente open source

```bash
# 1. Instalar Ollama
#    https://ollama.com/download

# 2. Levantar el servidor
ollama serve

# 3. Crear el modelo custom (optimizado para tool calling)
ollama create costforecast-gemma4 -f infra/Modelfile

# 4. Lanzar la app — el tab aparece automáticamente
make app
```

**Diferencia vs IA convencional**: un agente percibe, razona, decide qué herramienta usar, ejecuta acciones y ajusta su siguiente paso según el resultado — no solo genera texto.

Ejemplo de pregunta que solo un agente puede responder:

> *"Si el acero sube 15% el próximo mes, ¿cómo afecta el presupuesto de Q2 del Equipo 2?"*

El agente buscará noticias de mercado, simulará el escenario con los coeficientes reales del EDA, consultará SHAP para cuantificar el impacto, y responderá con cifras concretas.

## 🔮 Próximos pasos

Documentados en [`docs/informe_ejecutivo.pdf`](docs/informe_ejecutivo.pdf), sección "Futuros ajustes o mejoras".

## 👤 Autor

**Santiago Rueda** — Data Scientist / ML Engineer /
[LinkedIn](https://linkedin.com/in/santiagorueda) · [GitHub](https://github.com/santiagrueda)

## 📄 Licencia

MIT
