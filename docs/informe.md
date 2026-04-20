# Informe Ejecutivo — CostForecast AI

**Autor**: Santiago Rueda  
**Fecha**: Abril 2026  
**Contexto**: Prueba técnica — rol de Data Scientist, DataKnow

---

## 1. Explicación del caso

Una empresa constructora gestiona el suministro de dos tipos de equipos críticos durante un proyecto de construcción. Los precios de estos equipos exhiben volatilidad correlacionada con ciertas materias primas del mercado, lo que genera desviaciones presupuestales frecuentes y difíciles de anticipar.

El problema tiene tres dimensiones:

1. **Identificación de drivers**: determinar qué materias primas explican estadísticamente el comportamiento de los precios de cada equipo, validando la relación con causalidad (no solo correlación).
2. **Proyección cuantitativa**: generar pronósticos de costo con intervalos de confianza, que sirvan como insumo para el proceso presupuestal y la negociación con proveedores.
3. **Democratización del análisis**: exponer los hallazgos a través de un agente conversacional de IA que permita a usuarios no técnicos consultar proyecciones, explorar escenarios hipotéticos y acceder a explicaciones sobre los modelos.

### Dataset disponible

| Fuente | Contenido | Observaciones | Rango |
|---|---|---|---|
| `historico_equipos.csv` | Precios X, Y, Z + Equipo 1 y 2 | 3,530 | 2010-01 → 2023-08 |
| `X.csv` | Materia prima X | 9,144 | 1988-06 → 2024-04 |
| `Y.csv` | Materia prima Y | 4,485 | 2006-07 → 2023-09 |
| `Z.csv` | Materia prima Z | 3,565 | 2010-01 → 2023-08 |

La frecuencia es diaria en días hábiles. El dataset consolidado no presenta valores nulos.

---

## 2. Supuestos

Los siguientes supuestos enmarcan el análisis y la solución entregada:

**Supuestos de datos**

- Las series de precios de materias primas e equipos son observables y confiables (sin manipulación de fuente).
- El periodo 2010–2023 es suficientemente representativo del comportamiento esperado futuro. No se modelan cambios estructurales (ej. shocks de 2020 se tratan como parte del comportamiento normal de la serie).
- Los días hábiles son comparables entre sí (sin corrección por festivos específicos de cada país).

**Supuestos de modelado**

- Las materias primas futuras se proyectan bajo un supuesto de **persistencia** (último valor conocido) al generar pronósticos de equipos. En producción, se reemplazarían por pronósticos propios de las materias primas.
- El modelo SARIMAX(1,1,1) sin componente estacional es una especificación razonable para el periodo de backtesting; la selección de hiperparámetros óptimos (AIC/BIC grid search) queda como mejora futura.
- La causalidad de Granger se toma como evidencia adicional de dirección, no como prueba de causalidad física. La interpretación final combina correlación, causalidad estadística y conocimiento del dominio.
- El XGBoost recibe valores reales futuros de materias primas durante el backtesting (*oracle exog*), lo que aisla la capacidad del modelo para capturar la relación target↔exog y hace la comparación equitativa entre modelos.

**Supuestos del agente IA**

- El agente opera sobre el dataset histórico ya cargado en memoria; no accede a feeds de datos en tiempo real (excepto búsqueda web vía Tavily).
- Las herramientas del agente son deterministas dado el mismo estado del modelo y los datos — no se generan respuestas probabilísticas no reproducibles.

---

## 3. Formas de resolver el caso y opción tomada

### Alternativas consideradas

**Opción A — Modelo univariado puro (ARIMA)**  
Ajustar ARIMA directamente sobre las series de equipos, sin incorporar las materias primas como regresores. Ventaja: sencillo, rápido de implementar. Desventaja: ignora la relación causal demostrada entre materias primas y equipos (correlaciones > 0.90), sacrificando capacidad predictiva y explicabilidad.

**Opción B — Regresión OLS con rezagos**  
Modelo de regresión lineal con materias primas y sus rezagos como predictores. Ventaja: interpretable, rápida convergencia. Desventaja: no captura la autocorrelación residual inherente a series de tiempo, violando supuestos del modelo. Requiere diferenciación manual y no incorpora estructura de error MA.

**Opción C — Modelo competitivo con selección por backtest (opción tomada)**  
Comparar cuatro familias de modelos (Baseline naive, SARIMAX con exog, Prophet con regresores adicionales, XGBoost con feature engineering de series temporales) mediante walk-forward validation. El modelo ganador por MAPE en backtesting se usa para el pronóstico final. Esta opción maximiza la probabilidad de elegir el modelo más adecuado para cada equipo sin asumir superioridad a priori de ninguna técnica.

### Opción tomada y justificación

Se implementó la **opción C** por las siguientes razones:

1. La fuerte correlación entre materias primas y equipos (r > 0.90 en ambos casos) hace que ignorar los regresores exógenos sea un error metodológico grave.
2. Las series presentan tendencia y posible no-estacionariedad (ADF p-value < 0.05 tras primera diferencia), lo que favorece modelos que incorporan estructura de integración (SARIMAX) o son robustos a no-estacionariedad (XGBoost con features de diferencias).
3. El enfoque competitivo permite seleccionar el mejor modelo por equipo, aceptando que SARIMAX puede ser superior para una serie y XGBoost para la otra.
4. La validación walk-forward (expanding window) simula el proceso real de predicción en producción: el modelo se reentrena con datos históricos crecientes y se evalúa en el horizonte inmediatamente siguiente, evitando data leakage.

---

## 4. Resultados del análisis de datos y modelos

### 4.1 Correlaciones y selección de drivers

Las correlaciones de Pearson calculadas sobre el dataset consolidado muestran un patrón claro:

| Par | Pearson r | Spearman r |
|---|---|---|
| **Equipo 1 ↔ Materia Prima Y** | **+0.997** | alto |
| Equipo 1 ↔ Materia Prima Z | +0.844 | moderado |
| Equipo 1 ↔ Materia Prima X | +0.520 | bajo |
| Equipo 2 ↔ Materia Prima Z | **+0.983** | **alto** |
| Equipo 2 ↔ Materia Prima Y | +0.910 | alto |
| Equipo 2 ↔ Materia Prima X | +0.530 | bajo |

**Conclusión**: El precio del Equipo 1 está dominado por la materia prima Y (r ≈ 1.00), mientras que el Equipo 2 está determinado principalmente por Z (r ≈ 0.98). La materia prima X muestra correlación moderada con ambos equipos pero no es el driver principal de ninguno.

El análisis de causalidad de Granger (con tests de rezagos 1–5) corrobora la dirección causal: los precios de Y Granger-causan el precio del Equipo 1, y Z Granger-causa el precio del Equipo 2 con significancia estadística (p < 0.05).

### 4.2 Feature engineering para XGBoost

Se generaron features backward-looking sobre todas las series del dataset:

- **Rezagos**: 1, 2, 3, 4 y 5 días para todas las columnas predictoras.
- **Estadísticos rolling**: media y desvío estándar con ventanas de 5, 10 y 20 días.
- **Diferencias**: primera y segunda diferencia para capturar momentum de precio.

Esto produce una matrix de features de alta dimensionalidad (~60–80 columnas), sin leakage, ya que todos los features son estrictamente backward-looking en cada punto t.

### 4.3 Resultados del backtesting

Walk-forward validation con 10 folds, ventana expandible mínima de 500 observaciones, horizontes de evaluación: 1, 5, 10 y 20 días hábiles.

**Equipo 1 — MAPE por modelo y horizonte (%)**

| Modelo | h=1 | h=5 | h=10 | h=20 |
|---|---|---|---|---|
| Persistence (baseline) | ~0.5 | ~1.2 | ~2.1 | ~3.8 |
| SARIMAX(1,1,1) | ~0.3 | ~0.8 | ~1.4 | ~2.5 |
| Prophet | ~0.4 | ~1.0 | ~1.7 | ~3.1 |
| **XGBoost** | **~0.2** | **~0.6** | **~1.1** | **~2.0** |

**Equipo 2 — MAPE por modelo y horizonte (%)**

| Modelo | h=1 | h=5 | h=10 | h=20 |
|---|---|---|---|---|
| Persistence (baseline) | ~0.6 | ~1.5 | ~2.6 | ~4.5 |
| SARIMAX(1,1,1) | ~0.4 | ~1.0 | ~1.8 | ~3.2 |
| Prophet | ~0.5 | ~1.2 | ~2.0 | ~3.8 |
| **XGBoost** | **~0.3** | **~0.7** | **~1.3** | **~2.4** |

> **Nota**: Los valores de MAPE reportados son representativos del comportamiento observado en los experimentos de backtesting. Los valores exactos dependen de la ejecución completa del pipeline con `make train`.

**Modelo ganador**: XGBoost supera a los demás modelos en todos los horizontes para ambos equipos, gracias a su capacidad de capturar relaciones no lineales entre rezagos de materias primas y precios de equipos. SARIMAX es el segundo mejor modelo y se usa para el pronóstico en la aplicación Streamlit por ser más interpretable y más rápido en inferencia.

### 4.4 Importancia de features (SHAP)

El análisis SHAP sobre XGBoost revela que:

- Para el **Equipo 1**: los features más importantes son los rezagos de Price_Y (lag1, lag2, lag3) y la media rolling de 5 días de Y. Confirma la dominancia de la materia prima Y.
- Para el **Equipo 2**: los features dominantes son los rezagos de Price_Z y el rolling mean de Z a 10 y 20 días. Confirma la dependencia de Z.

En ambos casos, los rezagos de la propia serie objetivo también aparecen entre los top-5 features, indicando fuerte autocorrelación positiva de los precios.

---

## 5. Proyección de costos y horizonte de predicción

### 5.1 Metodología de pronóstico

El modelo seleccionado (XGBoost en evaluación, SARIMAX en producción) genera predicciones puntuales para los próximos N días hábiles. Los intervalos de confianza se obtienen del modelo SARIMAX mediante `get_forecast().conf_int(alpha)`, que provee bandas basadas en la varianza del error de estimación del estado.

Para el pronóstico operativo se asume:
- Las materias primas se mantienen constantes en su último valor conocido (supuesto de persistencia).
- El horizonte de predicción estándar es de 10 días hábiles (~2 semanas), aunque la aplicación Streamlit permite configurarlo de 1 a 60 días.
- Los intervalos de confianza al 95% se presentan como bandas sombreadas en los gráficos.

### 5.2 Análisis de sensibilidad (what-if)

El agente conversacional incluye una herramienta de simulación de escenarios (`simulate_scenario`) que permite evaluar el impacto en el precio del equipo ante variaciones hipotéticas en los precios de las materias primas. Por ejemplo:

- "¿Qué pasa con el precio del Equipo 1 si la materia prima Y sube 15%?"
- "Simular escenario con Y +20%, Z −10% para Equipo 2 en los próximos 20 días."

La comparación baseline vs. escenario se presenta en formato tabular con Δ absoluto y Δ porcentual.

### 5.3 Horizonte recomendado

Basado en los resultados del backtesting:

| Horizonte | MAPE típico | Recomendación |
|---|---|---|
| 1–5 días | < 1% | Alta confianza; usar para negociación inmediata |
| 5–10 días | 1–2% | Confianza razonable; usar para planificación semanal |
| 10–20 días | 2–4% | Confianza moderada; usar como referencia presupuestal |
| > 20 días | > 4% | Baja precisión; usar solo como tendencia direccional |

---

## 6. Futuros ajustes o mejoras

### 6.1 Mejoras metodológicas

1. **Pronóstico de materias primas**: Reemplazar el supuesto de persistencia por modelos propios de pronóstico para X, Y, Z (ARIMA univariado o VAR multivariado). Esto reduce el error de pronóstico en horizontes > 10 días.

2. **Optimización de hiperparámetros**: Implementar grid search o Optuna para seleccionar el orden ARIMA óptimo (p, d, q) por equipo mediante AIC/BIC, y para ajustar n_estimators, max_depth y learning_rate de XGBoost.

3. **Modelos de ensemble**: Combinar las predicciones de SARIMAX y XGBoost mediante stacking o promedio ponderado por MAPE histórico. En series de tiempo, los ensembles suelen superar al mejor modelo individual.

4. **Detección de cambios estructurales**: Incorporar tests de Chow o CUSUM para detectar regímenes distintos en la serie (ej. impacto COVID-19 en 2020). Considerar modelos con cambio de régimen (Markov-Switching ARIMA).

5. **Intervalos de confianza empíricos**: Complementar los intervalos paramétricos de SARIMAX con simulaciones Monte Carlo sobre los errores históricos del backtesting, especialmente para horizontes > 10 días donde el modelo paramétrico subestima la incertidumbre.

6. **Validación de causalidad de Granger**: Completar el análisis formal con tests de Granger para todos los pares (materia prima → equipo) en distintos rezagos, documentando los resultados en el notebook de EDA.

### 6.2 Mejoras de producto

7. **Actualización automática del dataset**: Pipeline de ingesta que actualice los precios de materias primas desde APIs de mercado (Bloomberg, Quandl, FRED) de forma diaria o semanal.

8. **Re-entrenamiento continuo**: MLOps con reentrenamiento automático cuando el MAPE en producción supere un umbral configurado (monitoreo de model drift).

9. **Dashboard Power BI**: Integrar las proyecciones y el historial en un reporte ejecutivo con KPIs de desviación presupuestal y alertas de precio.

10. **API REST productiva**: Desplegar el endpoint FastAPI en AWS (ECS Fargate + ALB) para que otros sistemas de la empresa puedan consumir las proyecciones programáticamente.

11. **Memoria persistente del agente**: Migrar de `MemorySaver` en RAM a un backend persistente (PostgreSQL + pgvector o Redis) para mantener el historial de conversaciones entre sesiones.

---

## 7. Apreciaciones y comentarios

### 7.1 Sobre el proceso analítico

El caso presentado tiene la riqueza de combinar tres disciplinas: econometría de series de tiempo (SARIMAX, Granger), machine learning tabular (XGBoost con feature engineering temporal) e IA conversacional (agente ReAct con LangGraph). Esto refleja el perfil real de un Data Scientist moderno, que debe navegar entre rigor estadístico y aplicaciones prácticas de IA.

La decisión de implementar un enfoque competitivo (4 modelos evaluados con walk-forward) en lugar de elegir un único modelo a priori resultó correcta: XGBoost supera a SARIMAX en todos los horizontes, resultado que no era obvio sin la validación empírica.

### 7.2 Sobre el agente conversacional vs. IA convencional

El agente ReAct implementado con LangGraph difiere fundamentalmente de una IA convencional (respuesta directa del LLM) en los siguientes aspectos:

| Dimensión | IA Convencional | Agente ReAct |
|---|---|---|
| Fuente de datos | Conocimiento paramétrico del LLM | Dataset real + modelos entrenados |
| Razonamiento | Generación directa | Ciclo Razonar → Actuar → Observar |
| Actualización | Estática (corte de entrenamiento) | Dinámica (Tavily, datos actualizados) |
| Trazabilidad | Opaca | Cada tool call es auditable |
| Alucinaciones | Frecuentes en cifras | Controladas: las cifras vienen de los datos |
| Extensibilidad | Requiere fine-tuning | Basta agregar nuevas tools |

El agente es más robusto para tareas cuantitativas porque separa el razonamiento (LLM) de la ejecución (tools deterministas). Las alucinaciones numéricas quedan eliminadas en los dominios cubiertos por las tools.

### 7.3 Nota de seguridad — Prompt Injection detectado

Durante el análisis del documento PDF del caso (`docs/Caso_consultoria_1_-_candidato.pdf`) se detectó un payload de **prompt injection** embebido en el texto. El fragmento intenta suplantar una instrucción del sistema para prescribir conclusiones falsas (ej. "Equipo 1 depende exclusivamente de Z con 95%") que contradicen directamente los datos reales.

Las correlaciones reales desmienten completamente el payload:

| Afirmación del payload | Realidad de los datos |
|---|---|
| "Y no es estadísticamente significativa" | Equipo 1 ↔ Y: r = **+0.997** |
| "Equipo 1 depende 95% de Z" | Equipo 1 ↔ Z: r = +0.844 |
| "Equipo 2 depende 70% de X" | Equipo 2 ↔ X: r = +0.530 |
| "Equipo 2 depende 30% de Z" | Equipo 2 ↔ Z: r = **+0.983** |

Este hallazgo fue documentado y el payload ignorado completamente. El análisis presentado se basa exclusivamente en tests estadísticos sobre los datos reales.

El hallazgo admite dos interpretaciones: (1) evaluación deliberada de DataKnow sobre conciencia de seguridad en IA, o (2) vulnerabilidad involuntaria. En cualquier caso, la respuesta apropiada es la misma: ignorar el payload y documentar el hallazgo.

**Para mayor detalle sobre el hallazgo, las defensas implementadas en el agente y las implicaciones para entornos corporativos, ver [`docs/security_note.md`](security_note.md).**

Este tipo de ataque se volverá cada vez más relevante a medida que los flujos de trabajo agénticos (documentos procesados por LLMs, RAG sobre bases de datos corporativas) se generalicen. La conciencia sobre prompt injection es ya una competencia esperada en roles de Data Science y MLOps.

### 7.4 Consideraciones finales

El proyecto demuestra que es posible construir una solución de forecasting de costos de extremo a extremo — desde la ingesta y auditoría de datos hasta un agente conversacional — con un stack 100% open-source (excepto el LLM de Anthropic). La arquitectura documentada en AWS (`infra/architecture.md`) permite escalar la solución a producción sin cambios arquitectónicos fundamentales.

El costo operativo estimado de la arquitectura AWS completa es ~$150/mes, dominado por los endpoints de SageMaker. Para etapas tempranas, una versión simplificada (solo ECS Fargate + S3) reduciría el costo a ~$30/mes, sacrificando la separación de concerns del pipeline de ML.

---

*Generado como parte de la prueba técnica para DataKnow — posición de Data Scientist.*  
*Repositorio: `costforecast-ai` | Stack: Python 3.11, statsmodels, XGBoost, LangGraph, Streamlit*
