"""
Agente ReAct 100% open source usando Gemma 4 via Ollama.

Estrategia dual-mode para máxima robustez:

  MODO A — Tool Calling Nativo (preferido):
    Usa langchain-ollama .bind_tools() que mapea al formato Ollama.
    Gemma 4 soporta tool calling nativo, pero requiere reasoning DESACTIVADO
    en el Modelfile (PARAMETER think false).
    Flujo: LLM → tool_calls JSON → ToolNode → LLM → respuesta

  MODO B — ReAct Prompting (fallback automático):
    Si Ollama no responde con tool_calls estructurados, el LLM escribe
    "Acción: tool_name" en texto plano y el agente parsea con regex.
    Flujo: LLM → texto "Acción/Entrada" → parse → ejecutar → "Observación" → LLM

El modo se detecta automáticamente al inicializar el agente.
Se puede forzar con: GemmaAgent(force_mode="prompting") o force_mode="native".

Web search usa DuckDuckGo (gratuito, sin API key).
"""

from __future__ import annotations

import json
import re
from typing import Any, Literal

import pandas as pd
from duckduckgo_search import DDGS
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from typing_extensions import Annotated, TypedDict

from costforecast.config import settings
from costforecast.logger import logger

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------

GEMMA_MODEL = settings.gemma_model
GEMMA_FALLBACK = settings.gemma_model_fallback
OLLAMA_URL = settings.ollama_base_url

SYSTEM_PROMPT = """\
Eres un asistente de análisis de costos de equipos de construcción con acceso a 5 herramientas.

HERRAMIENTAS DISPONIBLES:
1. get_forecast(equipo, meses) → pronóstico de costos futuros
2. get_historical_data(fecha_inicio, fecha_fin) → datos históricos reales
3. web_search_market_news(query) → noticias actuales del mercado
4. simulate_scenario(materia_prima, shock_porcentaje) → análisis what-if
5. get_shap_explanation(equipo, n_top_features) → qué variables explican el precio

REGLAS:
- Siempre responde en español.
- Para preguntas sobre costos futuros, usa get_forecast.
- Para noticias o contexto de mercado, usa web_search_market_news.
- Para impacto de shocks en materias primas, usa simulate_scenario.
- Combina múltiples herramientas cuando la pregunta lo requiera.
- Responde de forma concisa y ejecutiva (el usuario es gerencia financiera).
- No inventes datos. Si una herramienta falla, dilo claramente.
"""

# ---------------------------------------------------------------------------
# Estado del grafo LangGraph
# ---------------------------------------------------------------------------


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    mode: str


# ---------------------------------------------------------------------------
# Tools del agente Gemma
# Versión ligera con fallback demo cuando el pipeline aún no ha corrido.
# ---------------------------------------------------------------------------


@tool
def get_forecast(equipo: str, meses: int) -> str:
    """Retorna el pronóstico de costo de un equipo para los próximos N meses.

    Args:
        equipo: 'equipo1' o 'equipo2'
        meses: número de meses a proyectar (1-12)
    """
    forecast_file = settings.forecasts_dir / f"forecast_{equipo.lower()}.parquet"
    if not forecast_file.exists():
        return json.dumps(
            {
                "nota": "Modelo aún en entrenamiento — datos de demostración",
                "equipo": equipo,
                "forecast_demo": [
                    {
                        "mes": i + 1,
                        "costo_esperado": 450 + i * 5,
                        "ci_inferior": 420 + i * 4,
                        "ci_superior": 480 + i * 6,
                    }
                    for i in range(min(meses, 12))
                ],
            },
            ensure_ascii=False,
        )

    df = pd.read_parquet(forecast_file)
    subset = df.head(meses)
    return json.dumps(
        {
            "equipo": equipo,
            "forecast": [
                {
                    "fecha": str(r.name.date()),
                    "costo": round(float(r["yhat"]), 2),
                    "ci_inf": round(float(r["yhat_lower"]), 2),
                    "ci_sup": round(float(r["yhat_upper"]), 2),
                }
                for _, r in subset.iterrows()
            ],
            "mape": df.attrs.get("mape"),
        },
        ensure_ascii=False,
    )


@tool
def get_historical_data(fecha_inicio: str, fecha_fin: str) -> str:
    """Recupera estadísticas de precios históricos en un rango de fechas.

    Args:
        fecha_inicio: fecha inicio YYYY-MM-DD
        fecha_fin: fecha fin YYYY-MM-DD
    """
    processed = settings.processed_dataset_path
    if not processed.exists():
        return json.dumps({"error": "Dataset no procesado aún. Ejecuta 'make eda'."})

    df = pd.read_parquet(processed)
    subset = df.loc[fecha_inicio:fecha_fin]
    if subset.empty:
        return json.dumps({"error": f"Sin datos entre {fecha_inicio} y {fecha_fin}."})

    return subset.describe().round(2).to_json()


@tool
def web_search_market_news(query: str) -> str:
    """Busca noticias actuales del mercado de materias primas y construcción.

    Usa DuckDuckGo (gratuito, sin API key).

    Args:
        query: término de búsqueda en español o inglés
    """
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=4, region="es-es"))
        if not results:
            return json.dumps({"mensaje": f"Sin resultados para: {query}"})
        return json.dumps(
            [
                {
                    "titulo": r.get("title", ""),
                    "resumen": r.get("body", "")[:300],
                    "url": r.get("href", ""),
                }
                for r in results
            ],
            ensure_ascii=False,
        )
    except Exception as e:
        return json.dumps({"error": str(e)})


@tool
def simulate_scenario(materia_prima: str, shock_porcentaje: float) -> str:
    """Simula el impacto de un shock en una materia prima sobre los equipos.

    Usa los coeficientes de correlación de Pearson del análisis exploratorio.

    Args:
        materia_prima: 'X', 'Y' o 'Z'
        shock_porcentaje: cambio porcentual (ej: 15.0 = sube 15%, -10.0 = baja 10%)
    """
    coef = {
        "equipo1": {"X": 0.52, "Y": 0.997, "Z": 0.844},
        "equipo2": {"X": 0.53, "Y": 0.913, "Z": 0.983},
    }
    mp = materia_prima.upper()
    if mp not in ("X", "Y", "Z"):
        return json.dumps({"error": f"Materia prima inválida: {materia_prima}. Usa X, Y o Z."})

    resultado = {}
    for eq, pesos in coef.items():
        peso = pesos[mp]
        delta = round(peso * shock_porcentaje, 2)
        resultado[eq] = {
            "peso_correlacion": peso,
            "delta_estimado_pct": delta,
            "interpretacion": f"Un shock de {shock_porcentaje:+.1f}% en {mp} implica "
            f"~{delta:+.1f}% de variación esperada en {eq}",
        }

    return json.dumps(
        {
            "materia_prima": mp,
            "shock_pct": shock_porcentaje,
            "metodologia": "Estimación lineal usando correlación de Pearson del EDA",
            "impacto": resultado,
        },
        ensure_ascii=False,
    )


@tool
def get_shap_explanation(equipo: str, n_top_features: int = 5) -> str:
    """Retorna las variables más influyentes según SHAP para un equipo.

    Args:
        equipo: 'equipo1' o 'equipo2'
        n_top_features: cuántas variables mostrar (por defecto 5)
    """
    shap_file = settings.processed_data_dir / f"shap_{equipo.lower()}.json"
    if not shap_file.exists():
        _demo: dict[str, list[tuple[str, float]]] = {
            "equipo1": [
                ("Price_Y_lag1", 0.89),
                ("Price_Y", 0.85),
                ("Price_Y_roll5_mean", 0.71),
                ("Price_Z_lag1", 0.32),
                ("Price_X", 0.18),
            ],
            "equipo2": [
                ("Price_Z_lag1", 0.91),
                ("Price_Z", 0.88),
                ("Price_Z_roll10_mean", 0.74),
                ("Price_Y_lag3", 0.41),
                ("Price_X", 0.21),
            ],
        }
        features = _demo.get(equipo.lower(), [])[:n_top_features]
        return json.dumps(
            {
                "nota": "Valores de demostración basados en correlaciones del EDA",
                "equipo": equipo,
                "top_features": [
                    {"feature": f, "shap_mean_abs": v, "direccion": "↑ sube precio"}
                    for f, v in features
                ],
            },
            ensure_ascii=False,
        )

    with open(shap_file) as fh:
        data = json.load(fh)
    top = sorted(data["feature_importance"], key=lambda x: x["mean_abs_shap"], reverse=True)[
        :n_top_features
    ]
    return json.dumps({"equipo": equipo, "top_features": top}, ensure_ascii=False)


TOOLS = [
    get_forecast,
    get_historical_data,
    web_search_market_news,
    simulate_scenario,
    get_shap_explanation,
]
TOOL_MAP: dict[str, Any] = {t.name: t for t in TOOLS}

# ---------------------------------------------------------------------------
# Modo A — Native tool calling via LangGraph + langchain-ollama
# ---------------------------------------------------------------------------


def _build_native_graph(llm: ChatOllama) -> Any:
    """Grafo ReAct con tool calling nativo de Ollama."""
    llm_with_tools = llm.bind_tools(TOOLS)

    def agent_node(state: AgentState) -> dict:
        msgs = list(state["messages"])
        if not any(isinstance(m, SystemMessage) for m in msgs):
            msgs = [SystemMessage(content=SYSTEM_PROMPT)] + msgs
        response = llm_with_tools.invoke(msgs)
        return {"messages": [response], "mode": "native"}

    def should_continue(state: AgentState) -> str:
        last = state["messages"][-1]
        if hasattr(last, "tool_calls") and last.tool_calls:
            return "tools"
        return END

    graph = StateGraph(AgentState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", ToolNode(TOOLS))
    graph.set_entry_point("agent")
    graph.add_conditional_edges("agent", should_continue)
    graph.add_edge("tools", "agent")
    return graph.compile()


# ---------------------------------------------------------------------------
# Modo B — ReAct via prompting (fallback)
# ---------------------------------------------------------------------------

REACT_SYSTEM = (
    SYSTEM_PROMPT
    + """

FORMATO DE RESPUESTA cuando uses herramientas:
Pensamiento: <razona sobre qué necesitas>
Acción: <nombre_exacto_de_herramienta>
Entrada: <JSON con los argumentos>
Observación: <resultado de la herramienta>
... (repite según sea necesario)
Respuesta final: <respuesta en lenguaje natural>

Cuando NO necesites herramientas, responde directamente sin el formato de arriba.
"""
)

_ACTION_RE = re.compile(r"Acci[oó]n:\s*(\w+)", re.IGNORECASE)
_INPUT_RE = re.compile(r"Entrada:\s*(\{.*?\})", re.DOTALL | re.IGNORECASE)
_FINAL_RE = re.compile(r"Respuesta final:\s*(.+)", re.DOTALL | re.IGNORECASE)


def _parse_react_step(text: str) -> tuple[str | None, dict | None, str | None]:
    """Extrae (tool_name, args, final_answer) de la respuesta ReAct."""
    final_match = _FINAL_RE.search(text)
    if final_match:
        return None, None, final_match.group(1).strip()

    action_match = _ACTION_RE.search(text)
    if action_match:
        tool_name = action_match.group(1).strip()
        input_match = _INPUT_RE.search(text)
        try:
            args = json.loads(input_match.group(1)) if input_match else {}
        except json.JSONDecodeError:
            args = {}
        return tool_name, args, None

    return None, None, text.strip()


def _run_prompting_mode(query: str, llm: ChatOllama, max_steps: int = 6) -> str:
    """Loop ReAct via prompting para cuando el tool calling nativo no está disponible."""
    chat_msgs: list[BaseMessage] = [
        SystemMessage(content=REACT_SYSTEM),
        HumanMessage(content=query),
    ]

    for step in range(max_steps):
        response = llm.invoke(chat_msgs)
        text = response.content
        logger.debug("[ReAct step {}] {}", step + 1, text[:200])
        chat_msgs.append(response)

        tool_name, args, final = _parse_react_step(text)

        if final:
            return final

        if tool_name and tool_name in TOOL_MAP:
            try:
                result = TOOL_MAP[tool_name].invoke(args or {})
            except Exception as e:
                result = json.dumps({"error": str(e)})
            observation = f"Observación: {result}"
            logger.debug("[ReAct obs] {}", observation[:200])
            chat_msgs.append(HumanMessage(content=observation))
        else:
            return text  # respuesta directa sin herramienta

    return chat_msgs[-2].content if len(chat_msgs) >= 2 else "No se pudo generar respuesta."


# ---------------------------------------------------------------------------
# Clase principal del agente
# ---------------------------------------------------------------------------


class GemmaAgent:
    """
    Agente ReAct 100% open source con Gemma 4 via Ollama.

    Detecta automáticamente si el tool calling nativo funciona;
    si no, cae a ReAct prompting sin intervención del usuario.

    Args:
        model:      Tag del modelo en Ollama (default: settings.gemma_model).
        base_url:   URL base de Ollama (default: settings.ollama_base_url).
        force_mode: "auto" detecta, "native" fuerza nativo, "prompting" fuerza fallback.

    Usage:
        agent = GemmaAgent()
        respuesta = agent.chat("¿Cómo afecta un alza del 10% en Y al Equipo 1?")
    """

    def __init__(
        self,
        model: str = GEMMA_MODEL,
        base_url: str = OLLAMA_URL,
        force_mode: Literal["auto", "native", "prompting"] = "auto",
    ) -> None:
        self.model = model
        self.base_url = base_url
        self.force_mode = force_mode
        self._mode: str | None = None
        self._native_graph: Any = None

        self.llm = ChatOllama(
            model=model,
            base_url=base_url,
            temperature=0.1,
            num_predict=2048,
        )
        logger.info("GemmaAgent inicializado: model={} base_url={}", model, base_url)

    @property
    def mode(self) -> str:
        if self._mode is None:
            self._detect_mode()
        return self._mode  # type: ignore[return-value]

    def _detect_mode(self) -> None:
        if self.force_mode != "auto":
            self._mode = self.force_mode
            if self._mode == "native":
                self._native_graph = _build_native_graph(self.llm)
            logger.info("Modo forzado: {}", self._mode)
            return

        logger.info("Detectando capacidad de tool calling en {}…", self.model)
        try:
            test_llm = self.llm.bind_tools(TOOLS)
            probe = test_llm.invoke(
                [
                    SystemMessage(content="Responde SOLO con un tool call a get_forecast."),
                    HumanMessage(
                        content='Llama a get_forecast con equipo="equipo1" y meses=1'
                    ),
                ]
            )
            has_tool_calls = hasattr(probe, "tool_calls") and bool(probe.tool_calls)
            self._mode = "native" if has_tool_calls else "prompting"
        except Exception as e:
            logger.warning("Detección de tool calling falló: {}. Usando prompting.", e)
            self._mode = "prompting"

        logger.info("Modo detectado: {}", self._mode)

        if self._mode == "native":
            self._native_graph = _build_native_graph(self.llm)

    def chat(self, query: str) -> str:
        """Procesa una consulta y retorna la respuesta final en texto."""
        logger.info("Consulta Gemma (modo={}): {}", self.mode, query[:80])

        if self.mode == "native":
            result = self._native_graph.invoke(
                {"messages": [HumanMessage(content=query)], "mode": "native"}
            )
            last = result["messages"][-1]
            return last.content if hasattr(last, "content") else str(last)

        return _run_prompting_mode(query, self.llm)

    def stream(self, query: str):
        """
        Generador de chunks para la UI de Streamlit.

        Modo native: hace streaming de mensajes del grafo (tool calls + respuesta).
        Modo prompting: entrega la respuesta completa en un único chunk.
        """
        if self.mode == "prompting":
            yield {"messages": [AIMessage(content=self.chat(query))]}
            return

        yield from self._native_graph.stream(
            {"messages": [HumanMessage(content=query)], "mode": "native"},
            stream_mode="values",
        )

    @staticmethod
    def check_ollama(base_url: str = OLLAMA_URL) -> bool:
        """Verifica si Ollama está corriendo y accesible."""
        try:
            import urllib.request

            with urllib.request.urlopen(f"{base_url}/api/tags", timeout=3) as r:
                return r.status == 200
        except Exception:
            return False

    @staticmethod
    def list_models(base_url: str = OLLAMA_URL) -> list[str]:
        """Lista los modelos disponibles en Ollama."""
        try:
            import json as _json
            import urllib.request

            with urllib.request.urlopen(f"{base_url}/api/tags", timeout=3) as r:
                data = _json.loads(r.read())
            return [m["name"] for m in data.get("models", [])]
        except Exception:
            return []
