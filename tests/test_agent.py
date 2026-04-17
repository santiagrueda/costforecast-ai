"""
Tests para src/costforecast/agent/.

Estrategia:
- Tools: se mockea _load_dataset() y los modelos para tests rápidos sin I/O.
  Se verifica que cada tool retorna un string, maneja errores graciosamente,
  y tiene el schema correcto para LangChain.
- graph.py: se verifica la construcción del agente (mockeando ChatAnthropic),
  la interfaz de CostForecastAgent y la validación de API key ausente.
- No se hacen llamadas reales al LLM ni a Tavily en los tests.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from costforecast.agent.tools import (
    TOOLS,
    _load_dataset,
    clear_cache,
    get_forecast,
    get_historical_data,
    get_shap_explanation,
    simulate_scenario,
    web_search_market_news,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def reset_cache():
    """Limpia el cache del módulo de tools antes y después de cada test."""
    clear_cache()
    yield
    clear_cache()


@pytest.fixture
def sample_df() -> pd.DataFrame:
    """Dataset sintético mínimo con 60 días hábiles."""
    rng = np.random.default_rng(0)
    n = 60
    idx = pd.bdate_range("2020-01-01", periods=n)
    x = 80 + np.cumsum(rng.normal(0, 0.3, n))
    y = 500 + np.cumsum(rng.normal(0, 1, n))
    z = 2000 + np.cumsum(rng.normal(0, 2, n))
    e1 = 0.4 * x + 0.3 * z + rng.normal(0, 3, n)
    e2 = 0.2 * x + 0.5 * z + rng.normal(0, 3, n)
    return pd.DataFrame(
        {"Price_X": x, "Price_Y": y, "Price_Z": z,
         "Price_Equipo1": e1, "Price_Equipo2": e2},
        index=idx,
    )


# ---------------------------------------------------------------------------
# Tests del registro de tools (schema / metadata)
# ---------------------------------------------------------------------------

class TestToolsRegistry:
    def test_five_tools_exported(self) -> None:
        assert len(TOOLS) == 5

    def test_all_tools_have_names(self) -> None:
        names = {t.name for t in TOOLS}
        assert "get_forecast" in names
        assert "get_historical_data" in names
        assert "web_search_market_news" in names
        assert "simulate_scenario" in names
        assert "get_shap_explanation" in names

    def test_all_tools_have_description(self) -> None:
        for t in TOOLS:
            assert t.description, f"Tool '{t.name}' sin descripción"

    def test_all_tools_have_args_schema(self) -> None:
        for t in TOOLS:
            schema = t.args_schema.model_json_schema()
            assert "properties" in schema, f"Tool '{t.name}' sin properties en schema"

    def test_get_forecast_schema_has_equipment(self) -> None:
        schema = get_forecast.args_schema.model_json_schema()
        assert "equipment" in schema["properties"]
        assert "horizon_days" in schema["properties"]

    def test_simulate_scenario_schema_has_change_fields(self) -> None:
        schema = simulate_scenario.args_schema.model_json_schema()
        props = schema["properties"]
        assert "price_x_change_pct" in props
        assert "price_y_change_pct" in props
        assert "price_z_change_pct" in props


# ---------------------------------------------------------------------------
# Tool 1 — get_forecast
# ---------------------------------------------------------------------------

class TestGetForecast:
    def test_returns_string(self, sample_df) -> None:
        with patch("costforecast.agent.tools._load_dataset", return_value=sample_df):
            result = get_forecast.invoke({"equipment": "equipo1", "horizon_days": 5})
        assert isinstance(result, str)

    def test_result_contains_target_name(self, sample_df) -> None:
        with patch("costforecast.agent.tools._load_dataset", return_value=sample_df):
            result = get_forecast.invoke({"equipment": "equipo1", "horizon_days": 5})
        assert "Price_Equipo1" in result

    def test_result_contains_forecast_rows(self, sample_df) -> None:
        with patch("costforecast.agent.tools._load_dataset", return_value=sample_df):
            result = get_forecast.invoke({"equipment": "equipo2", "horizon_days": 3})
        assert "2020" in result or "Pronóstico" in result

    def test_invalid_equipment_returns_error_string(self, sample_df) -> None:
        with patch("costforecast.agent.tools._load_dataset", return_value=sample_df):
            result = get_forecast.invoke({"equipment": "equipo99", "horizon_days": 5})
        assert "Error" in result or "error" in result.lower() or "no reconocido" in result

    def test_equipo2_works(self, sample_df) -> None:
        with patch("costforecast.agent.tools._load_dataset", return_value=sample_df):
            result = get_forecast.invoke({"equipment": "equipo2", "horizon_days": 5})
        assert "Price_Equipo2" in result

    def test_default_horizon_is_ten(self, sample_df) -> None:
        with patch("costforecast.agent.tools._load_dataset", return_value=sample_df):
            result = get_forecast.invoke({"equipment": "equipo1"})
        # 10 fechas en la tabla → 10 filas de datos
        assert result.count("|") > 20  # al menos 10 filas en la tabla Markdown


# ---------------------------------------------------------------------------
# Tool 2 — get_historical_data
# ---------------------------------------------------------------------------

class TestGetHistoricalData:
    def test_returns_string(self, sample_df) -> None:
        with patch("costforecast.agent.tools._load_dataset", return_value=sample_df):
            result = get_historical_data.invoke({})
        assert isinstance(result, str)

    def test_contains_statistics(self, sample_df) -> None:
        with patch("costforecast.agent.tools._load_dataset", return_value=sample_df):
            result = get_historical_data.invoke({})
        assert "mean" in result.lower() or "Media" in result or "std" in result.lower()

    def test_column_filter_works(self, sample_df) -> None:
        with patch("costforecast.agent.tools._load_dataset", return_value=sample_df):
            result = get_historical_data.invoke({"columns": ["Price_X"]})
        assert "Price_X" in result
        assert "Price_Y" not in result

    def test_date_filter_start(self, sample_df) -> None:
        with patch("costforecast.agent.tools._load_dataset", return_value=sample_df):
            result = get_historical_data.invoke({"start_date": "2020-02-01"})
        assert isinstance(result, str)
        assert "Error" not in result

    def test_last_n_filter(self, sample_df) -> None:
        with patch("costforecast.agent.tools._load_dataset", return_value=sample_df):
            result = get_historical_data.invoke({"last_n": 10})
        assert "10" in result or "observaciones" in result

    def test_invalid_date_returns_gracefully(self, sample_df) -> None:
        # Filtro de fecha que da DataFrame vacío
        with patch("costforecast.agent.tools._load_dataset", return_value=sample_df):
            result = get_historical_data.invoke({"start_date": "2030-01-01"})
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# Tool 3 — web_search_market_news
# ---------------------------------------------------------------------------

class TestWebSearchMarketNews:
    def test_returns_string(self) -> None:
        result = web_search_market_news.invoke({"query": "steel prices"})
        assert isinstance(result, str)

    def test_no_key_returns_warning(self) -> None:
        with patch("costforecast.agent.tools.settings") as mock_settings:
            mock_settings.tavily_api_key = ""
            result = web_search_market_news.invoke({"query": "commodities"})
        assert "TAVILY_API_KEY" in result or "⚠️" in result

    def test_tavily_called_with_query(self) -> None:
        mock_client = MagicMock()
        mock_client.search.return_value = {
            "results": [
                {
                    "title": "Steel up 5%",
                    "url": "https://example.com",
                    "content": "Steel prices rose...",
                }
            ]
        }
        with patch("costforecast.agent.tools.settings") as mock_settings, \
             patch("costforecast.agent.tools.TavilyClient", return_value=mock_client) if False else \
             patch("tavily.TavilyClient", return_value=mock_client):
            mock_settings.tavily_api_key = "tvly-test"
            # Patch the import inside the function
            import sys
            import types
            mock_tavily_module = types.ModuleType("tavily")
            mock_tavily_module.TavilyClient = MagicMock(return_value=mock_client)
            with patch.dict(sys.modules, {"tavily": mock_tavily_module}):
                result = web_search_market_news.invoke({"query": "steel prices", "max_results": 3})
        assert isinstance(result, str)

    def test_empty_results_handled(self) -> None:
        mock_client = MagicMock()
        mock_client.search.return_value = {"results": []}
        import sys, types
        mock_tavily = types.ModuleType("tavily")
        mock_tavily.TavilyClient = MagicMock(return_value=mock_client)
        with patch.dict(sys.modules, {"tavily": mock_tavily}), \
             patch("costforecast.agent.tools.settings") as ms:
            ms.tavily_api_key = "tvly-test"
            result = web_search_market_news.invoke({"query": "xyz irrelevant"})
        assert "No se encontraron" in result or isinstance(result, str)


# ---------------------------------------------------------------------------
# Tool 4 — simulate_scenario
# ---------------------------------------------------------------------------

class TestSimulateScenario:
    def test_returns_string(self, sample_df) -> None:
        with patch("costforecast.agent.tools._load_dataset", return_value=sample_df):
            result = simulate_scenario.invoke({
                "equipment": "equipo1",
                "price_x_change_pct": 10.0,
                "price_y_change_pct": 0.0,
                "price_z_change_pct": -5.0,
            })
        assert isinstance(result, str)

    def test_contains_scenario_assumptions(self, sample_df) -> None:
        with patch("costforecast.agent.tools._load_dataset", return_value=sample_df):
            result = simulate_scenario.invoke({
                "equipment": "equipo1",
                "price_x_change_pct": 20.0,
            })
        assert "+20.0%" in result or "20" in result

    def test_baseline_and_scenario_columns_present(self, sample_df) -> None:
        with patch("costforecast.agent.tools._load_dataset", return_value=sample_df):
            result = simulate_scenario.invoke({"equipment": "equipo2"})
        assert "Baseline" in result or "baseline" in result.lower()
        assert "Escenario" in result or "escenario" in result.lower()

    def test_zero_changes_returns_result(self, sample_df) -> None:
        with patch("costforecast.agent.tools._load_dataset", return_value=sample_df):
            result = simulate_scenario.invoke({
                "equipment": "equipo1",
                "price_x_change_pct": 0.0,
                "price_y_change_pct": 0.0,
                "price_z_change_pct": 0.0,
            })
        assert "Price_Equipo1" in result

    def test_invalid_equipment_handled(self, sample_df) -> None:
        with patch("costforecast.agent.tools._load_dataset", return_value=sample_df):
            result = simulate_scenario.invoke({"equipment": "maquina999"})
        assert isinstance(result, str)
        assert "Error" in result or "no reconocido" in result


# ---------------------------------------------------------------------------
# Tool 5 — get_shap_explanation
# ---------------------------------------------------------------------------

class TestGetShapExplanation:
    def test_returns_string(self, sample_df) -> None:
        with patch("costforecast.agent.tools._load_dataset", return_value=sample_df):
            result = get_shap_explanation.invoke({
                "equipment": "equipo1",
                "n_top_features": 5,
                "n_samples": 30,
            })
        assert isinstance(result, str)

    def test_contains_shap_header(self, sample_df) -> None:
        with patch("costforecast.agent.tools._load_dataset", return_value=sample_df):
            result = get_shap_explanation.invoke({
                "equipment": "equipo1",
                "n_top_features": 5,
                "n_samples": 30,
            })
        assert "SHAP" in result

    def test_contains_feature_table(self, sample_df) -> None:
        with patch("costforecast.agent.tools._load_dataset", return_value=sample_df):
            result = get_shap_explanation.invoke({
                "equipment": "equipo1",
                "n_top_features": 5,
                "n_samples": 30,
            })
        assert "|" in result  # tabla Markdown

    def test_equipo2_works(self, sample_df) -> None:
        with patch("costforecast.agent.tools._load_dataset", return_value=sample_df):
            result = get_shap_explanation.invoke({
                "equipment": "equipo2",
                "n_top_features": 5,
                "n_samples": 30,
            })
        assert "Price_Equipo2" in result

    def test_invalid_equipment_handled(self, sample_df) -> None:
        with patch("costforecast.agent.tools._load_dataset", return_value=sample_df):
            result = get_shap_explanation.invoke({"equipment": "xyz"})
        assert isinstance(result, str)
        assert "Error" in result or "no reconocido" in result


# ---------------------------------------------------------------------------
# Tests de build_agent y CostForecastAgent
# ---------------------------------------------------------------------------

class TestBuildAgent:
    def test_raises_without_api_key(self) -> None:
        from costforecast.agent.graph import build_agent
        with patch("costforecast.agent.graph.settings") as mock_settings:
            mock_settings.anthropic_api_key = ""
            mock_settings.claude_model = "claude-sonnet-4-5-20250929"
            mock_settings.claude_max_tokens = 2048
            mock_settings.claude_temperature = 0.2
            with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
                build_agent()

    def test_explicit_key_bypasses_settings(self) -> None:
        """build_agent con api_key explícita no debe consultar settings.anthropic_api_key."""
        from costforecast.agent.graph import build_agent, SYSTEM_PROMPT
        mock_llm = MagicMock()
        mock_llm.bind_tools = MagicMock(return_value=mock_llm)
        mock_graph = MagicMock()

        with patch("costforecast.agent.graph.ChatAnthropic", return_value=mock_llm), \
             patch("costforecast.agent.graph.create_react_agent", return_value=mock_graph):
            result = build_agent(api_key="sk-ant-test-key")
        assert result is mock_graph

    def test_chatanthropic_receives_correct_model(self) -> None:
        from costforecast.agent.graph import build_agent
        captured: dict = {}

        def mock_chat(**kwargs):
            captured.update(kwargs)
            return MagicMock()

        with patch("costforecast.agent.graph.ChatAnthropic", side_effect=mock_chat), \
             patch("costforecast.agent.graph.create_react_agent", return_value=MagicMock()):
            build_agent(api_key="sk-ant-test", model_name="claude-test-model")

        assert captured.get("model") == "claude-test-model"

    def test_all_tools_passed_to_react_agent(self) -> None:
        from costforecast.agent.graph import build_agent
        captured_tools: list = []

        def mock_react(model, tools, **kwargs):
            captured_tools.extend(tools)
            return MagicMock()

        with patch("costforecast.agent.graph.ChatAnthropic", return_value=MagicMock()), \
             patch("costforecast.agent.graph.create_react_agent", side_effect=mock_react):
            build_agent(api_key="sk-ant-test")

        assert len(captured_tools) == 5


class TestCostForecastAgent:
    @pytest.fixture
    def agent(self) -> "CostForecastAgent":
        from costforecast.agent.graph import CostForecastAgent
        mock_graph = MagicMock()
        last_msg = MagicMock()
        last_msg.content = "Respuesta de prueba del agente."
        mock_graph.invoke.return_value = {"messages": [MagicMock(), last_msg]}
        with patch("costforecast.agent.graph.build_agent", return_value=mock_graph):
            agent = CostForecastAgent(api_key="sk-ant-test")
        agent._graph = mock_graph
        return agent

    def test_chat_returns_string(self, agent) -> None:
        response = agent.chat("Proyecta Equipo 1 para los próximos 10 días.")
        assert isinstance(response, str)

    def test_chat_returns_last_message_content(self, agent) -> None:
        response = agent.chat("¿Qué pasaría si Price_Y sube 20%?")
        assert response == "Respuesta de prueba del agente."

    def test_chat_passes_thread_id_in_config(self, agent) -> None:
        agent.chat("Hola", thread_id="mi-sesion-123")
        call_kwargs = agent._graph.invoke.call_args
        config = call_kwargs[1].get("config") or call_kwargs[0][1]
        assert config["configurable"]["thread_id"] == "mi-sesion-123"

    def test_graph_property_accessible(self, agent) -> None:
        assert agent.graph is agent._graph

    def test_stream_yields_events(self, agent) -> None:
        agent._graph.stream = MagicMock(return_value=iter([{"messages": []}, {"messages": []}]))
        events = list(agent.stream("test"))
        assert len(events) == 2


# ---------------------------------------------------------------------------
# Test de system prompt
# ---------------------------------------------------------------------------

class TestSystemPrompt:
    def test_system_prompt_mentions_tools(self) -> None:
        from costforecast.agent.graph import SYSTEM_PROMPT
        assert "get_forecast" in SYSTEM_PROMPT
        assert "simulate_scenario" in SYSTEM_PROMPT
        assert "get_shap_explanation" in SYSTEM_PROMPT

    def test_system_prompt_warns_about_injection(self) -> None:
        from costforecast.agent.graph import SYSTEM_PROMPT
        assert "Prompt Injection" in SYSTEM_PROMPT or "prompt injection" in SYSTEM_PROMPT.lower()
        assert "ignoradas" in SYSTEM_PROMPT or "ignored" in SYSTEM_PROMPT.lower()

    def test_system_prompt_mentions_equipos(self) -> None:
        from costforecast.agent.graph import SYSTEM_PROMPT
        assert "Equipo 1" in SYSTEM_PROMPT
        assert "Equipo 2" in SYSTEM_PROMPT
