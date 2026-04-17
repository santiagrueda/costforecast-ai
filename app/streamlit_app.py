"""
CostForecast AI — Streamlit App
================================
Three tabs:
  1. Pronóstico   — forecast chart with confidence intervals (SARIMAX)
  2. Explorar datos — interactive Plotly dashboard
  3. Agente de IA   — chat with the LangGraph ReAct agent
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make sure `src/` is importable when running from project root
_src = Path(__file__).resolve().parent.parent / "src"
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ---------------------------------------------------------------------------
# Page config  (must be first Streamlit call)
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="CostForecast AI",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# Domain constants
# ---------------------------------------------------------------------------

EXOG_COLS = ["Price_X", "Price_Y", "Price_Z"]
EQUIP_MAP = {
    "Equipo 1": "Price_Equipo1",
    "Equipo 2": "Price_Equipo2",
}
COLORS: dict[str, str] = {
    "Price_Equipo1": "#2563EB",
    "Price_Equipo2": "#16A34A",
    "Price_X": "#F59E0B",
    "Price_Y": "#EF4444",
    "Price_Z": "#8B5CF6",
    "forecast": "#F97316",
    "ci_fill": "rgba(249,115,22,0.15)",
}

# ---------------------------------------------------------------------------
# Cached data & model loaders
# ---------------------------------------------------------------------------


@st.cache_data(show_spinner="Cargando dataset consolidado…")
def load_dataset() -> pd.DataFrame:
    from costforecast.config import settings
    from costforecast.data.consolidator import build_consolidated_dataset

    raw = settings.raw_data_dir
    return build_consolidated_dataset(
        historico_path=raw / "historico_equipos.csv",
        x_path=raw / "X.csv",
        y_path=raw / "Y.csv",
        z_path=raw / "Z.csv",
    )


@st.cache_resource(show_spinner="Entrenando modelo SARIMAX(1,1,1)…")
def get_sarimax(target_col: str):
    """Fit and cache a SARIMAXModel for the given target column."""
    from costforecast.models.sarimax_model import SARIMAXModel

    df = load_dataset()
    model = SARIMAXModel(order=(1, 1, 1))
    model.fit(df[EXOG_COLS], df[target_col])
    return model


# ---------------------------------------------------------------------------
# App header
# ---------------------------------------------------------------------------

st.markdown("## 📈 CostForecast AI")
st.caption(
    "Proyección de costos de equipos de construcción | "
    "Prueba técnica DataKnow · Santiago Rueda"
)

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

tab_forecast, tab_eda, tab_agent = st.tabs(
    ["📊 Pronóstico", "🔍 Explorar datos", "🤖 Agente de IA"]
)


# ===========================================================================
# TAB 1 — PRONÓSTICO
# ===========================================================================

with tab_forecast:
    st.subheader("Proyección de precios con intervalos de confianza")
    st.markdown(
        "Modelo **SARIMAX(1,1,1)** con materias primas X, Y, Z como variables exógenas. "
        "El pronóstico asume persistencia del último valor conocido de las materias primas."
    )

    # ── Controls ────────────────────────────────────────────────────────────
    c1, c2, c3 = st.columns([1, 1, 2])
    with c1:
        equip_label = st.selectbox("Equipo", list(EQUIP_MAP), key="fc_equip")
        target_col = EQUIP_MAP[equip_label]
    with c2:
        horizon = st.slider("Horizonte (días hábiles)", 1, 60, 20, key="fc_horizon")
    with c3:
        history_days = st.slider("Historial a mostrar (días)", 60, 500, 180, key="fc_history")

    alpha = st.select_slider(
        "Nivel de confianza",
        options=[0.01, 0.05, 0.10, 0.20],
        value=0.05,
        format_func=lambda x: f"{int((1 - x) * 100)}%",
        key="fc_alpha",
    )

    run_btn = st.button("⚡ Generar pronóstico", type="primary", key="fc_run")

    # ── Compute forecast ────────────────────────────────────────────────────
    fc_key = f"fc_{target_col}_{horizon}_{alpha}"
    if run_btn or fc_key not in st.session_state:
        with st.spinner("Ajustando modelo y calculando pronóstico…"):
            df = load_dataset()
            model = get_sarimax(target_col)

            last_exog = df[EXOG_COLS].iloc[-1]
            last_date = df.index[-1]
            future_dates = pd.bdate_range(
                start=last_date + pd.offsets.BDay(1), periods=horizon
            )
            X_future = pd.DataFrame(
                np.tile(last_exog.values, (horizon, 1)),
                columns=EXOG_COLS,
                index=future_dates,
            )

            # Point forecast through the model wrapper
            preds = model.predict(X_future)

            # Confidence intervals directly from statsmodels result object
            exog_arr = (
                X_future[model._exog_cols].values if model._exog_cols else None
            )
            fc_obj = model._result.get_forecast(steps=horizon, exog=exog_arr)
            ci = fc_obj.conf_int(alpha=alpha)

            st.session_state[fc_key] = {
                "hist": df[target_col].iloc[-history_days:],
                "future_dates": future_dates,
                "preds": preds,
                "lower": ci.iloc[:, 0].values,
                "upper": ci.iloc[:, 1].values,
                "last_date": last_date,
            }

    # ── Render chart ────────────────────────────────────────────────────────
    if fc_key in st.session_state:
        res = st.session_state[fc_key]
        hist: pd.Series = res["hist"]
        future_dates = res["future_dates"]
        preds: pd.Series = res["preds"]
        lower: np.ndarray = res["lower"]
        upper: np.ndarray = res["upper"]
        last_date = res["last_date"]

        fig = go.Figure()

        # Historical series
        fig.add_trace(
            go.Scatter(
                x=hist.index,
                y=hist.values,
                mode="lines",
                name=f"{equip_label} (histórico)",
                line=dict(color=COLORS[target_col], width=2),
            )
        )

        # Confidence interval band (shaded)
        fig.add_trace(
            go.Scatter(
                x=list(future_dates) + list(future_dates[::-1]),
                y=list(upper) + list(lower[::-1]),
                fill="toself",
                fillcolor=COLORS["ci_fill"],
                line=dict(color="rgba(0,0,0,0)"),
                name=f"IC {int((1 - alpha) * 100)}%",
                showlegend=True,
            )
        )

        # Forecast line
        fig.add_trace(
            go.Scatter(
                x=future_dates,
                y=preds.values,
                mode="lines+markers",
                name="Pronóstico SARIMAX",
                line=dict(color=COLORS["forecast"], width=2.5, dash="dash"),
                marker=dict(size=5),
            )
        )

        # Cutoff vertical line
        fig.add_vline(
            x=str(last_date.date()),
            line_width=1,
            line_dash="dot",
            line_color="gray",
            annotation_text="Fin histórico",
            annotation_position="top left",
        )

        fig.update_layout(
            template="plotly_white",
            height=430,
            hovermode="x unified",
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
            ),
            xaxis_title="Fecha",
            yaxis_title="Precio (USD)",
            margin=dict(l=0, r=0, t=45, b=0),
        )
        st.plotly_chart(fig, use_container_width=True)

        # ── Metric cards ────────────────────────────────────────────────────
        last_actual = float(hist.iloc[-1])
        mean_fc = float(preds.mean())
        delta_pct = (mean_fc - last_actual) / last_actual * 100
        prev_delta = float(hist.pct_change().iloc[-1] * 100)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric(
            "Último precio real",
            f"{last_actual:,.2f}",
            f"{prev_delta:+.2f}% vs día ant.",
        )
        m2.metric(
            "Media pronóstico",
            f"{mean_fc:,.2f}",
            f"{delta_pct:+.2f}% vs actual",
        )
        m3.metric("Mínimo esperado", f"{preds.min():,.2f}")
        m4.metric("Máximo esperado", f"{preds.max():,.2f}")

        # ── Data table (collapsed) ───────────────────────────────────────────
        with st.expander("Ver tabla completa del pronóstico"):
            tbl = pd.DataFrame(
                {
                    "Fecha": future_dates.date,
                    "Pronóstico": preds.round(2).values,
                    f"IC inf {int((1-alpha)*100)}%": lower.round(2),
                    f"IC sup {int((1-alpha)*100)}%": upper.round(2),
                    "Amplitud IC": (upper - lower).round(2),
                }
            )
            st.dataframe(tbl, use_container_width=True, hide_index=True)

        # ── Exog context ─────────────────────────────────────────────────────
        with st.expander("Valores de materias primas usados en el pronóstico"):
            df_full = load_dataset()
            last_exog_vals = df_full[EXOG_COLS].iloc[-1]
            st.dataframe(
                last_exog_vals.rename("Último valor (persistencia)").to_frame().T,
                use_container_width=True,
            )


# ===========================================================================
# TAB 2 — EXPLORAR DATOS
# ===========================================================================

with tab_eda:
    st.subheader("Dashboard exploratorio interactivo")

    df_full = load_dataset()
    all_cols = list(df_full.columns)

    # ── Controls ─────────────────────────────────────────────────────────────
    e1, e2 = st.columns([2, 1])
    with e1:
        sel_cols = st.multiselect(
            "Series a visualizar",
            all_cols,
            default=all_cols,
            key="eda_cols",
        )
    with e2:
        date_vals = st.date_input(
            "Rango de fechas",
            value=(df_full.index[0].date(), df_full.index[-1].date()),
            min_value=df_full.index[0].date(),
            max_value=df_full.index[-1].date(),
            key="eda_dates",
        )

    # Apply filter
    if isinstance(date_vals, (list, tuple)) and len(date_vals) == 2:
        start_d, end_d = date_vals
    else:
        start_d, end_d = df_full.index[0].date(), df_full.index[-1].date()

    df_view = df_full.loc[str(start_d) : str(end_d), sel_cols] if sel_cols else pd.DataFrame()

    if df_view.empty or not sel_cols:
        st.info("Selecciona al menos una serie para visualizar.")
        st.stop()

    # ── Row 1: Time series ────────────────────────────────────────────────────
    st.markdown("#### Series de tiempo")
    fig_ts = go.Figure()
    for col in sel_cols:
        fig_ts.add_trace(
            go.Scatter(
                x=df_view.index,
                y=df_view[col],
                mode="lines",
                name=col,
                line=dict(color=COLORS.get(col, "#6B7280"), width=1.5),
            )
        )
    fig_ts.update_layout(
        template="plotly_white",
        height=360,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis_title="Fecha",
        yaxis_title="Precio",
        margin=dict(l=0, r=0, t=40, b=0),
    )
    st.plotly_chart(fig_ts, use_container_width=True)

    # ── Row 2: Normalized + Correlation ───────────────────────────────────────
    r2c1, r2c2 = st.columns([3, 2])

    with r2c1:
        st.markdown("#### Series normalizadas (índice base 100)")
        df_norm = df_view / df_view.iloc[0] * 100
        fig_norm = go.Figure()
        for col in sel_cols:
            fig_norm.add_trace(
                go.Scatter(
                    x=df_norm.index,
                    y=df_norm[col],
                    mode="lines",
                    name=col,
                    line=dict(color=COLORS.get(col, "#6B7280"), width=1.5),
                )
            )
        fig_norm.add_hline(y=100, line_dash="dot", line_color="gray", opacity=0.5)
        fig_norm.update_layout(
            template="plotly_white",
            height=310,
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=0, r=0, t=35, b=0),
            yaxis_title="Índice (base 100)",
        )
        st.plotly_chart(fig_norm, use_container_width=True)

    with r2c2:
        st.markdown("#### Correlación de Pearson")
        if len(sel_cols) >= 2:
            corr = df_view.corr()
            fig_corr = px.imshow(
                corr,
                color_continuous_scale="RdBu_r",
                zmin=-1,
                zmax=1,
                text_auto=".2f",
                aspect="auto",
            )
            fig_corr.update_layout(
                template="plotly_white",
                height=310,
                coloraxis_showscale=False,
                margin=dict(l=0, r=0, t=35, b=0),
            )
            st.plotly_chart(fig_corr, use_container_width=True)
        else:
            st.info("Selecciona 2 o más series para ver la correlación.")

    # ── Row 3: Rolling mean + Distribution ────────────────────────────────────
    r3c1, r3c2 = st.columns([3, 2])

    with r3c1:
        roll_window = st.select_slider(
            "Ventana de la media móvil (días)",
            options=[5, 10, 20, 60, 120],
            value=20,
            key="eda_window",
        )
        st.markdown(f"#### Media móvil ({roll_window} días)")
        fig_roll = go.Figure()
        for col in sel_cols:
            roll = df_view[col].rolling(roll_window, min_periods=roll_window).mean()
            fig_roll.add_trace(
                go.Scatter(
                    x=roll.index,
                    y=roll.values,
                    mode="lines",
                    name=col,
                    line=dict(color=COLORS.get(col, "#6B7280"), width=2),
                )
            )
        fig_roll.update_layout(
            template="plotly_white",
            height=280,
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=0, r=0, t=35, b=0),
        )
        st.plotly_chart(fig_roll, use_container_width=True)

    with r3c2:
        dist_col = st.selectbox(
            "Distribución de serie:",
            sel_cols,
            key="eda_dist_col",
        )
        st.markdown(f"#### Distribución — {dist_col}")
        fig_dist = px.histogram(
            df_view,
            x=dist_col,
            nbins=50,
            color_discrete_sequence=[COLORS.get(dist_col, "#2563EB")],
            marginal="box",
        )
        fig_dist.update_layout(
            template="plotly_white",
            height=280,
            showlegend=False,
            xaxis_title=dist_col,
            yaxis_title="Frecuencia",
            margin=dict(l=0, r=0, t=35, b=0),
        )
        st.plotly_chart(fig_dist, use_container_width=True)

    # ── Row 4: Descriptive statistics ─────────────────────────────────────────
    with st.expander("Estadísticas descriptivas completas"):
        desc = df_view.describe().T.round(2).rename(columns={"50%": "mediana"})
        st.dataframe(desc, use_container_width=True)

    # ── Row 5: Pairwise scatter (optional) ────────────────────────────────────
    if len(sel_cols) >= 2:
        with st.expander("Scatterplot entre dos series"):
            sc1, sc2 = st.columns(2)
            x_col = sc1.selectbox("Eje X", sel_cols, index=0, key="scatter_x")
            y_col = sc2.selectbox(
                "Eje Y",
                sel_cols,
                index=min(1, len(sel_cols) - 1),
                key="scatter_y",
            )
            if x_col != y_col:
                fig_sc = px.scatter(
                    df_view,
                    x=x_col,
                    y=y_col,
                    color_discrete_sequence=[COLORS.get(y_col, "#2563EB")],
                    trendline="ols",
                    trendline_color_override="#F97316",
                    opacity=0.4,
                )
                fig_sc.update_layout(
                    template="plotly_white",
                    height=350,
                    margin=dict(l=0, r=0, t=30, b=0),
                )
                st.plotly_chart(fig_sc, use_container_width=True)
            else:
                st.info("Selecciona dos series distintas.")


# ===========================================================================
# TAB 3 — AGENTE DE IA
# ===========================================================================

with tab_agent:
    st.subheader("Chat con CostForecast AI Agent")
    st.caption(
        "Agente **ReAct** (LangGraph + Claude) con 5 tools: "
        "`get_forecast`, `get_historical_data`, `web_search_market_news`, "
        "`simulate_scenario`, `get_shap_explanation`."
    )

    # ── API key setup ─────────────────────────────────────────────────────────
    try:
        from costforecast.config import settings as _cfg

        configured_key = _cfg.anthropic_api_key
    except Exception:
        configured_key = ""

    if not configured_key:
        api_key = st.text_input(
            "🔑 ANTHROPIC_API_KEY",
            type="password",
            placeholder="sk-ant-…",
            help="Ingresa tu clave de Anthropic. No se persiste entre sesiones.",
            key="agent_api_key",
        )
    else:
        api_key = configured_key
        st.success("API key cargada desde `.env`", icon="✅")

    st.divider()

    # ── Quick prompts ─────────────────────────────────────────────────────────
    st.markdown("**Preguntas de ejemplo:**")
    quick_prompts = [
        "¿Cuál es la proyección del Equipo 1 para los próximos 20 días?",
        "¿Qué materias primas impactan más en el precio del Equipo 2?",
        "Simula un aumento del 15% en Price_Y para el Equipo 1 en 10 días.",
        "Muéstrame las estadísticas del último año de Price_Z.",
        "Explica con SHAP qué features son más importantes para el Equipo 2.",
    ]
    qcols = st.columns(len(quick_prompts))
    for i, (qcol, qp) in enumerate(zip(qcols, quick_prompts)):
        if qcol.button(
            qp[:40] + "…" if len(qp) > 40 else qp,
            key=f"quick_{i}",
            use_container_width=True,
            help=qp,
        ):
            st.session_state["_pending_msg"] = qp

    st.divider()

    # ── Initialize session state ──────────────────────────────────────────────
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    # ── Render conversation history ───────────────────────────────────────────
    for msg in st.session_state["chat_history"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # ── Resolve user input (typed or quick-prompt) ────────────────────────────
    typed_input: str | None = st.chat_input(
        "Escribe tu pregunta sobre costos de equipos…",
        key="agent_chat_input",
    )
    user_input: str | None = st.session_state.pop("_pending_msg", None) or typed_input

    # ── Process message ───────────────────────────────────────────────────────
    if user_input:
        if not api_key:
            st.warning(
                "Ingresa tu **ANTHROPIC_API_KEY** arriba para usar el agente.",
                icon="⚠️",
            )
        else:
            # Display user bubble
            with st.chat_message("user"):
                st.markdown(user_input)
            st.session_state["chat_history"].append(
                {"role": "user", "content": user_input}
            )

            # Build/reuse agent (cached in session state)
            if "agent_instance" not in st.session_state:
                try:
                    try:
                        from langgraph.checkpoint.memory import MemorySaver
                    except ImportError:
                        from langgraph.checkpoint import MemorySaver  # type: ignore[no-redef]

                    from costforecast.agent.graph import CostForecastAgent

                    st.session_state["agent_instance"] = CostForecastAgent(
                        api_key=api_key,
                        checkpointer=MemorySaver(),
                    )
                except Exception as exc:
                    st.error(f"No se pudo inicializar el agente: {exc}")
                    st.stop()

            agent: object = st.session_state["agent_instance"]

            # Get response with streaming status
            with st.chat_message("assistant"):
                thinking_placeholder = st.empty()
                tool_status = st.status("Razonando…", expanded=False)

                try:
                    # Stream intermediate steps to status widget
                    response_text = ""
                    for chunk in agent.stream(user_input, thread_id="streamlit"):  # type: ignore[attr-defined]
                        msgs = chunk.get("messages", [])
                        if msgs:
                            last = msgs[-1]
                            # Tool call messages show in the status panel
                            if hasattr(last, "tool_calls") and last.tool_calls:
                                for tc in last.tool_calls:
                                    tool_status.write(
                                        f"🔧 `{tc['name']}` ← {str(tc.get('args', {}))[:120]}"
                                    )
                            # Final AI message
                            elif hasattr(last, "content") and isinstance(last.content, str):
                                response_text = last.content

                    tool_status.update(label="Herramientas ejecutadas", state="complete")
                    thinking_placeholder.markdown(response_text)

                except Exception as exc:
                    response_text = f"Error al invocar el agente: {exc}"
                    tool_status.update(label="Error", state="error")
                    thinking_placeholder.error(response_text)

            st.session_state["chat_history"].append(
                {"role": "assistant", "content": response_text}
            )

    # ── Clear conversation ────────────────────────────────────────────────────
    if st.session_state["chat_history"]:
        st.divider()
        if st.button("🗑️ Limpiar conversación", key="clear_chat"):
            st.session_state["chat_history"] = []
            st.session_state.pop("agent_instance", None)
            st.rerun()
