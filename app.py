"""Streamlit chat UI for the lakehouse SQL agent.

    streamlit run app.py
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from lakehouse_agent.agent import AgentResult, LakehouseAgent
from lakehouse_agent.catalog import LAYERS
from lakehouse_agent.cli import DEMO_QUESTIONS
from lakehouse_agent.config import DB_PATH
from lakehouse_agent.llm import OFFLINE_BANNER

st.set_page_config(page_title="Lakehouse SQL Agent", layout="wide")


@st.cache_resource(show_spinner="Opening lakehouse ...")
def get_agent() -> LakehouseAgent:
    return LakehouseAgent()


@st.cache_data(show_spinner=False)
def table_overview(layer: str) -> list[dict]:
    return get_agent().tools.list_tables(layer)["tables"]


@st.cache_data(show_spinner=False)
def table_details(name: str) -> dict:
    return get_agent().tools.describe_table(name, sample_rows=5)


def render_result(res: AgentResult) -> None:
    st.markdown(res.answer)
    if res.rows:
        st.dataframe(pd.DataFrame(res.rows, columns=res.columns), width="stretch", hide_index=True)
    if res.sql:
        with st.expander("SQL", expanded=False):
            st.code(res.sql, language="sql")
    if res.explanation:
        st.caption(f"How: {res.explanation}")
    meta = f"{res.mode} · tool calls {res.tool_calls} · SQL attempts {res.sql_attempts} · {res.latency_s:.2f}s"
    if res.mode != "mock":
        meta += f" · {res.prompt_tokens + res.completion_tokens:,} tokens · ${res.cost_usd:.4f}"
    st.caption(meta)
    with st.expander("Agent trace", expanded=False):
        st.json(res.trace)


def sidebar_schema_browser() -> None:
    st.sidebar.header("Schema browser")
    layer = st.sidebar.radio("Layer", LAYERS, index=2, horizontal=True)
    st.sidebar.caption(get_agent().tools.catalog.layer_descriptions.get(layer, ""))
    for t in table_overview(layer):
        with st.sidebar.expander(f"{t['name']}  ·  {t['row_count']:,} rows"):
            d = table_details(t["name"])
            st.write(d["description"])
            st.markdown(f"**Primary key:** `{', '.join(d['primary_key']) or 'none (raw)'}`")
            for fk in d["foreign_keys"]:
                st.markdown(f"**FK:** `{', '.join(fk['columns'])}` → `{fk['references']}`")
            for ref in d["referenced_by"]:
                st.markdown(f"**Referenced by:** `{ref['table']}.{', '.join(ref['columns'])}`")
            up = ", ".join(d["upstream"]) or (d["source_file"] or "—")
            st.markdown(f"**Upstream:** {up}  \n**Downstream:** {', '.join(d['downstream']) or '—'}")
            st.dataframe(
                pd.DataFrame(d["columns"])[["name", "type", "description"]],
                hide_index=True, width="stretch",
            )


def main() -> None:
    st.title("Lakehouse SQL Agent")
    st.caption("Ask questions about the Olist e-commerce lakehouse (bronze / silver / gold). "
               "Answers come with the SQL, the result table and which layers, keys and joins were used.")
    if not DB_PATH.exists():
        st.error("Lakehouse not built yet. Run `make data build` and reload.")
        st.stop()

    agent = get_agent()
    if agent.llm.is_mock:
        st.info(OFFLINE_BANNER + ". Try the sample questions below.")
    sidebar_schema_browser()

    history: list[AgentResult] = st.session_state.setdefault("history", [])
    cols = st.columns(3)
    clicked = None
    for i, q in enumerate(DEMO_QUESTIONS):
        if cols[i % 3].button(q, width="stretch"):
            clicked = q

    for res in history:
        with st.chat_message("user"):
            st.write(res.question)
        with st.chat_message("assistant"):
            render_result(res)

    question = st.chat_input("e.g. Which 5 sellers have the highest revenue?") or clicked
    if question:
        with st.chat_message("user"):
            st.write(question)
        with st.chat_message("assistant"):
            with st.spinner("Thinking ..."):
                res = agent.ask(question)
            render_result(res)
        history.append(res)


main()
