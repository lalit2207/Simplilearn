"""Admin view: configuration, index status, and maintenance actions.

Everything shown here is read from `settings`, so this page doubles as proof
that configuration really does come from `.env` or Streamlit secrets, and
nowhere else. The API key is never displayed - only whether one is present.
"""

from __future__ import annotations

import streamlit as st

from src.config import llm_available, settings
from src.data import seed_database, table_counts
from src.llmops import clear_logs, load_evaluations, load_runs
from src.rag import build_patient_index, store_stats
from src.rag.embeddings import embedding_dimension
from src.tools import tool_overview
from src.ui.components import dataframe, metric_row, page_header


def _configuration_tab() -> None:
    st.caption(
        "Loaded from Streamlit secrets or `.env` via `src/config.py`. "
        "The API key value is never shown."
    )

    st.subheader("Language model")
    dataframe(
        [
            {"Setting": "Agent model", "Value": settings.groq_model},
            {"Setting": "Judge model", "Value": settings.groq_eval_model},
            {"Setting": "Temperature", "Value": settings.groq_temperature},
            {"Setting": "Max tokens", "Value": settings.groq_max_tokens},
            {"Setting": "Timeout (s)", "Value": settings.groq_timeout},
            {"Setting": "API key present", "Value": "yes" if llm_available() else "no"},
        ]
    )

    st.subheader("Retrieval")
    dataframe(
        [
            {"Setting": "Embedding model", "Value": settings.embedding_model},
            {"Setting": "Vector store", "Value": settings.vector_db},
            {"Setting": "Top k", "Value": settings.faiss_top_k},
            {"Setting": "Chunk size", "Value": settings.chunk_size},
            {"Setting": "Chunk overlap", "Value": settings.chunk_overlap},
        ]
    )

    st.subheader("Paths and memory")
    dataframe(
        [
            {"Setting": "Database", "Value": str(settings.ehr_db_path)},
            {"Setting": "FAISS directory", "Value": str(settings.faiss_index_dir)},
            {"Setting": "Log directory", "Value": str(settings.log_dir)},
            {"Setting": "Memory window (turns)", "Value": settings.memory_window},
        ]
    )

    problems = settings.missing_settings()
    if problems:
        st.warning("Configuration issues:\n\n" + "\n".join(f"- {p}" for p in problems))
    else:
        st.success("Configuration is complete.")


def _data_tab() -> None:
    st.subheader("Database")
    counts = table_counts()
    metric_row([(table.replace("_", " ").title(), count) for table, count in counts.items()], 3)

    st.subheader("Vector indexes")
    stats = store_stats()
    dataframe(
        [
            {
                "Index": name,
                "Vectors": info["vectors"],
                "Built": info["built"],
                "Path": info["path"],
            }
            for name, info in stats.items()
        ]
    )
    st.caption(f"Embedding dimension: {embedding_dimension()}")

    st.subheader("Logs")
    metric_row([("Runs logged", len(load_runs())), ("Evaluations logged", len(load_evaluations()))])

    st.divider()
    st.subheader("Maintenance")
    st.caption(
        "FAISS has no cheap in-place update, so the patient index is rebuilt rather "
        "than patched. Rebuilding takes about a second at this data size."
    )

    left, middle, right = st.columns(3)

    if left.button("Rebuild patient index", width="stretch"):
        with st.spinner("Rebuilding..."):
            result = build_patient_index(rebuild=True)
        st.success(
            f"{result['patients']} patients indexed as {result['chunks']} chunks "
            f"({result['vectors']} vectors)."
        )

    if middle.button("Re-seed demo data", width="stretch"):
        with st.spinner("Seeding..."):
            counts = seed_database(reset=True)
        st.cache_data.clear()
        st.success(f"Database reset. Doctors: {counts['doctors']}, slots: {counts['slots']}.")

    if right.button("Clear logs", width="stretch"):
        clear_logs()
        st.success("Run and evaluation logs cleared.")


def _tools_tab() -> None:
    st.caption(
        "The tools bound to the agent. These descriptions are what the model sees when "
        "deciding which tool to call, so their wording directly affects behaviour."
    )
    dataframe(
        [
            {"Tool": tool["name"], "Description": tool["description"]}
            for tool in tool_overview()
        ]
    )


def render() -> None:
    page_header("System", "Configuration, data, indexes, and maintenance.")
    configuration, data, tools = st.tabs(["Configuration", "Data and indexes", "Agent tools"])
    with configuration:
        _configuration_tab()
    with data:
        _data_tab()
    with tools:
        _tools_tab()
