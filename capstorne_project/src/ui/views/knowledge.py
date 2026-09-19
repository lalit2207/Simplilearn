"""Attendant view: retrieved medical information.

Covers "summary of latest retrieved medical information". Searches hit the
FAISS cache first and only reach MedlinePlus/WHO when the cache is thin, which
is visible in the two tabs.
"""

from __future__ import annotations

import streamlit as st

from src.rag import fetch_and_index, retrieve_medical_knowledge
from src.rag.vector_store import knowledge_store
from src.tools.medical_search import search_medical_information
from src.ui.components import metric_row, page_header

SUGGESTIONS = [
    "chronic kidney disease treatment",
    "diabetic kidney disease",
    "hypertension and kidney damage",
    "dialysis options",
    "low protein renal diet",
]


def render() -> None:
    page_header(
        "Medical Information",
        "Disease information from MedlinePlus (US National Library of Medicine) and WHO.",
    )

    metric_row(
        [
            ("Cached chunks", knowledge_store.count()),
            ("Index built", "yes" if knowledge_store.exists() else "no"),
        ]
    )

    query = st.text_input(
        "Search a condition, symptom, or treatment",
        placeholder="chronic kidney disease treatment",
    )
    st.caption("Suggestions: " + " · ".join(f"`{s}`" for s in SUGGESTIONS))

    cached_tab, live_tab = st.tabs(["From the cache (fast)", "Live fetch (slower)"])

    with cached_tab:
        st.caption("Semantic search over articles already indexed in FAISS.")
        if query:
            hits = retrieve_medical_knowledge(query, k=5)
            if not hits:
                st.info(
                    "Nothing cached for this query yet. Use the live fetch tab to "
                    "retrieve and index it."
                )
            for hit in hits:
                st.markdown(f"**{hit['title']}** — {hit['source']}  ·  distance `{hit['score']}`")
                st.write(hit["text"])
                if hit["url"]:
                    st.markdown(f"[Read the full article]({hit['url']})")
                st.divider()

    with live_tab:
        st.caption(
            "Calls MedlinePlus and WHO directly, then adds the results to the index. "
            "Expect a few seconds. Not every topic has a WHO fact sheet."
        )
        if st.button("Fetch and index", type="primary", disabled=not query):
            with st.spinner(f"Searching trusted sources for '{query}'..."):
                results = search_medical_information(query)
                outcome = fetch_and_index(query)

            if not results:
                st.warning("No results returned for that query.")
            else:
                st.success(
                    f"{outcome['results']} results indexed as {outcome['chunks_indexed']} "
                    f"chunks. Sources: {', '.join(outcome['sources'])}."
                )
                st.session_state["last_search"] = results

        if st.session_state.get("last_search"):
            st.subheader("Latest retrieved information")
            for index, result in enumerate(st.session_state["last_search"], start=1):
                with st.expander(f"[{index}] {result['title']} — {result['source']}"):
                    if result.get("also_called"):
                        st.caption(f"Also called: {result['also_called']}")
                    st.write(result["summary"])
                    if result.get("url"):
                        st.markdown(f"[Source]({result['url']})")
