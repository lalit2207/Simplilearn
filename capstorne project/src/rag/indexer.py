"""Populating and querying the two FAISS indexes.

This is the bridge between Step 2/3 (SQLite + search tools) and the retrieval
the agent performs in Step 5. Nothing here talks to an LLM.
"""

from __future__ import annotations

from datetime import datetime

from langchain_core.documents import Document

from src.data.database import fetch_all
from src.rag.vector_store import knowledge_store, patient_store
from src.tools.medical_search import search_medical_information
from src.tools.patient_tools import build_history_digest, get_patient_profile


# --------------------------------------------------------------------------
# Patient index
# --------------------------------------------------------------------------
def _patient_documents(patient_id: int) -> list[Document]:
    """One document per medical record, plus a profile document.

    Records are indexed individually rather than as one blob so retrieval can
    return the two relevant visits instead of the entire chart.
    """
    profile = get_patient_profile(patient_id)
    if not profile:
        return []

    base_metadata = {
        "patient_id": patient_id,
        "patient_name": profile["full_name"],
        "source": "ehr",
    }

    documents = [
        Document(
            page_content=(
                f"Patient profile. Name: {profile['full_name']}. "
                f"Age: {profile.get('age')}. Gender: {profile.get('gender')}. "
                f"Blood group: {profile.get('blood_group')}. "
                f"Known allergies: {profile.get('allergies') or 'none recorded'}."
            ),
            metadata={**base_metadata, "doc_type": "profile", "record_id": 0},
        )
    ]

    records = fetch_all(
        """SELECT id, record_date, record_type, condition, diagnosis,
                  treatment, medications, notes, alert_level
           FROM medical_history WHERE patient_id = ? ORDER BY record_date""",
        (patient_id,),
    )

    for record in records:
        parts = [
            f"Medical record dated {record['record_date']} "
            f"for {profile['full_name']} ({record['record_type']}).",
        ]
        for label, key in (
            ("Condition", "condition"),
            ("Diagnosis", "diagnosis"),
            ("Treatment", "treatment"),
            ("Medications", "medications"),
            ("Notes", "notes"),
        ):
            if record.get(key):
                parts.append(f"{label}: {record[key]}")
        parts.append(f"Alert level: {record['alert_level']}.")

        documents.append(
            Document(
                page_content=" ".join(parts),
                metadata={
                    **base_metadata,
                    "doc_type": "record",
                    "record_id": record["id"],
                    "record_date": record["record_date"],
                    "record_type": record["record_type"],
                    "condition": record.get("condition") or "",
                    "alert_level": record["alert_level"],
                },
            )
        )
    return documents


def build_patient_index(rebuild: bool = False) -> dict:
    """Index every patient's chart.

    FAISS has no cheap in-place update, so a refresh rebuilds the whole
    patient index. With a handful of patients this takes under a second and
    avoids stale duplicate chunks.
    """
    if rebuild:
        patient_store.reset()

    patient_ids = [row["id"] for row in fetch_all("SELECT id FROM patients ORDER BY id")]
    documents: list[Document] = []
    for patient_id in patient_ids:
        documents.extend(_patient_documents(patient_id))

    chunks = patient_store.add_documents(documents)
    return {
        "patients": len(patient_ids),
        "documents": len(documents),
        "chunks": chunks,
        "vectors": patient_store.count(),
    }


def retrieve_patient_context(patient_id: int, query: str, k: int | None = None) -> list[dict]:
    """Retrieve only the parts of one patient's chart relevant to `query`.

    The metadata filter is the privacy boundary: without it a query could
    surface another patient's records.
    """
    hits = patient_store.search(query, k=k, filter={"patient_id": patient_id})
    return [
        {
            "text": document.page_content,
            "score": round(float(score), 4),
            "record_date": document.metadata.get("record_date", ""),
            "record_type": document.metadata.get("record_type", "profile"),
            "alert_level": document.metadata.get("alert_level", ""),
        }
        for document, score in hits
    ]


# --------------------------------------------------------------------------
# Knowledge index
# --------------------------------------------------------------------------
def index_search_results(query: str, results: list[dict]) -> int:
    """Store fetched MedlinePlus/WHO articles so later questions reuse them."""
    documents = [
        Document(
            page_content=f"{result['title']}\n\n{result['summary']}",
            metadata={
                "source": result.get("source", "unknown"),
                "title": result.get("title", ""),
                "url": result.get("url", ""),
                "query": query,
                "fetched_at": datetime.now().isoformat(timespec="seconds"),
                "doc_type": "article",
            },
        )
        for result in results
        if result.get("summary")
    ]
    return knowledge_store.add_documents(documents)


def fetch_and_index(query: str, max_results: int | None = None) -> dict:
    """Live-search the trusted sources and add whatever comes back to FAISS."""
    results = search_medical_information(query, max_results)
    chunks = index_search_results(query, results)
    return {
        "query": query,
        "results": len(results),
        "chunks_indexed": chunks,
        "sources": sorted({r.get("source", "unknown") for r in results}),
    }


def retrieve_medical_knowledge(query: str, k: int | None = None) -> list[dict]:
    """Search the cached article index."""
    hits = knowledge_store.search(query, k=k)
    return [
        {
            "text": document.page_content,
            "score": round(float(score), 4),
            "title": document.metadata.get("title", ""),
            "url": document.metadata.get("url", ""),
            "source": document.metadata.get("source", ""),
        }
        for document, score in hits
    ]


def answer_context(query: str, k: int | None = None, refresh: bool = True) -> list[dict]:
    """Retrieve knowledge for a query, fetching from the web if the index is thin.

    This is the actual RAG entry point the agent uses: cache first, live
    fetch only when the cache can't answer.
    """
    hits = retrieve_medical_knowledge(query, k=k)
    if hits or not refresh:
        return hits
    fetch_and_index(query)
    return retrieve_medical_knowledge(query, k=k)


def format_context(hits: list[dict]) -> str:
    """Render retrieved chunks as citable context for a prompt."""
    if not hits:
        return "No retrieved context available."
    blocks = []
    for index, hit in enumerate(hits, start=1):
        header = f"[{index}]"
        if hit.get("title"):
            header += f" {hit['title']}"
        if hit.get("source"):
            header += f" ({hit['source']})"
        if hit.get("record_date"):
            header += f" record dated {hit['record_date']}"
        header += f" [distance {hit['score']}]"
        blocks.append(f"{header}\n{hit['text']}")
        if hit.get("url"):
            blocks[-1] += f"\nURL: {hit['url']}"
    return "\n\n".join(blocks)
