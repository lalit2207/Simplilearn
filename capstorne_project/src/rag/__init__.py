"""Retrieval-Augmented Generation: embeddings, FAISS indexes, retrieval."""

from src.rag.embeddings import embedding_dimension, get_embeddings
from src.rag.indexer import (
    answer_context,
    build_patient_index,
    fetch_and_index,
    format_context,
    index_search_results,
    retrieve_medical_knowledge,
    retrieve_patient_context,
)
from src.rag.vector_store import knowledge_store, patient_store, store_stats

__all__ = [
    "answer_context",
    "build_patient_index",
    "embedding_dimension",
    "fetch_and_index",
    "format_context",
    "get_embeddings",
    "index_search_results",
    "knowledge_store",
    "patient_store",
    "retrieve_medical_knowledge",
    "retrieve_patient_context",
    "store_stats",
]
