"""Embedding model loader.

Groq serves chat completions only - it has no embedding endpoint - so
embeddings run locally through sentence-transformers. That keeps the vector
store free of API calls, rate limits, and per-token cost.

The model is a module-level singleton because loading it takes a few seconds
and Streamlit re-runs the whole script on every interaction.
"""

from __future__ import annotations

from langchain_huggingface import HuggingFaceEmbeddings

from src.config import settings

_embeddings: HuggingFaceEmbeddings | None = None


def get_embeddings() -> HuggingFaceEmbeddings:
    """Return the shared embedding model, loading it on first use.

    The first call downloads the model (~90 MB) into the Hugging Face cache;
    later calls are instant.
    """
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(
            model_name=settings.embedding_model,
            model_kwargs={"device": "cpu"},
            # Normalised vectors make FAISS inner-product scores behave like
            # cosine similarity, which is what the relevance filtering assumes.
            encode_kwargs={"normalize_embeddings": True},
        )
    return _embeddings


def embedding_dimension() -> int:
    """Vector width of the configured model - handy for the UI panel."""
    return len(get_embeddings().embed_query("dimension probe"))
