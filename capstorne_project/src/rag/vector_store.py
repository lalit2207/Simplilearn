"""FAISS vector store wrapper.

The assistant keeps two separate indexes rather than one shared index:

  * `patients`  - chunks of each patient's chart (private, per-patient)
  * `knowledge` - chunks of MedlinePlus/WHO articles already fetched (public)

They are kept apart on purpose. A search for "kidney treatment options" must
never pull one patient's notes into another patient's answer, and mixing
private records with public articles in one index makes that mistake easy.
"""

from __future__ import annotations

from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import settings
from src.rag.embeddings import get_embeddings

PATIENT_COLLECTION = "patients"
KNOWLEDGE_COLLECTION = "knowledge"


def get_splitter() -> RecursiveCharacterTextSplitter:
    """Split on paragraph, then line, then sentence, then word boundaries."""
    return RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )


class VectorStore:
    """A single named FAISS index persisted under the configured directory."""

    def __init__(self, collection: str):
        self.collection = collection
        self.path: Path = settings.faiss_index_dir / collection
        self._store: FAISS | None = None

    # -- persistence -------------------------------------------------------
    def exists(self) -> bool:
        return (self.path / "index.faiss").exists()

    def load(self) -> FAISS | None:
        """Load the index from disk, or return None if it was never built."""
        if self._store is not None:
            return self._store
        if not self.exists():
            return None
        self._store = FAISS.load_local(
            str(self.path),
            get_embeddings(),
            # Safe here because we wrote this file ourselves; FAISS uses
            # pickle for the docstore and refuses to load without this flag.
            allow_dangerous_deserialization=True,
        )
        return self._store

    def save(self) -> None:
        if self._store is None:
            return
        self.path.mkdir(parents=True, exist_ok=True)
        self._store.save_local(str(self.path))

    def reset(self) -> None:
        """Drop the in-memory and on-disk index."""
        self._store = None
        if self.path.exists():
            for file in self.path.iterdir():
                file.unlink()

    # -- writing -----------------------------------------------------------
    def add_documents(self, documents: list[Document]) -> int:
        """Chunk and add documents, creating the index if it doesn't exist yet."""
        if not documents:
            return 0

        chunks = get_splitter().split_documents(documents)
        if not chunks:
            return 0

        store = self.load()
        if store is None:
            self._store = FAISS.from_documents(chunks, get_embeddings())
        else:
            store.add_documents(chunks)
        self.save()
        return len(chunks)

    def add_texts(self, texts: list[str], metadatas: list[dict] | None = None) -> int:
        metadatas = metadatas or [{} for _ in texts]
        return self.add_documents(
            [Document(page_content=text, metadata=meta) for text, meta in zip(texts, metadatas)]
        )

    # -- reading -----------------------------------------------------------
    def search(
        self,
        query: str,
        k: int | None = None,
        filter: dict | None = None,
        min_score: float | None = None,
    ) -> list[tuple[Document, float]]:
        """Similarity search returning (document, score) pairs.

        FAISS returns L2 distance, so **lower is closer**. `min_score` is
        therefore an upper bound on distance, applied after retrieval.
        """
        store = self.load()
        if store is None:
            return []

        top_k = k or settings.faiss_top_k
        # Over-fetch when filtering, since FAISS applies the metadata filter
        # after the nearest-neighbour search and can otherwise return nothing.
        fetch_k = top_k * 4 if filter else top_k
        results = store.similarity_search_with_score(
            query, k=top_k, filter=filter, fetch_k=fetch_k
        )
        if min_score is not None:
            results = [(doc, score) for doc, score in results if score <= min_score]
        return results

    def count(self) -> int:
        store = self.load()
        return 0 if store is None else store.index.ntotal


# Module-level handles so callers share one loaded index each.
patient_store = VectorStore(PATIENT_COLLECTION)
knowledge_store = VectorStore(KNOWLEDGE_COLLECTION)


def store_stats() -> dict[str, dict]:
    """Index sizes for the Streamlit diagnostics panel."""
    return {
        store.collection: {
            "vectors": store.count(),
            "path": str(store.path),
            "built": store.exists(),
        }
        for store in (patient_store, knowledge_store)
    }
