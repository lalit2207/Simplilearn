"""Central configuration for the Agentic Healthcare Assistant.

Values are read from Streamlit secrets first (used on Streamlit Community
Cloud), then from the process environment / local `.env`. The API key must
never be committed. Nothing else in the codebase should call `os.getenv`
directly - import `settings` from here instead.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]

load_dotenv(PROJECT_ROOT / ".env")


def _from_streamlit_secrets(key: str) -> str | None:
    """Read a value from Streamlit secrets only while the app is running.

    Secrets live in the Streamlit Cloud dashboard (or a local gitignored
    `.streamlit/secrets.toml`). Scripts and tests skip this path so they
    keep using `.env` and do not trigger a Streamlit secrets warning.
    """
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx

        if get_script_run_ctx() is None:
            return None
        import streamlit as st

        value = st.secrets.get(key)
    except Exception:
        return None
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _str(key: str, default: str = "") -> str:
    secret = _from_streamlit_secrets(key)
    if secret is not None:
        return secret
    value = os.getenv(key)
    return default if value is None or value.strip() == "" else value.strip()


def _int(key: str, default: int) -> int:
    try:
        return int(_str(key, str(default)))
    except ValueError:
        return default


def _float(key: str, default: float) -> float:
    try:
        return float(_str(key, str(default)))
    except ValueError:
        return default


def _bool(key: str, default: bool = False) -> bool:
    return _str(key, str(default)).lower() in {"1", "true", "yes", "on"}


def _path(key: str, default: str) -> Path:
    """Resolve a configured path relative to the project root."""
    raw = Path(_str(key, default))
    return raw if raw.is_absolute() else PROJECT_ROOT / raw


@dataclass(frozen=True)
class Settings:
    # LLM (Groq)
    groq_api_key: str
    groq_model: str
    groq_eval_model: str
    groq_vision_model: str
    groq_temperature: float
    groq_max_tokens: int
    groq_timeout: int

    # Embeddings + vector store
    embedding_model: str
    vector_db: str
    faiss_index_dir: Path
    faiss_top_k: int
    chunk_size: int
    chunk_overlap: int

    # Local data stores
    ehr_db_path: Path
    seed_on_startup: bool

    # Medical information search
    medline_search_url: str
    who_factsheet_url: str
    search_max_results: int
    search_timeout: int

    # Application
    app_title: str
    log_dir: Path
    log_level: str
    memory_window: int

    @classmethod
    def load(cls) -> "Settings":
        return cls(
            groq_api_key=_str("GROQ_API_KEY"),
            groq_model=_str("GROQ_MODEL", "openai/gpt-oss-120b"),
            groq_eval_model=_str("GROQ_EVAL_MODEL", "openai/gpt-oss-20b"),
            groq_vision_model=_str(
                "GROQ_VISION_MODEL", "meta-llama/llama-4-scout-17b-16e-instruct"
            ),
            groq_temperature=_float("GROQ_TEMPERATURE", 0.2),
            groq_max_tokens=_int("GROQ_MAX_TOKENS", 2048),
            groq_timeout=_int("GROQ_TIMEOUT", 60),
            embedding_model=_str(
                "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
            ),
            vector_db=_str("VECTOR_DB", "faiss").lower(),
            faiss_index_dir=_path("FAISS_INDEX_DIR", "data/faiss_index"),
            faiss_top_k=_int("FAISS_TOP_K", 4),
            chunk_size=_int("CHUNK_SIZE", 800),
            chunk_overlap=_int("CHUNK_OVERLAP", 120),
            ehr_db_path=_path("EHR_DB_PATH", "data/healthcare.db"),
            seed_on_startup=_bool("SEED_ON_STARTUP", True),
            medline_search_url=_str(
                "MEDLINE_SEARCH_URL", "https://wsearch.nlm.nih.gov/ws/query"
            ),
            who_factsheet_url=_str(
                "WHO_FACTSHEET_URL", "https://www.who.int/news-room/fact-sheets"
            ),
            search_max_results=_int("SEARCH_MAX_RESULTS", 5),
            search_timeout=_int("SEARCH_TIMEOUT", 20),
            app_title=_str("APP_TITLE", "Health Care Assistant Agent"),
            log_dir=_path("LOG_DIR", "logs"),
            log_level=_str("LOG_LEVEL", "INFO").upper(),
            memory_window=_int("MEMORY_WINDOW", 10),
        )

    @property
    def llm_ready(self) -> bool:
        """True when a usable Groq key is present."""
        return bool(self.groq_api_key) and not self.groq_api_key.startswith("PASTE_")

    def ensure_directories(self) -> None:
        """Create the folders the app writes into."""
        for directory in (
            self.faiss_index_dir,
            self.ehr_db_path.parent,
            self.log_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)

    def missing_settings(self) -> list[str]:
        """Report configuration problems instead of failing at import time."""
        problems: list[str] = []
        if not self.llm_ready:
            problems.append(
                "GROQ_API_KEY is not set (add it to .env locally, or to "
                "Streamlit secrets when hosted)"
            )
        if self.vector_db != "faiss":
            problems.append(f"VECTOR_DB={self.vector_db!r} is not supported (use 'faiss')")
        if self.chunk_overlap >= self.chunk_size:
            problems.append("CHUNK_OVERLAP must be smaller than CHUNK_SIZE")
        return problems


settings = Settings.load()


def llm_available() -> bool:
    """Whether a usable Groq key is present in secrets or `.env`."""
    return settings.llm_ready
