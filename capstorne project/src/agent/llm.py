"""Groq chat model factory.

Two models are configured in `.env` for different jobs:

  * GROQ_MODEL      - the planner and responder; needs reasoning quality
  * GROQ_EVAL_MODEL - scoring and lightweight calls; needs speed and low cost

Note that Groq shut down the Llama 3.x models on 2026-08-16, so the defaults
are the GPT-OSS models Groq recommends in their place.
"""

from __future__ import annotations

from langchain_groq import ChatGroq

from src.config import llm_available, settings


class MissingAPIKeyError(RuntimeError):
    """Raised when a Groq call is attempted without a configured key."""


def _require_key() -> str:
    """The Groq key from Streamlit secrets or `.env`."""
    if not settings.llm_ready:
        raise MissingAPIKeyError(
            "GROQ_API_KEY is missing or still a placeholder. Add it to .env "
            "locally, or to Streamlit secrets when hosted "
            "(get one free at https://console.groq.com/keys) and restart."
        )
    return settings.groq_api_key


def get_llm(
    temperature: float | None = None,
    model: str | None = None,
    max_tokens: int | None = None,
) -> ChatGroq:
    """Main reasoning model used by the planner and the agent loop."""
    return ChatGroq(
        model=model or settings.groq_model,
        temperature=settings.groq_temperature if temperature is None else temperature,
        max_tokens=max_tokens or settings.groq_max_tokens,
        timeout=settings.groq_timeout,
        groq_api_key=_require_key(),
    )


def get_eval_llm(temperature: float = 0.0) -> ChatGroq:
    """Cheaper, faster model for evaluation. Temperature 0 keeps scores stable."""
    return ChatGroq(
        model=settings.groq_eval_model,
        temperature=temperature,
        max_tokens=settings.groq_max_tokens,
        timeout=settings.groq_timeout,
        groq_api_key=_require_key(),
    )


def llm_status() -> dict:
    """Non-throwing status report for the UI."""
    return {
        "ready": llm_available(),
        "model": settings.groq_model,
        "eval_model": settings.groq_eval_model,
        "temperature": settings.groq_temperature,
    }
