"""Verify that the environment is configured before building further steps.

Run from the project root:

    python scripts/check_setup.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import PROJECT_ROOT, settings  # noqa: E402


def main() -> int:
    print(f"Project root      : {PROJECT_ROOT}")
    print(f"App title         : {settings.app_title}")
    print(f"Groq model        : {settings.groq_model}")
    print(f"Groq eval model   : {settings.groq_eval_model}")
    print(f"Embedding model   : {settings.embedding_model}")
    print(f"Vector store      : {settings.vector_db} -> {settings.faiss_index_dir}")
    print(f"EHR database      : {settings.ehr_db_path}")
    print(f"Log directory     : {settings.log_dir}")
    print(f"Groq key present  : {settings.llm_ready}")

    settings.ensure_directories()
    print("\nCreated required directories.")

    problems = settings.missing_settings()
    if problems:
        print("\nConfiguration issues:")
        for problem in problems:
            print(f"  - {problem}")
        return 1

    print("\nConfiguration looks good.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
