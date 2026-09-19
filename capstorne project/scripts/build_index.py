"""Build the FAISS indexes.

    python scripts/build_index.py                    # patient charts only
    python scripts/build_index.py --rebuild          # wipe and rebuild
    python scripts/build_index.py --seed-knowledge   # also pre-fetch articles

The first run downloads the sentence-transformers model (~90 MB).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import settings  # noqa: E402
from src.rag import build_patient_index, fetch_and_index, store_stats  # noqa: E402
from src.rag.embeddings import embedding_dimension  # noqa: E402

# Conditions already present in the seeded charts, so the knowledge index
# starts out able to answer questions about the demo patients.
SEED_TOPICS = [
    "chronic kidney disease",
    "type 2 diabetes",
    "hypertension",
    "hypothyroidism",
    "osteoarthritis",
]


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the FAISS indexes.")
    parser.add_argument("--rebuild", action="store_true", help="wipe the patient index first")
    parser.add_argument(
        "--seed-knowledge",
        action="store_true",
        help="pre-fetch MedlinePlus/WHO articles for the seeded conditions",
    )
    args = parser.parse_args()

    settings.ensure_directories()

    print(f"Embedding model: {settings.embedding_model}")
    print("Loading model (first run downloads it)...")
    print(f"Vector dimension: {embedding_dimension()}\n")

    result = build_patient_index(rebuild=args.rebuild)
    print(
        f"Patient index: {result['patients']} patients -> "
        f"{result['documents']} documents -> {result['chunks']} chunks "
        f"({result['vectors']} vectors total)"
    )

    if args.seed_knowledge:
        print("\nFetching medical knowledge:")
        for topic in SEED_TOPICS:
            outcome = fetch_and_index(topic)
            print(
                f"  {topic:<26} {outcome['results']} results, "
                f"{outcome['chunks_indexed']} chunks, "
                f"sources: {', '.join(outcome['sources']) or 'none'}"
            )

    print("\nIndex status:")
    for name, stats in store_stats().items():
        print(f"  {name:<10} {stats['vectors']:>5} vectors  built={stats['built']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
