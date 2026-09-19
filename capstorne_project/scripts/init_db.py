"""Create and seed the local healthcare database.

    python scripts/init_db.py            # create + seed if empty
    python scripts/init_db.py --reset    # wipe and rebuild from scratch
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import settings  # noqa: E402
from src.data import seed_database  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Initialise the healthcare database.")
    parser.add_argument(
        "--reset",
        action="store_true",
        help="delete the existing database file before rebuilding",
    )
    args = parser.parse_args()

    settings.ensure_directories()
    counts = seed_database(reset=args.reset)

    print(f"Database: {settings.ehr_db_path}")
    print("Row counts:")
    for table, count in counts.items():
        print(f"  {table:<16} {count:>5}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
