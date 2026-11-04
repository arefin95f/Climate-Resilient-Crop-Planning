"""Load the crop modeling files into MySQL.

Usage:
    python scripts/load_mysql.py

Connection settings come from the environment or a .env file:
    MYSQL_HOST, MYSQL_PORT, MYSQL_USER, MYSQL_PASSWORD, MYSQL_DATABASE
"""

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import CROP_CATALOG, CROP_DATA_DIR
from src.db import mysql_settings, publish_datasets, publish_trained_models


def main() -> None:
    frames = []
    for crop_id in CROP_CATALOG:
        path = CROP_DATA_DIR / f"{crop_id}.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing crop dataset: {path}")
        frames.append(pd.read_csv(path))

    settings = mysql_settings()
    print(
        f"Loading {len(frames)} crops into MySQL database "
        f"'{settings['database']}' at {settings['host']}:{settings['port']}"
    )
    counts = publish_datasets(frames)
    counts["trained_models"] = publish_trained_models()
    for name, count in counts.items():
        print(f"  {name}: {count}")
    print("MySQL load complete.")


if __name__ == "__main__":
    main()
