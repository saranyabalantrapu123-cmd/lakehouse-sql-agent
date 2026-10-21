"""Paths and limits shared across the package (overridable via environment variables)."""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = Path(os.getenv("LAKEHOUSE_RAW_DIR", DATA_DIR / "raw"))
DB_PATH = Path(os.getenv("LAKEHOUSE_DB", DATA_DIR / "lakehouse.duckdb"))
MANIFEST_PATH = DB_PATH.with_name("build_manifest.json")
CATALOG_PATH = PROJECT_ROOT / "catalog.yaml"

MAX_ROWS = int(os.getenv("SQL_MAX_ROWS", "200"))
QUERY_TIMEOUT_S = float(os.getenv("SQL_TIMEOUT_SECONDS", "15"))
MAX_SQL_RETRIES = 2
