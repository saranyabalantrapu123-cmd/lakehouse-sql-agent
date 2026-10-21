"""Shared fixtures: build a tiny lakehouse from tests/fixtures (no download, no network, no key)."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

# Force offline mode for every test, regardless of the developer's environment.
os.environ["LLM_MODEL"] = "mock"
# Use LiteLLM's bundled price map instead of fetching it over the network.
os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")

from lakehouse_agent.build_lakehouse import build  # noqa: E402
from lakehouse_agent.catalog import load_catalog  # noqa: E402
from lakehouse_agent.tools import LakehouseTools  # noqa: E402

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session")
def lakehouse(tmp_path_factory) -> tuple[Path, dict]:
    db_path = tmp_path_factory.mktemp("lake") / "lakehouse.duckdb"
    manifest = build(FIXTURES, db_path, quiet=True)
    return db_path, manifest


@pytest.fixture(scope="session")
def catalog():
    return load_catalog()


@pytest.fixture(scope="session")
def tools(lakehouse, catalog) -> LakehouseTools:
    db_path, _ = lakehouse
    return LakehouseTools(db_path, catalog)
