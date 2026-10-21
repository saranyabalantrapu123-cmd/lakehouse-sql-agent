"""MCP server exposing the lakehouse tools (list_tables, describe_table, get_lineage, run_sql).

Any MCP client (Claude Desktop, Claude Code, IDE agents) can then discover the bronze/silver/gold
tables, read keys and lineage, and run guarded read-only SQL. Runs over stdio:

    python mcp_server.py
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

try:  # mcp >= 2.0
    from mcp.server.mcpserver import MCPServer as _Server
except ImportError:  # mcp 1.x
    from mcp.server.fastmcp import FastMCP as _Server

from lakehouse_agent.tools import LakehouseTools

server = _Server(
    "lakehouse-sql-agent",
    instructions=(
        "Data discovery over an Olist e-commerce lakehouse with bronze (raw), silver (clean) and gold "
        "(star schema + marts) layers. Use list_tables and describe_table before writing SQL; prefer gold. "
        "run_sql accepts one read-only SELECT with schema-qualified table names and returns at most 200 rows."
    ),
)


@lru_cache(maxsize=1)
def _tools() -> LakehouseTools:
    # Created lazily so the server can start (and report a clear error) even before `make build`.
    return LakehouseTools()


@server.tool()
def list_tables(layer: str | None = None) -> dict[str, Any]:
    """List lakehouse tables with description, row count and primary key. layer: bronze | silver | gold."""
    return _tools().list_tables(layer)


@server.tool()
def describe_table(name: str) -> dict[str, Any]:
    """Columns (type + description), primary/foreign keys, referencing tables, lineage and sample rows."""
    return _tools().describe_table(name)


@server.tool()
def get_lineage(name: str) -> dict[str, Any]:
    """Upstream sources and downstream dependents of a table across bronze/silver/gold."""
    return _tools().get_lineage(name)


@server.tool()
def run_sql(sql: str) -> dict[str, Any]:
    """Run ONE read-only DuckDB SELECT/WITH over catalog tables (e.g. gold.fact_orders). Max 200 rows."""
    return _tools().run_sql(sql)


if __name__ == "__main__":
    server.run()
