"""Agent tools: plain Python functions shared by the LangGraph agent, the MCP server and the UI.

Every public method returns JSON-serialisable dicts so results can be passed straight back
to an LLM as tool output.
"""

from __future__ import annotations

import datetime as dt
import decimal
import threading
import time
from pathlib import Path
from typing import Any

import duckdb
import sqlglot
from sqlglot import exp

from lakehouse_agent.catalog import LAYERS, Catalog, load_catalog
from lakehouse_agent.config import CATALOG_PATH, DB_PATH, MAX_ROWS, QUERY_TIMEOUT_S
from lakehouse_agent.guardrails import GuardrailError, validate_sql


def to_jsonable(value: Any) -> Any:
    """Convert DuckDB result values into JSON-friendly Python types."""
    if isinstance(value, decimal.Decimal):
        return float(value)
    if isinstance(value, (dt.datetime, dt.date, dt.time)):
        return value.isoformat()
    if isinstance(value, dt.timedelta):
        return value.total_seconds()
    if isinstance(value, (bytes, bytearray)):
        return value.hex()
    return value


class LakehouseTools:
    """Read-only access to the lakehouse, scoped to tables in the catalog."""

    def __init__(self, db_path: Path | str = DB_PATH, catalog: Catalog | None = None,
                 max_rows: int = MAX_ROWS, timeout_s: float = QUERY_TIMEOUT_S):
        self.db_path = Path(db_path)
        self.catalog = catalog or load_catalog(CATALOG_PATH)
        self.max_rows = max_rows
        self.timeout_s = timeout_s
        if not self.db_path.exists():
            raise FileNotFoundError(f"Lakehouse not found at {self.db_path}. Run `make data build` first.")

    # ------------------------------------------------------------ connection
    def _connect(self) -> duckdb.DuckDBPyConnection:
        # Read-only file handle + no filesystem/network access from SQL.
        return duckdb.connect(str(self.db_path), read_only=True, config={"enable_external_access": False})

    def _row_count(self, con: duckdb.DuckDBPyConnection, name: str) -> int:
        return int(con.execute(f"SELECT count(*) FROM {name}").fetchone()[0])

    def _resolve(self, name: str):
        info = self.catalog.resolve(name)
        if info is None:
            options = self.catalog.candidates(name)
            hint = f" Candidates: {', '.join(options)}." if options else ""
            raise KeyError(f"Unknown table '{name}'.{hint} Use list_tables to see available tables.")
        return info

    # ----------------------------------------------------------------- tools
    def list_tables(self, layer: str | None = None) -> dict:
        """List catalog tables (optionally for one layer) with description, row count and primary key."""
        if layer:
            layer = layer.lower().strip()
            if layer not in LAYERS:
                return {"error": f"Unknown layer '{layer}'. Use one of: {', '.join(LAYERS)}."}
        with self._connect() as con:
            tables = [
                {
                    "name": t.name,
                    "layer": t.layer,
                    "description": t.description,
                    "primary_key": list(t.primary_key),
                    "row_count": self._row_count(con, t.name),
                }
                for t in self.catalog.by_layer(layer)
            ]
        return {
            "layers": {k: v for k, v in self.catalog.layer_descriptions.items() if layer in (None, k)},
            "tables": tables,
        }

    def describe_table(self, name: str, sample_rows: int = 3) -> dict:
        """Columns (type + description), primary/foreign keys, lineage and sample rows for a table."""
        try:
            info = self._resolve(name)
        except KeyError as exc:
            return {"error": str(exc.args[0])}
        schema, table = info.name.split(".")
        with self._connect() as con:
            types = dict(
                con.execute(
                    "SELECT column_name, data_type FROM information_schema.columns "
                    "WHERE table_schema = ? AND table_name = ? ORDER BY ordinal_position",
                    [schema, table],
                ).fetchall()
            )
            cur = con.execute(f"SELECT * FROM {info.name} LIMIT {int(sample_rows)}")
            sample_cols = [d[0] for d in cur.description]
            samples = [
                {c: _shorten(to_jsonable(v)) for c, v in zip(sample_cols, row)} for row in cur.fetchall()
            ]
            row_count = self._row_count(con, info.name)

        referenced_by = [
            {"table": src, **fk.as_dict()} for src, fk in self.catalog.fk_edges() if fk.references == info.name
        ]
        return {
            "name": info.name,
            "layer": info.layer,
            "description": info.description,
            "row_count": row_count,
            "primary_key": list(info.primary_key),
            "foreign_keys": [fk.as_dict() for fk in info.foreign_keys],
            "referenced_by": referenced_by,
            "upstream": list(info.upstream),
            "downstream": self.catalog.downstream(info.name),
            "source_file": info.source,
            "columns": [
                {
                    "name": col,
                    "type": types.get(col, "?"),
                    "description": desc,
                    "is_primary_key": col in info.primary_key,
                }
                for col, desc in info.columns.items()
            ],
            "sample_rows": samples,
        }

    def get_lineage(self, name: str) -> dict:
        """Upstream (where it comes from) and downstream (what depends on it) for a table."""
        try:
            info = self._resolve(name)
        except KeyError as exc:
            return {"error": str(exc.args[0])}
        return self.catalog.lineage(info.name)

    def run_sql(self, sql: str) -> dict:
        """Validate with guardrails, then execute read-only with a row cap and timeout."""
        try:
            checked = validate_sql(sql, self.catalog, self.max_rows)
        except GuardrailError as exc:
            return {"ok": False, "stage": "validate", "error": str(exc), "sql": sql}

        started = time.perf_counter()
        con = self._connect()
        timer = threading.Timer(self.timeout_s, con.interrupt)
        timer.start()
        try:
            cur = con.execute(checked.sql)
            columns = [d[0] for d in cur.description]
            rows = [[to_jsonable(v) for v in row] for row in cur.fetchall()]
        except duckdb.InterruptException:
            return {"ok": False, "stage": "execute", "sql": checked.sql,
                    "error": f"Query exceeded the {self.timeout_s:.0f}s timeout and was cancelled."}
        except duckdb.Error as exc:
            return {"ok": False, "stage": "execute", "sql": checked.sql,
                    "error": f"{type(exc).__name__}: {' '.join(str(exc).split())[:400]}"}
        finally:
            timer.cancel()
            con.close()

        return {
            "ok": True,
            "sql": checked.sql,
            "columns": columns,
            "rows": rows,
            "row_count": len(rows),
            "truncated": checked.limit_applied and len(rows) >= self.max_rows,
            "tables": checked.tables,
            "layers": sorted({t.split(".")[0] for t in checked.tables}, key=LAYERS.index),
            "elapsed_ms": round((time.perf_counter() - started) * 1000, 1),
        }

    def explain_sql(self, sql: str) -> dict:
        """Static anatomy of a query: tables + layers, and joins matched against catalog keys."""
        try:
            tree = sqlglot.parse_one(sql, read="duckdb")
        except sqlglot.errors.ParseError:
            return {"tables": [], "joins": []}
        alias_to_table: dict[str, str] = {}
        tables: list[dict] = []
        for t in tree.find_all(exp.Table):
            info = self.catalog.resolve(f"{t.db}.{t.name}" if t.db else t.name)
            if info is None:
                continue
            alias_to_table[(t.alias_or_name or t.name).lower()] = info.name
            alias_to_table.setdefault(info.short_name, info.name)
            if all(x["name"] != info.name for x in tables):
                tables.append({"name": info.name, "layer": info.layer, "primary_key": list(info.primary_key)})

        joins: list[dict] = []
        for join in tree.find_all(exp.Join):
            right = join.this
            right_name = alias_to_table.get(right.alias_or_name.lower()) if isinstance(right, exp.Table) else None
            using = join.args.get("using") or []
            if using and right_name:
                left_name = next((t["name"] for t in tables if t["name"] != right_name), None)
                for col in using:
                    joins.append(self._describe_join(left_name, col.name, right_name, col.name))
            on = join.args.get("on")
            if on is not None:
                for eq in on.find_all(exp.EQ):
                    left, right_col = eq.left, eq.right
                    if isinstance(left, exp.Column) and isinstance(right_col, exp.Column):
                        joins.append(self._describe_join(
                            alias_to_table.get(left.table.lower()), left.name,
                            alias_to_table.get(right_col.table.lower()), right_col.name))
        return {"tables": tables, "joins": joins}

    def _describe_join(self, lt: str | None, lc: str, rt: str | None, rc: str) -> dict:
        kind = "ad-hoc join"
        for src, fk in self.catalog.fk_edges():
            if (src, fk.references) == (lt, rt) and lc in fk.columns and rc in fk.ref_columns:
                kind = "foreign key -> primary key"
            elif (src, fk.references) == (rt, lt) and rc in fk.columns and lc in fk.ref_columns:
                kind = "primary key <- foreign key"
        return {"left": f"{lt or '?'}.{lc}", "right": f"{rt or '?'}.{rc}", "relationship": kind}


def _shorten(value: Any, limit: int = 60) -> Any:
    if isinstance(value, str) and len(value) > limit:
        return value[: limit - 3] + "..."
    return value


# OpenAI/LiteLLM tool schemas. The agent and MCP server expose exactly these four tools.
TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "list_tables",
            "description": "List lakehouse tables with descriptions, row counts and primary keys. "
                           "Optionally filter to one layer (bronze, silver, gold).",
            "parameters": {
                "type": "object",
                "properties": {"layer": {"type": "string", "enum": list(LAYERS)}},
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "describe_table",
            "description": "Describe one table: columns with types and descriptions, primary key, "
                           "foreign keys, tables that reference it, lineage and sample rows. "
                           "Call this before writing SQL against a table.",
            "parameters": {
                "type": "object",
                "properties": {"name": {"type": "string", "description": "e.g. gold.fact_orders"}},
                "required": ["name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_lineage",
            "description": "Upstream sources and downstream dependents of a table across bronze/silver/gold.",
            "parameters": {
                "type": "object",
                "properties": {"name": {"type": "string"}},
                "required": ["name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_sql",
            "description": "Run ONE read-only DuckDB SELECT (or WITH ... SELECT) over catalog tables. "
                           "Use schema-qualified names (gold.fact_orders). Results are capped at 200 rows.",
            "parameters": {
                "type": "object",
                "properties": {"sql": {"type": "string"}},
                "required": ["sql"],
            },
        },
    },
]
