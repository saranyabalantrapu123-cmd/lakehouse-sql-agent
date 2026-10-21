"""SQL guardrails: only a single read-only SELECT over catalog tables may reach DuckDB.

Defence in depth:
1. Parse with sqlglot (DuckDB dialect). Exactly one statement; root must be a query
   (SELECT / WITH ... SELECT / UNION / INTERSECT / EXCEPT).
2. Reject any DDL/DML/admin node anywhere in the tree (INSERT, DROP, ATTACH, COPY, PRAGMA, SET, ...).
3. Reject file / network / system functions (read_csv, glob, getenv, duckdb_*, ...) and any
   table-valued function in FROM.
4. Every table reference must be a catalog table. Bare names are qualified to their layer.
5. Enforce a row cap by adding or tightening the outer LIMIT.
The executor then runs on a read-only connection with external access disabled and a timeout.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import sqlglot
from sqlglot import exp
from sqlglot.errors import ParseError

from lakehouse_agent.catalog import Catalog
from lakehouse_agent.config import MAX_ROWS

# sqlglot logs a warning when it falls back to a generic Command; we reject those anyway.
logging.getLogger("sqlglot").setLevel(logging.ERROR)


class GuardrailError(ValueError):
    """Raised when a SQL statement is not allowed. The message is safe to show the LLM."""


# Statement/expression types that must never appear anywhere in the tree.
_FORBIDDEN_NODES: tuple[type[exp.Expression], ...] = tuple(
    getattr(exp, name)
    for name in (
        "Insert", "Update", "Delete", "Merge", "Create", "Drop", "Alter", "AlterColumn",
        "TruncateTable", "Command", "Pragma", "Attach", "Detach", "Copy", "Set", "Use",
        "Install", "Load", "Describe", "Summarize", "Grant", "Revoke", "Transaction",
        "Commit", "Rollback", "Export", "Into", "LoadData", "Analyze",
    )
    if hasattr(exp, name)
)

_FORBIDDEN_FUNC_PREFIXES = ("read_", "duckdb_", "pragma_", "parquet_", "sqlite_", "postgres_", "iceberg_", "delta_")
_FORBIDDEN_FUNCS = {
    "glob", "getenv", "query", "query_table", "sniff_csv", "csv_scan", "json_scan", "parquet_scan",
    "current_setting", "set_variable", "getvariable", "load", "install", "checkpoint", "force_checkpoint",
    "which_secret", "httpfs", "write_csv", "write_parquet", "system", "shell", "import_database", "export_database",
}


@dataclass
class ValidatedSQL:
    sql: str  # the rewritten SQL that will be executed
    tables: list[str] = field(default_factory=list)  # fully-qualified catalog tables referenced
    limit_applied: bool = False


def _func_name(node: exp.Func) -> str:
    if isinstance(node, exp.Anonymous):
        return str(node.name).lower()
    return node.sql_name().lower()


def validate_sql(sql: str, catalog: Catalog, max_rows: int = MAX_ROWS) -> ValidatedSQL:
    """Validate and rewrite `sql`. Raises GuardrailError with an actionable message."""
    if not sql or not sql.strip():
        raise GuardrailError("Empty SQL.")
    try:
        statements = [s for s in sqlglot.parse(sql, read="duckdb") if s is not None]
    except ParseError as exc:
        raise GuardrailError(f"SQL could not be parsed: {str(exc).splitlines()[0]}") from exc
    if len(statements) != 1:
        raise GuardrailError(f"Exactly one statement is allowed (got {len(statements)}).")
    tree = statements[0]

    if not isinstance(tree, exp.Query):
        raise GuardrailError(
            f"Only read-only SELECT / WITH queries are allowed (got {tree.key.upper()})."
        )
    for node in tree.walk():
        if isinstance(node, _FORBIDDEN_NODES):
            raise GuardrailError(f"Statement type {node.key.upper()} is not allowed; queries must be read-only.")
        if isinstance(node, exp.Func):
            name = _func_name(node)
            if name in _FORBIDDEN_FUNCS or name.startswith(_FORBIDDEN_FUNC_PREFIXES):
                raise GuardrailError(f"Function {name}() is not allowed (file/system access is blocked).")

    cte_names = {cte.alias_or_name.lower() for cte in tree.find_all(exp.CTE)}
    referenced: list[str] = []
    for table in tree.find_all(exp.Table):
        if not isinstance(table.this, exp.Identifier):
            raise GuardrailError("Table functions are not allowed in FROM; query catalog tables only.")
        if table.args.get("catalog"):
            raise GuardrailError(f"Cross-database references are not allowed: {table.sql('duckdb')}.")
        name = table.name.lower()
        schema = (table.db or "").lower()
        if not schema and name in cte_names:
            continue
        qualified = f"{schema}.{name}" if schema else name
        info = catalog.resolve(qualified)
        if info is None:
            options = catalog.candidates(name)
            hint = f" Did you mean one of: {', '.join(options)}?" if options else " Use list_tables to see available tables."
            raise GuardrailError(f"Unknown or ambiguous table '{qualified}'.{hint}")
        if not schema:  # qualify bare names so DuckDB can find them
            table.set("db", exp.to_identifier(info.layer))
        if info.name not in referenced:
            referenced.append(info.name)
    if not referenced:
        raise GuardrailError("Query must read from at least one catalog table.")

    limit_applied = False
    limit_node = tree.args.get("limit")
    if limit_node is None:
        tree = tree.limit(max_rows)
        limit_applied = True
    else:
        value = limit_node.expression if isinstance(limit_node, exp.Limit) else None
        if not (isinstance(value, exp.Literal) and value.is_int):
            tree = exp.select("*").from_(tree.subquery("capped")).limit(max_rows)
            limit_applied = True
        elif int(value.this) > max_rows:
            limit_node.set("expression", exp.Literal.number(max_rows))
            limit_applied = True

    return ValidatedSQL(sql=tree.sql(dialect="duckdb", pretty=True), tables=referenced, limit_applied=limit_applied)
