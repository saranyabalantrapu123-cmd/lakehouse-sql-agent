"""SQL guardrails block anything that is not a single read-only SELECT over catalog tables."""

from __future__ import annotations

import duckdb
import pytest

from lakehouse_agent.guardrails import GuardrailError, validate_sql

BLOCKED = [
    ("DROP TABLE gold.fact_orders", "DROP"),
    ("INSERT INTO gold.dim_sellers VALUES ('x', '1', 'a', 'SP')", "INSERT"),
    ("UPDATE gold.dim_sellers SET state = 'XX'", "UPDATE"),
    ("DELETE FROM gold.dim_sellers", "DELETE"),
    ("CREATE TABLE gold.x AS SELECT 1", "CREATE"),
    ("ATTACH 'other.duckdb' AS other", "ATTACH"),
    ("COPY gold.fact_orders TO 'out.csv'", "COPY"),
    ("PRAGMA table_info('gold.fact_orders')", "PRAGMA"),
    ("SET threads = 1", "SET"),
    ("INSTALL httpfs", "INSTALL"),
    ("SELECT * FROM read_csv('data/raw/olist_orders_dataset.csv')", "read_csv"),
    ("SELECT * FROM read_csv_auto('/etc/passwd')", "read_csv_auto"),
    ("SELECT * FROM read_parquet('x.parquet')", "read_parquet"),
    ("SELECT * FROM 'data/raw/olist_orders_dataset.csv'", "Unknown"),
    ("SELECT * FROM glob('*')", "glob"),
    ("SELECT getenv('HOME') FROM gold.dim_sellers", "getenv"),
    ("SELECT * FROM duckdb_settings()", "duckdb_settings"),
    ("SELECT 1; DROP TABLE gold.fact_orders", "one statement"),
    ("SELECT * FROM gold.fact_orders; SELECT * FROM gold.dim_sellers", "one statement"),
    ("SELECT * FROM gold.secret_table", "Unknown"),
    ("SELECT * FROM information_schema.tables", "Unknown"),
    ("SELECT * FROM orders", "ambiguous"),
    ("SELECT * FROM other.gold.fact_orders", "Cross-database"),
    ("SELECT 1", "at least one catalog table"),
    ("", "Empty"),
]


@pytest.mark.parametrize("sql,needle", BLOCKED)
def test_blocked(sql, needle, catalog):
    with pytest.raises(GuardrailError) as exc:
        validate_sql(sql, catalog)
    assert needle.lower() in str(exc.value).lower()


def test_allows_select_with_cte_and_join(catalog):
    sql = """
        WITH s AS (SELECT seller_id, revenue FROM gold.agg_seller_performance)
        SELECT d.state, sum(s.revenue) FROM s JOIN gold.dim_sellers d ON s.seller_id = d.seller_id GROUP BY 1
    """
    checked = validate_sql(sql, catalog)
    assert set(checked.tables) == {"gold.agg_seller_performance", "gold.dim_sellers"}
    assert "LIMIT 200" in checked.sql


def test_bare_names_are_qualified(catalog):
    checked = validate_sql("SELECT count(*) FROM fact_orders", catalog)
    assert "gold.fact_orders" in checked.sql


def test_limit_is_capped_not_raised(catalog):
    assert "LIMIT 200" in validate_sql("SELECT * FROM gold.fact_orders LIMIT 100000", catalog).sql
    assert "LIMIT 5" in validate_sql("SELECT * FROM gold.fact_orders LIMIT 5", catalog).sql


def test_run_sql_blocks_and_data_survives(tools):
    res = tools.run_sql("DROP TABLE gold.fact_orders")
    assert res["ok"] is False and res["stage"] == "validate"
    assert tools.run_sql("SELECT count(*) AS n FROM gold.fact_orders")["rows"] == [[4]]


def test_run_sql_reports_execution_errors(tools):
    res = tools.run_sql("SELECT no_such_column FROM gold.fact_orders")
    assert res["ok"] is False and res["stage"] == "execute"
    assert "no_such_column" in res["error"]


def test_connection_is_read_only_and_sandboxed(tools):
    """Second line of defence: even SQL that bypassed the validator cannot write or read files."""
    with tools._connect() as con:
        with pytest.raises(duckdb.Error):
            con.execute("CREATE TABLE gold.evil AS SELECT 1")
        with pytest.raises(duckdb.Error):
            con.execute("SELECT * FROM read_csv('/etc/hostname')")


def test_timeout_interrupts_long_queries(lakehouse, catalog):
    from lakehouse_agent.tools import LakehouseTools

    slow = LakehouseTools(lakehouse[0], catalog, timeout_s=0.2)
    res = slow.run_sql(
        "SELECT count(*) FROM gold.fact_orders a, gold.fact_orders b, gold.fact_orders c, "
        "gold.fact_order_items d, gold.fact_order_items e, gold.dim_customers f, gold.dim_customers g, "
        "gold.dim_products h, gold.dim_products i, gold.agg_daily_sales j, gold.agg_daily_sales k, "
        "bronze.order_reviews l, bronze.order_reviews m, bronze.customers n, bronze.customers o, "
        "bronze.order_items p, bronze.order_items q, silver.order_payments r, silver.order_payments s "
        "WHERE md5(a.order_id || b.order_id || c.order_id || d.order_id || e.order_id) <> ''"
    )
    assert res["ok"] is False and "timeout" in res["error"].lower()
