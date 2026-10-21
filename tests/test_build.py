"""Lakehouse build: bronze keeps raw rows, silver cleans/dedupes, gold computes correct measures."""

from __future__ import annotations

import json

import duckdb
import pytest

from lakehouse_agent.build_lakehouse import build


@pytest.fixture(scope="module")
def con(lakehouse):
    db_path, _ = lakehouse
    c = duckdb.connect(str(db_path), read_only=True)
    yield c
    c.close()


def scalar(con, sql):
    return con.execute(sql).fetchone()[0]


def test_row_counts_recorded(lakehouse):
    db_path, manifest = lakehouse
    counts = manifest["row_counts"]
    assert counts["bronze.customers"] == 5        # raw duplicate kept
    assert counts["silver.customers"] == 4        # deduplicated
    assert counts["bronze.order_reviews"] == 5    # multi-line quoted comment parsed as one row
    assert counts["silver.order_reviews"] == 4    # exact duplicate review removed
    assert counts["gold.fact_orders"] == 4
    assert counts["gold.fact_order_items"] == 5
    assert counts["gold.agg_daily_sales"] == 3    # canceled-only day excluded
    on_disk = json.loads(db_path.with_name("build_manifest.json").read_text())
    assert on_disk["row_counts"] == counts


def test_bronze_is_raw_with_audit_columns(con):
    types = dict(con.execute(
        "SELECT column_name, data_type FROM information_schema.columns "
        "WHERE table_schema='bronze' AND table_name='order_items'").fetchall())
    assert types["price"] == "VARCHAR"
    assert "_ingested_at" in types and "_source_file" in types
    assert scalar(con, "SELECT DISTINCT _source_file FROM bronze.order_items") == "olist_order_items_dataset.csv"


def test_silver_typing_and_cleaning(con):
    assert scalar(con, "SELECT typeof(price) FROM silver.order_items LIMIT 1").startswith("DECIMAL")
    assert scalar(con, "SELECT typeof(purchased_at) FROM silver.orders LIMIT 1") == "TIMESTAMP"
    assert scalar(con, "SELECT city || '|' || state FROM silver.customers WHERE customer_id='c4'") == "belo horizonte|MG"
    assert scalar(con, "SELECT zip_code_prefix FROM silver.customers WHERE customer_id='c1'") == "01310"


def test_category_translation(con):
    rows = dict(con.execute("SELECT product_id, category_name FROM silver.products").fetchall())
    assert rows == {"p1": "health_beauty", "p2": "sports_leisure", "p3": "unknown", "p4": "pc_gamer"}


def test_gold_measures(con):
    o1 = con.execute(
        "SELECT item_count, items_value, freight_value, order_value, is_late, review_score "
        "FROM gold.fact_orders WHERE order_id='o1'").fetchone()
    assert o1[0] == 2 and float(o1[1]) == 150.0 and float(o1[2]) == 15.0 and float(o1[3]) == 165.0
    assert o1[4] is False and o1[5] == 5.0
    assert scalar(con, "SELECT is_late FROM gold.fact_orders WHERE order_id='o2'") is True
    assert scalar(con, "SELECT is_late FROM gold.fact_orders WHERE order_id='o4'") is None
    assert scalar(con, "SELECT primary_payment_type FROM gold.fact_orders WHERE order_id='o4'") == "credit_card"
    assert float(scalar(con, "SELECT sum(revenue) FROM gold.agg_daily_sales")) == 430.0


def test_gold_repeat_customers(con):
    repeat = {r[0] for r in con.execute("SELECT customer_id FROM gold.dim_customers WHERE is_repeat_customer").fetchall()}
    assert repeat == {"c1", "c3"}


def test_seller_performance_excludes_canceled(con):
    rows = {r[0]: (r[1], float(r[2])) for r in con.execute(
        "SELECT seller_id, orders, revenue FROM gold.agg_seller_performance").fetchall()}
    assert rows == {"s1": (3, 380.0), "s2": (1, 50.0)}


def test_missing_source_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        build(tmp_path, tmp_path / "x.duckdb", quiet=True)
