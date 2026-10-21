"""Discovery tools: list/describe/lineage return correct keys and graph closures."""

from __future__ import annotations


def test_list_tables_by_layer(tools):
    gold = tools.list_tables("gold")
    assert {t["layer"] for t in gold["tables"]} == {"gold"}
    assert len(gold["tables"]) == 7
    assert "error" in tools.list_tables("platinum")
    assert len(tools.list_tables()["tables"]) == len(tools.catalog.tables)


def test_describe_table_keys_and_samples(tools):
    d = tools.describe_table("fact_order_items")  # bare name resolves to gold
    assert d["name"] == "gold.fact_order_items"
    assert d["primary_key"] == ["order_id", "order_item_id"]
    assert {fk["references"] for fk in d["foreign_keys"]} == {
        "gold.fact_orders", "gold.dim_products", "gold.dim_sellers", "gold.dim_customers"}
    assert len(d["sample_rows"]) == 3
    types = {c["name"]: c["type"] for c in d["columns"]}
    assert types["price"].startswith("DECIMAL")


def test_describe_reports_incoming_references(tools):
    d = tools.describe_table("gold.dim_customers")
    assert {r["table"] for r in d["referenced_by"]} == {"gold.fact_orders", "gold.fact_order_items"}


def test_describe_unknown_and_ambiguous(tools):
    assert "error" in tools.describe_table("nope")
    err = tools.describe_table("orders")["error"]
    assert "bronze.orders" in err and "silver.orders" in err


def test_lineage_upstream_closure(tools):
    lin = tools.get_lineage("gold.agg_daily_sales")
    assert lin["direct_upstream"] == ["gold.fact_orders", "gold.dim_customers"]
    assert {"silver.orders", "bronze.orders", "bronze.customers"} <= set(lin["all_upstream"])
    assert "olist_orders_dataset.csv" in lin["raw_sources"]
    assert lin["all_downstream"] == []


def test_lineage_downstream_closure(tools):
    lin = tools.get_lineage("bronze.order_reviews")
    assert lin["direct_downstream"] == ["silver.order_reviews"]
    assert {"gold.fact_orders", "gold.agg_daily_sales", "gold.agg_seller_performance"} <= set(lin["all_downstream"])
    assert {"from": "bronze.order_reviews", "to": "silver.order_reviews"} in lin["edges"]


def test_explain_sql_matches_foreign_keys(tools):
    info = tools.explain_sql(
        "SELECT c.state, count(*) FROM gold.fact_orders f "
        "JOIN gold.dim_customers c ON f.customer_id = c.customer_id GROUP BY 1")
    assert [t["name"] for t in info["tables"]] == ["gold.fact_orders", "gold.dim_customers"]
    assert info["joins"][0]["relationship"] == "foreign key -> primary key"
