"""Build the medallion lakehouse (bronze -> silver -> gold) as a single DuckDB file.

* bronze: raw CSVs loaded as-is (every column VARCHAR) + `_ingested_at`, `_source_file`.
* silver: typed, trimmed, deduplicated on the business key, standardised column names,
          Portuguese category names translated to English.
* gold:   star schema (facts + dimensions) and pre-aggregated marts for analytics.

Usage:  python -m lakehouse_agent.build_lakehouse [--raw-dir DIR] [--db PATH]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import duckdb

from lakehouse_agent.config import DB_PATH, RAW_DIR

# bronze table name -> source CSV file name
BRONZE_SOURCES: dict[str, str] = {
    "customers": "olist_customers_dataset.csv",
    "orders": "olist_orders_dataset.csv",
    "order_items": "olist_order_items_dataset.csv",
    "order_payments": "olist_order_payments_dataset.csv",
    "order_reviews": "olist_order_reviews_dataset.csv",
    "products": "olist_products_dataset.csv",
    "sellers": "olist_sellers_dataset.csv",
    "category_translation": "product_category_name_translation.csv",
}

# Two categories in the source have no English translation; we add them here.
_EXTRA_CATEGORY_TRANSLATIONS = {
    "pc_gamer": "pc_gamer",
    "portateis_cozinha_e_preparadores_de_alimentos": "portable_kitchen_food_preparers",
}

# Order statuses that never became a sale. Gold marts exclude them from revenue.
NON_SALE_STATUSES = ("canceled", "unavailable")
_NON_SALE_SQL = ", ".join(f"'{s}'" for s in NON_SALE_STATUSES)

_extra_values = ", ".join(f"('{pt}', '{en}')" for pt, en in _EXTRA_CATEGORY_TRANSLATIONS.items())

SILVER_SQL: dict[str, str] = {
    "customers": """
        SELECT trim(customer_id)                      AS customer_id,
               trim(customer_unique_id)               AS customer_unique_id,
               lpad(trim(customer_zip_code_prefix), 5, '0') AS zip_code_prefix,
               lower(trim(customer_city))             AS city,
               upper(trim(customer_state))            AS state
        FROM bronze.customers
        WHERE customer_id IS NOT NULL
        QUALIFY row_number() OVER (PARTITION BY trim(customer_id) ORDER BY _ingested_at DESC) = 1
    """,
    "sellers": """
        SELECT trim(seller_id)                        AS seller_id,
               lpad(trim(seller_zip_code_prefix), 5, '0') AS zip_code_prefix,
               lower(trim(seller_city))               AS city,
               upper(trim(seller_state))              AS state
        FROM bronze.sellers
        WHERE seller_id IS NOT NULL
        QUALIFY row_number() OVER (PARTITION BY trim(seller_id) ORDER BY _ingested_at DESC) = 1
    """,
    "product_categories": f"""
        WITH unioned AS (
            SELECT trim(product_category_name) AS category_name_pt,
                   trim(product_category_name_english) AS category_name
            FROM bronze.category_translation
            WHERE product_category_name IS NOT NULL
            UNION ALL
            SELECT * FROM (VALUES {_extra_values}) AS t(category_name_pt, category_name)
        )
        SELECT category_name_pt, category_name
        FROM unioned
        QUALIFY row_number() OVER (PARTITION BY category_name_pt ORDER BY category_name) = 1
    """,
    "products": """
        SELECT trim(p.product_id)                               AS product_id,
               coalesce(c.category_name, trim(p.product_category_name), 'unknown') AS category_name,
               coalesce(trim(p.product_category_name), 'unknown') AS category_name_pt,
               TRY_CAST(p.product_name_lenght AS INTEGER)        AS name_length,
               TRY_CAST(p.product_description_lenght AS INTEGER) AS description_length,
               TRY_CAST(p.product_photos_qty AS INTEGER)         AS photos_qty,
               TRY_CAST(p.product_weight_g AS INTEGER)           AS weight_g,
               TRY_CAST(p.product_length_cm AS INTEGER)          AS length_cm,
               TRY_CAST(p.product_height_cm AS INTEGER)          AS height_cm,
               TRY_CAST(p.product_width_cm AS INTEGER)           AS width_cm
        FROM bronze.products p
        LEFT JOIN silver.product_categories c
               ON c.category_name_pt = trim(p.product_category_name)
        WHERE p.product_id IS NOT NULL
        QUALIFY row_number() OVER (PARTITION BY trim(p.product_id) ORDER BY p._ingested_at DESC) = 1
    """,
    "orders": """
        SELECT trim(order_id)                                       AS order_id,
               trim(customer_id)                                    AS customer_id,
               lower(trim(order_status))                            AS order_status,
               TRY_CAST(order_purchase_timestamp AS TIMESTAMP)      AS purchased_at,
               TRY_CAST(order_approved_at AS TIMESTAMP)             AS approved_at,
               TRY_CAST(order_delivered_carrier_date AS TIMESTAMP)  AS delivered_carrier_at,
               TRY_CAST(order_delivered_customer_date AS TIMESTAMP) AS delivered_customer_at,
               TRY_CAST(order_estimated_delivery_date AS TIMESTAMP) AS estimated_delivery_at
        FROM bronze.orders
        WHERE order_id IS NOT NULL
        QUALIFY row_number() OVER (PARTITION BY trim(order_id) ORDER BY _ingested_at DESC) = 1
    """,
    "order_items": """
        SELECT trim(order_id)                               AS order_id,
               TRY_CAST(order_item_id AS INTEGER)           AS order_item_id,
               trim(product_id)                             AS product_id,
               trim(seller_id)                              AS seller_id,
               TRY_CAST(shipping_limit_date AS TIMESTAMP)   AS shipping_limit_at,
               TRY_CAST(price AS DECIMAL(12, 2))            AS price,
               TRY_CAST(freight_value AS DECIMAL(12, 2))    AS freight_value
        FROM bronze.order_items
        WHERE order_id IS NOT NULL
        QUALIFY row_number() OVER (
            PARTITION BY trim(order_id), TRY_CAST(order_item_id AS INTEGER) ORDER BY _ingested_at DESC) = 1
    """,
    "order_payments": """
        SELECT trim(order_id)                               AS order_id,
               TRY_CAST(payment_sequential AS INTEGER)      AS payment_sequential,
               lower(trim(payment_type))                    AS payment_type,
               TRY_CAST(payment_installments AS INTEGER)    AS installments,
               TRY_CAST(payment_value AS DECIMAL(12, 2))    AS payment_value
        FROM bronze.order_payments
        WHERE order_id IS NOT NULL
        QUALIFY row_number() OVER (
            PARTITION BY trim(order_id), TRY_CAST(payment_sequential AS INTEGER) ORDER BY _ingested_at DESC) = 1
    """,
    "order_reviews": """
        SELECT trim(review_id)                                  AS review_id,
               trim(order_id)                                   AS order_id,
               TRY_CAST(review_score AS INTEGER)                AS review_score,
               nullif(trim(review_comment_title), '')           AS comment_title,
               nullif(trim(review_comment_message), '')         AS comment_message,
               TRY_CAST(review_creation_date AS TIMESTAMP)      AS created_at,
               TRY_CAST(review_answer_timestamp AS TIMESTAMP)   AS answered_at
        FROM bronze.order_reviews
        WHERE review_id IS NOT NULL AND order_id IS NOT NULL
        QUALIFY row_number() OVER (
            PARTITION BY trim(review_id), trim(order_id)
            ORDER BY TRY_CAST(review_answer_timestamp AS TIMESTAMP) DESC NULLS LAST) = 1
    """,
}

GOLD_SQL: dict[str, str] = {
    "dim_customers": """
        WITH order_counts AS (
            SELECT c.customer_unique_id, count(*) AS n_orders
            FROM silver.orders o JOIN silver.customers c USING (customer_id)
            GROUP BY 1
        )
        SELECT c.customer_id,
               c.customer_unique_id,
               c.zip_code_prefix,
               c.city,
               c.state,
               coalesce(oc.n_orders, 0)        AS lifetime_orders,
               coalesce(oc.n_orders, 0) > 1    AS is_repeat_customer
        FROM silver.customers c
        LEFT JOIN order_counts oc USING (customer_unique_id)
    """,
    "dim_products": """
        SELECT product_id, category_name, category_name_pt, name_length, description_length,
               photos_qty, weight_g, length_cm, height_cm, width_cm,
               length_cm * height_cm * width_cm AS volume_cm3
        FROM silver.products
    """,
    "dim_sellers": """
        SELECT seller_id, zip_code_prefix, city, state FROM silver.sellers
    """,
    "fact_orders": """
        WITH items AS (
            SELECT order_id, count(*) AS item_count, sum(price) AS items_value,
                   sum(freight_value) AS freight_value, count(DISTINCT seller_id) AS seller_count
            FROM silver.order_items GROUP BY 1
        ),
        pay AS (
            SELECT order_id, sum(payment_value) AS payment_value,
                   arg_max(payment_type, payment_value) AS primary_payment_type,
                   max(installments) AS installments
            FROM silver.order_payments GROUP BY 1
        ),
        rev AS (
            SELECT order_id, avg(review_score)::DOUBLE AS review_score
            FROM silver.order_reviews GROUP BY 1
        )
        SELECT o.order_id,
               o.customer_id,
               o.order_status,
               o.purchased_at,
               CAST(o.purchased_at AS DATE)                         AS order_date,
               CAST(date_trunc('month', o.purchased_at) AS DATE)    AS order_month,
               o.approved_at,
               o.delivered_customer_at                              AS delivered_at,
               o.estimated_delivery_at,
               round((epoch(o.delivered_customer_at) - epoch(o.purchased_at)) / 86400.0, 2) AS delivery_days,
               CASE WHEN o.delivered_customer_at IS NULL THEN NULL
                    ELSE CAST(o.delivered_customer_at AS DATE) > CAST(o.estimated_delivery_at AS DATE)
               END                                                  AS is_late,
               coalesce(i.item_count, 0)                            AS item_count,
               coalesce(i.seller_count, 0)                          AS seller_count,
               coalesce(i.items_value, 0)                           AS items_value,
               coalesce(i.freight_value, 0)                         AS freight_value,
               coalesce(i.items_value, 0) + coalesce(i.freight_value, 0) AS order_value,
               p.payment_value,
               p.primary_payment_type,
               p.installments,
               r.review_score
        FROM silver.orders o
        LEFT JOIN items i USING (order_id)
        LEFT JOIN pay p USING (order_id)
        LEFT JOIN rev r USING (order_id)
    """,
    "fact_order_items": """
        SELECT oi.order_id,
               oi.order_item_id,
               oi.product_id,
               oi.seller_id,
               o.customer_id,
               o.order_status,
               CAST(o.purchased_at AS DATE)  AS order_date,
               p.category_name,
               oi.price,
               oi.freight_value,
               oi.price + oi.freight_value   AS total_value,
               oi.shipping_limit_at
        FROM silver.order_items oi
        JOIN silver.orders o USING (order_id)
        LEFT JOIN silver.products p USING (product_id)
    """,
    "agg_daily_sales": f"""
        SELECT f.order_date,
               count(*)                                   AS orders,
               CAST(sum(f.item_count) AS BIGINT)          AS items,
               sum(f.items_value)                         AS revenue,
               sum(f.freight_value)                       AS freight,
               sum(f.order_value)                         AS gmv,
               round(sum(f.order_value) / count(*), 2)    AS avg_order_value,
               count(DISTINCT c.customer_unique_id)       AS unique_customers
        FROM gold.fact_orders f
        JOIN gold.dim_customers c USING (customer_id)
        WHERE f.order_status NOT IN ({_NON_SALE_SQL})
        GROUP BY f.order_date
    """,
    "agg_seller_performance": f"""
        WITH items AS (
            SELECT i.seller_id, i.order_id, i.price, i.freight_value, i.order_date
            FROM gold.fact_order_items i
            WHERE i.order_status NOT IN ({_NON_SALE_SQL})
        ),
        seller_orders AS (
            SELECT DISTINCT seller_id, order_id FROM items
        )
        SELECT s.seller_id,
               s.state                                         AS seller_state,
               count(DISTINCT it.order_id)                     AS orders,
               count(*)                                        AS items_sold,
               sum(it.price)                                   AS revenue,
               sum(it.freight_value)                           AS freight,
               round(avg(it.price), 2)                         AS avg_item_price,
               min(it.order_date)                              AS first_sale_date,
               max(it.order_date)                              AS last_sale_date,
               (SELECT round(avg(f.review_score), 3) FROM seller_orders so
                  JOIN gold.fact_orders f USING (order_id) WHERE so.seller_id = s.seller_id) AS avg_review_score,
               (SELECT round(avg(f.delivery_days), 2) FROM seller_orders so
                  JOIN gold.fact_orders f USING (order_id) WHERE so.seller_id = s.seller_id) AS avg_delivery_days,
               (SELECT round(avg(CASE WHEN f.is_late THEN 1.0 ELSE 0.0 END), 4) FROM seller_orders so
                  JOIN gold.fact_orders f USING (order_id)
                  WHERE so.seller_id = s.seller_id AND f.is_late IS NOT NULL) AS late_delivery_rate
        FROM items it
        JOIN gold.dim_sellers s USING (seller_id)
        GROUP BY s.seller_id, s.state
    """,
}


def _sql_str(value: str) -> str:
    """Quote a Python string as a SQL string literal."""
    return "'" + value.replace("'", "''") + "'"


def build(raw_dir: Path = RAW_DIR, db_path: Path = DB_PATH, quiet: bool = False) -> dict:
    """Build the lakehouse from CSVs in `raw_dir` into `db_path`. Returns the build manifest."""
    raw_dir, db_path = Path(raw_dir), Path(db_path)
    missing = [f for f in BRONZE_SOURCES.values() if not (raw_dir / f).exists()]
    if missing:
        raise FileNotFoundError(f"Missing source files in {raw_dir}: {missing}. Run `make data` first.")

    db_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = db_path.with_suffix(".building.duckdb")
    tmp_path.unlink(missing_ok=True)
    started = time.perf_counter()
    ingested_at = datetime.now(timezone.utc).replace(microsecond=0)
    counts: dict[str, int] = {}

    def log(msg: str) -> None:
        if not quiet:
            print(msg)

    con = duckdb.connect(str(tmp_path))
    try:
        for schema in ("bronze", "silver", "gold"):
            con.execute(f"CREATE SCHEMA {schema}")

        for table, filename in BRONZE_SOURCES.items():
            path = (raw_dir / filename).resolve()
            con.execute(
                f"""CREATE TABLE bronze.{table} AS
                    SELECT *,
                           CAST({_sql_str(ingested_at.isoformat())} AS TIMESTAMPTZ) AS _ingested_at,
                           {_sql_str(filename)} AS _source_file
                    FROM read_csv({_sql_str(str(path))}, header = true, all_varchar = true)"""
            )
        for layer, statements in (("silver", SILVER_SQL), ("gold", GOLD_SQL)):
            for table, sql in statements.items():
                con.execute(f"CREATE TABLE {layer}.{table} AS {sql}")

        rows = con.execute(
            "SELECT table_schema, table_name FROM information_schema.tables "
            "WHERE table_schema IN ('bronze','silver','gold') ORDER BY 1, 2"
        ).fetchall()
        for schema, table in rows:
            counts[f"{schema}.{table}"] = con.execute(f"SELECT count(*) FROM {schema}.{table}").fetchone()[0]
        con.execute("CHECKPOINT")
    finally:
        con.close()

    db_path.unlink(missing_ok=True)
    tmp_path.rename(db_path)

    manifest = {
        "built_at": ingested_at.isoformat(),
        "db_path": str(db_path),
        "raw_dir": str(raw_dir),
        "build_seconds": round(time.perf_counter() - started, 2),
        "row_counts": counts,
    }
    db_path.with_name("build_manifest.json").write_text(json.dumps(manifest, indent=2))

    log(f"Built {db_path} in {manifest['build_seconds']}s")
    log(f"{'table':<36}{'rows':>12}")
    for name, n in counts.items():
        log(f"{name:<36}{n:>12,}")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the bronze/silver/gold DuckDB lakehouse.")
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--db", type=Path, default=DB_PATH)
    args = parser.parse_args()
    try:
        build(args.raw_dir, args.db)
    except FileNotFoundError as exc:
        print(exc, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
