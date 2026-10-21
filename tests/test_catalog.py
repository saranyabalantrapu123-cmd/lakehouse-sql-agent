"""The catalog must exactly describe the built lakehouse, and quality checks must pass on it."""

from __future__ import annotations

import duckdb

from lakehouse_agent.catalog import LAYERS
from lakehouse_agent.quality import run_checks


def _db_schema(db_path) -> dict[str, list[str]]:
    with duckdb.connect(str(db_path), read_only=True) as con:
        rows = con.execute(
            "SELECT table_schema || '.' || table_name, column_name FROM information_schema.columns "
            "WHERE table_schema IN ('bronze','silver','gold') ORDER BY table_schema, table_name, ordinal_position"
        ).fetchall()
    out: dict[str, list[str]] = {}
    for table, col in rows:
        out.setdefault(table, []).append(col)
    return out


def test_every_table_and_column_documented(lakehouse, catalog):
    db = _db_schema(lakehouse[0])
    assert set(db) == set(catalog.tables), "catalog tables differ from built tables"
    for name, cols in db.items():
        assert set(cols) == set(catalog.tables[name].columns), f"column mismatch in {name}"
        assert all(desc.strip() for desc in catalog.tables[name].columns.values()), f"empty description in {name}"


def test_required_gold_tables_present(catalog):
    expected = {"fact_orders", "fact_order_items", "dim_customers", "dim_products", "dim_sellers",
                "agg_daily_sales", "agg_seller_performance"}
    assert {t.short_name for t in catalog.by_layer("gold")} == expected


def test_keys_reference_real_columns(catalog):
    for t in catalog.tables.values():
        assert t.layer in LAYERS
        assert t.description
        for col in [*t.primary_key, *t.not_null, *t.max_null_rate]:
            assert col in t.columns, f"{t.name}.{col} not a column"
        for fk in t.foreign_keys:
            target = catalog.tables[fk.references]
            assert all(c in t.columns for c in fk.columns)
            assert tuple(fk.ref_columns) == target.primary_key, f"{t.name} FK must point at {target.name} PK"


def test_lineage_respects_layer_order(catalog):
    rank = {layer: i for i, layer in enumerate(LAYERS)}
    for t in catalog.tables.values():
        if t.layer == "bronze":
            assert t.source and not t.upstream
        else:
            assert t.upstream, f"{t.name} has no lineage"
        for up in t.upstream:
            assert up in catalog.tables
            assert rank[catalog.tables[up].layer] <= rank[t.layer]


def test_quality_checks_pass_on_fixture(lakehouse, catalog):
    results, null_rates = run_checks(lakehouse[0], catalog)
    failures = [r for r in results if r.status == "FAIL"]
    assert not failures, failures
    kinds = {r.check for r in results}
    assert {"pk_unique", "not_null", "fk_integrity"} <= kinds
    # p3 has no category in the source -> known 'warn' FK gap is reported, not hidden.
    assert any(r.status == "WARN" and r.table == "silver.products" for r in results)
    assert null_rates["gold.fact_orders"]["delivered_at"] > 0
