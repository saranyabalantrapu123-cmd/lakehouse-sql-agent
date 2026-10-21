"""Catalog-driven data-quality checks: PK uniqueness/not-null, FK integrity, null rates.

Usage:  python -m lakehouse_agent.quality [--db PATH]
Exit code is 1 if any check with severity "error" fails, so it can gate a pipeline.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import duckdb

from lakehouse_agent.catalog import Catalog, load_catalog
from lakehouse_agent.config import CATALOG_PATH, DB_PATH


@dataclass
class CheckResult:
    table: str
    check: str
    detail: str
    failing_rows: int
    severity: str = "error"

    @property
    def status(self) -> str:
        if self.failing_rows == 0:
            return "PASS"
        return "WARN" if self.severity == "warn" else "FAIL"


def run_checks(db_path: Path | str = DB_PATH, catalog: Catalog | None = None) -> tuple[list[CheckResult], dict]:
    """Run all checks. Returns (results, null_rates) where null_rates[table][column] = rate."""
    catalog = catalog or load_catalog(CATALOG_PATH)
    results: list[CheckResult] = []
    null_rates: dict[str, dict[str, float]] = {}
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        for t in catalog.tables.values():
            total = con.execute(f"SELECT count(*) FROM {t.name}").fetchone()[0]

            if t.primary_key:
                pk = ", ".join(t.primary_key)
                dupes = con.execute(
                    f"SELECT coalesce(sum(n - 1), 0) FROM (SELECT count(*) AS n FROM {t.name} GROUP BY {pk} HAVING n > 1)"
                ).fetchone()[0]
                results.append(CheckResult(t.name, "pk_unique", f"({pk})", int(dupes)))

            for col in dict.fromkeys([*t.primary_key, *t.not_null]):
                nulls = con.execute(f"SELECT count(*) FROM {t.name} WHERE {col} IS NULL").fetchone()[0]
                results.append(CheckResult(t.name, "not_null", col, int(nulls)))

            for fk in t.foreign_keys:
                cond = " AND ".join(f"p.{rc} = c.{cc}" for cc, rc in zip(fk.columns, fk.ref_columns))
                not_null = " AND ".join(f"c.{cc} IS NOT NULL" for cc in fk.columns)
                orphans = con.execute(
                    f"SELECT count(*) FROM {t.name} c WHERE {not_null} "
                    f"AND NOT EXISTS (SELECT 1 FROM {fk.references} p WHERE {cond})"
                ).fetchone()[0]
                results.append(CheckResult(
                    t.name, "fk_integrity",
                    f"({', '.join(fk.columns)}) -> {fk.references}({', '.join(fk.ref_columns)})",
                    int(orphans), fk.severity,
                ))

            if total:
                exprs = ", ".join(f'avg(CASE WHEN "{c}" IS NULL THEN 1.0 ELSE 0.0 END)' for c in t.columns)
                rates = con.execute(f"SELECT {exprs} FROM {t.name}").fetchone()
                null_rates[t.name] = {c: float(r) for c, r in zip(t.columns, rates)}
            for col, limit in t.max_null_rate.items():
                rate = null_rates.get(t.name, {}).get(col, 0.0)
                failing = 0 if rate <= limit else int(round(rate * total))
                results.append(CheckResult(t.name, "max_null_rate", f"{col} {rate:.2%} <= {limit:.0%}", failing))
    finally:
        con.close()
    return results, null_rates


def print_report(results: list[CheckResult], null_rates: dict) -> None:
    print("DATA QUALITY REPORT")
    print("=" * 110)
    print(f"{'status':<7}{'table':<30}{'check':<15}{'detail':<50}{'failing':>8}")
    print("-" * 110)
    for r in results:
        print(f"{r.status:<7}{r.table:<30}{r.check:<15}{r.detail[:49]:<50}{r.failing_rows:>8,}")
    print("-" * 110)

    print("\nNULL RATES (silver + gold columns with any nulls)")
    print("-" * 110)
    for table, cols in null_rates.items():
        if table.startswith("bronze."):
            continue
        for col, rate in cols.items():
            if rate > 0:
                print(f"  {table + '.' + col:<60}{rate:>10.3%}")

    counts = {s: sum(r.status == s for r in results) for s in ("PASS", "WARN", "FAIL")}
    print("-" * 110)
    print(f"Summary: {len(results)} checks | {counts['PASS']} pass | {counts['WARN']} warn | {counts['FAIL']} fail")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run catalog data-quality checks.")
    parser.add_argument("--db", type=Path, default=DB_PATH)
    args = parser.parse_args()
    if not args.db.exists():
        print(f"Lakehouse not found at {args.db}. Run `make build` first.", file=sys.stderr)
        return 1
    results, null_rates = run_checks(args.db)
    print_report(results, null_rates)
    return 1 if any(r.status == "FAIL" for r in results) else 0


if __name__ == "__main__":
    sys.exit(main())
