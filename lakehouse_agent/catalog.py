"""Load `catalog.yaml` into typed objects and answer lineage / key questions."""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import yaml

from lakehouse_agent.config import CATALOG_PATH

LAYERS = ("bronze", "silver", "gold")


@dataclass(frozen=True)
class ForeignKey:
    columns: tuple[str, ...]
    references: str
    ref_columns: tuple[str, ...]
    severity: str = "error"

    def as_dict(self) -> dict:
        return {
            "columns": list(self.columns),
            "references": self.references,
            "ref_columns": list(self.ref_columns),
        }


@dataclass(frozen=True)
class TableInfo:
    name: str  # fully qualified: layer.table
    layer: str
    description: str
    columns: dict[str, str]
    primary_key: tuple[str, ...]
    foreign_keys: tuple[ForeignKey, ...]
    upstream: tuple[str, ...]
    source: str | None = None
    not_null: tuple[str, ...] = ()
    max_null_rate: dict[str, float] = field(default_factory=dict)

    @property
    def short_name(self) -> str:
        return self.name.split(".", 1)[1]


class Catalog:
    """In-memory view of the data catalog with lineage helpers."""

    def __init__(self, raw: dict):
        self.dataset: str = raw.get("dataset", "")
        self.layer_descriptions: dict[str, str] = raw.get("layers", {})
        audit = raw.get("bronze_audit_columns", {}) or {}
        self.tables: dict[str, TableInfo] = {}
        for name, spec in raw["tables"].items():
            columns = dict(spec.get("columns", {}))
            if spec["layer"] == "bronze":
                columns.update(audit)
            self.tables[name] = TableInfo(
                name=name,
                layer=spec["layer"],
                description=" ".join(str(spec.get("description", "")).split()),
                columns=columns,
                primary_key=tuple(spec.get("primary_key") or ()),
                foreign_keys=tuple(
                    ForeignKey(
                        columns=tuple(fk["columns"]),
                        references=fk["references"],
                        ref_columns=tuple(fk["ref_columns"]),
                        severity=fk.get("severity", "error"),
                    )
                    for fk in spec.get("foreign_keys") or ()
                ),
                upstream=tuple(spec.get("upstream") or ()),
                source=spec.get("source"),
                not_null=tuple(spec.get("not_null") or ()),
                max_null_rate=dict(spec.get("max_null_rate") or {}),
            )

    # ------------------------------------------------------------------ lookup
    def resolve(self, name: str) -> TableInfo | None:
        """Resolve 'gold.fact_orders' or an unambiguous bare name like 'fact_orders'."""
        key = name.strip().strip('"').lower()
        if key in self.tables:
            return self.tables[key]
        matches = [t for t in self.tables.values() if t.short_name == key]
        if len(matches) == 1:
            return matches[0]
        return None

    def candidates(self, name: str) -> list[str]:
        """All fully-qualified names whose short name matches (for ambiguity messages)."""
        key = name.strip().lower().split(".")[-1]
        return [t.name for t in self.tables.values() if t.short_name == key]

    def by_layer(self, layer: str | None = None) -> list[TableInfo]:
        return [t for t in self.tables.values() if layer is None or t.layer == layer]

    # ----------------------------------------------------------------- lineage
    def downstream(self, name: str) -> list[str]:
        """Tables that list `name` as a direct upstream."""
        return sorted(t.name for t in self.tables.values() if name in t.upstream)

    def lineage(self, name: str) -> dict:
        """Full upstream and downstream closure, with direct edges for drawing graphs."""
        table = self.tables[name]

        def walk(start: str, step) -> list[str]:
            seen: list[str] = []
            frontier = list(step(start))
            while frontier:
                cur = frontier.pop(0)
                if cur in seen:
                    continue
                seen.append(cur)
                frontier.extend(step(cur))
            return seen

        upstream_all = walk(name, lambda n: self.tables[n].upstream if n in self.tables else ())
        downstream_all = walk(name, self.downstream)
        nodes = [name, *upstream_all, *downstream_all]
        edges = sorted({(u, n) for n in nodes if n in self.tables for u in self.tables[n].upstream if u in nodes})
        sources = sorted({self.tables[n].source for n in [name, *upstream_all] if self.tables[n].source})
        return {
            "table": name,
            "layer": table.layer,
            "direct_upstream": list(table.upstream),
            "direct_downstream": self.downstream(name),
            "all_upstream": upstream_all,
            "all_downstream": downstream_all,
            "raw_sources": sources,
            "edges": [{"from": u, "to": d} for u, d in edges],
        }

    def fk_edges(self) -> list[tuple[str, ForeignKey]]:
        return [(t.name, fk) for t in self.tables.values() for fk in t.foreign_keys]


@lru_cache(maxsize=4)
def _load(path: str) -> Catalog:
    with open(path, encoding="utf-8") as fh:
        return Catalog(yaml.safe_load(fh))


def load_catalog(path: Path | str = CATALOG_PATH) -> Catalog:
    return _load(str(Path(path).resolve()))
