"""Result-set comparison for execution-accuracy scoring.

Two result sets match when, after normalising values, the gold rows equal the predicted rows
as multisets (row order ignored), with numeric tolerance. If the agent returns extra columns,
we look for a subset/permutation of its columns that matches the gold columns.
"""

from __future__ import annotations

import datetime as dt
import decimal
import itertools
import re
from typing import Any, Sequence

REL_TOL = 5e-3   # 0.5% relative tolerance (allows the agent to round to 2-3 significant decimals)
ABS_TOL = 1e-2
MAX_EXTRA_COLUMNS = 6

_DATE_ONLY = re.compile(r"^(\d{4}-\d{2}-\d{2})[T ]00:00:00(\.0+)?$")
_YEAR_MONTH = re.compile(r"^(\d{4}-\d{2})$")


def normalise(value: Any) -> Any:
    """Map equivalent representations onto one canonical value."""
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float, decimal.Decimal)):
        return float(value)
    if isinstance(value, dt.datetime):
        value = value.isoformat()
    elif isinstance(value, dt.date):
        return value.isoformat()
    if isinstance(value, str):
        s = value.strip()
        if m := _DATE_ONLY.match(s):
            return m.group(1)
        if m := _YEAR_MONTH.match(s):
            return f"{m.group(1)}-01"
        if s.lower() in ("true", "false"):
            return int(s.lower() == "true")
        return s.lower()
    return value


def _values_equal(a: Any, b: Any) -> bool:
    if isinstance(a, float) and isinstance(b, float):
        return abs(a - b) <= max(ABS_TOL, REL_TOL * max(abs(a), abs(b)))
    return a == b


def _rows_match(gold: list[tuple], pred: list[tuple]) -> bool:
    if len(gold) != len(pred):
        return False
    remaining = list(pred)
    # Sort gold rows so exact-ish matches are consumed deterministically.
    for g in sorted(gold, key=repr):
        for i, p in enumerate(remaining):
            if all(_values_equal(x, y) for x, y in zip(g, p)):
                del remaining[i]
                break
        else:
            return False
    return True


def results_match(gold_rows: Sequence[Sequence[Any]], pred_rows: Sequence[Sequence[Any]]) -> bool:
    gold = [tuple(normalise(v) for v in r) for r in gold_rows]
    pred = [tuple(normalise(v) for v in r) for r in pred_rows]
    if len(gold) != len(pred):
        return False
    if not gold:
        return True
    g_width, p_width = len(gold[0]), len(pred[0])
    if p_width < g_width:
        return False
    if p_width == g_width and _rows_match(gold, pred):
        return True
    if p_width - g_width > MAX_EXTRA_COLUMNS:
        return False
    for cols in itertools.permutations(range(p_width), g_width):
        projected = [tuple(r[c] for c in cols) for r in pred]
        if _rows_match(gold, projected):
            return True
    return False
