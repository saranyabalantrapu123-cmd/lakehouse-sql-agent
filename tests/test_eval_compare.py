"""Result-set comparison used by the eval harness."""

from __future__ import annotations

import datetime as dt
from decimal import Decimal

from evals.compare import results_match


def test_order_insensitive():
    assert results_match([["a", 1], ["b", 2]], [["b", 2], ["a", 1]])


def test_numeric_tolerance_and_types():
    assert results_match([[Decimal("1255695.13")]], [[1255695.1]])
    assert results_match([[4.08642062404257]], [[4.086]])
    assert not results_match([[4.0864]], [[4.2]])


def test_extra_columns_allowed():
    assert results_match([["SP", 10]], [[10, "SP", 99.5]])
    assert not results_match([["SP", 10]], [["SP"]])


def test_dates_and_booleans_normalised():
    assert results_match([[dt.datetime(2017, 8, 1), 5]], [["2017-08-01", 5]])
    assert results_match([["2017-08-01T00:00:00", 5]], [["2017-08", 5]])
    assert results_match([[False, 4.29]], [[0, 4.29]])


def test_row_count_must_match():
    assert not results_match([["a"], ["b"]], [["a"]])
    assert results_match([], [])
