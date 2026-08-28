"""SCENARIO-06-01 SCENARIO-06-02"""
from __future__ import annotations

from datetime import date

import polars as pl

from src.backtest.execution import NextOpenExecution
from src.core.calendar import TradingCalendar


def test_SCENARIO_06_01_next_open_fill_price() -> None:
    """SCENARIO-06-01"""
    cal = TradingCalendar()
    execution = NextOpenExecution(cal)
    panel = pl.DataFrame(
        [
            {
                "date": date(2026, 8, 14),
                "ticker": "A",
                "close": 100.0,
                "open": 100.0,
                "is_tradable": True,
            },
            {
                "date": date(2026, 8, 18),
                "ticker": "A",
                "close": 130.0,
                "open": 120.0,
                "is_tradable": True,
            },
        ]
    )
    fills, unfilled = execution.resolve({"A": 1.0}, panel, decision_date=date(2026, 8, 14))
    assert unfilled == ()
    assert len(fills) == 1
    fill = fills[0]
    assert fill.execution_date == date(2026, 8, 18)
    assert fill.price == 120.0
    assert fill.price not in (100.0, 130.0)


def test_SCENARIO_06_02_last_session_and_unfilled_conditions() -> None:
    """SCENARIO-06-02"""
    cal = TradingCalendar()
    execution = NextOpenExecution(cal)
    panel = pl.DataFrame(
        [
            {
                "date": date(2026, 8, 14),
                "ticker": "A",
                "close": 100.0,
                "open": 100.0,
                "is_tradable": True,
            },
            {
                "date": date(2026, 8, 18),
                "ticker": "A",
                "close": 130.0,
                "open": None,
                "is_tradable": True,
            },
            {
                "date": date(2026, 8, 18),
                "ticker": "B",
                "close": 50.0,
                "open": 50.0,
                "is_tradable": False,
            },
        ]
    )
    last_day = date(2026, 8, 18)
    fills_last, _ = execution.resolve({"A": 1.0}, panel, decision_date=last_day)
    assert fills_last == []

    fills_null, unfilled_null = execution.resolve({"A": 1.0}, panel, decision_date=date(2026, 8, 14))
    assert fills_null == []
    assert unfilled_null == ("A",)

    panel_b = pl.DataFrame(
        [
            {
                "date": date(2026, 8, 14),
                "ticker": "B",
                "close": 50.0,
                "open": 50.0,
                "is_tradable": True,
            },
            {
                "date": date(2026, 8, 18),
                "ticker": "B",
                "close": 55.0,
                "open": 55.0,
                "is_tradable": False,
            },
        ]
    )
    fills_b, unfilled_b = execution.resolve({"B": 1.0}, panel_b, decision_date=date(2026, 8, 14))
    assert fills_b == []
    assert unfilled_b == ("B",)
