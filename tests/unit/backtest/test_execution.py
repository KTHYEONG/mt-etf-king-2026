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
    from datetime import date

    import polars as pl

    from src.backtest.execution import NextOpenExecution
    from src.core.calendar import TradingCalendar

    cal = TradingCalendar()
    execution = NextOpenExecution(cal)
    panel = pl.DataFrame(
        [
            {"date": date(2026, 8, 14), "ticker": "A", "close": 100.0, "open": 100.0, "is_tradable": True},
            {"date": date(2026, 8, 18), "ticker": "A", "close": 130.0, "open": None, "is_tradable": True},
            {"date": date(2026, 8, 18), "ticker": "B", "close": 50.0, "open": 50.0, "is_tradable": False},
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
            {"date": date(2026, 8, 14), "ticker": "B", "close": 50.0, "open": 50.0, "is_tradable": True},
            {"date": date(2026, 8, 18), "ticker": "B", "close": 55.0, "open": 55.0, "is_tradable": False},
        ]
    )
    fills_b, unfilled_b = execution.resolve({"B": 1.0}, panel_b, decision_date=date(2026, 8, 14))
    assert len(fills_b) == 1
    assert unfilled_b == ()


def test_next_open_fill_ignores_is_tradable_lookahead() -> None:
    from datetime import date

    import polars as pl

    from src.backtest.execution import NextOpenExecution
    from src.core.calendar import TradingCalendar

    cal = TradingCalendar()
    execution = NextOpenExecution(cal)
    panel = pl.DataFrame(
        [
            {"date": date(2026, 8, 14), "ticker": "A", "close": 100.0, "open": 100.0, "is_tradable": True},
            {
                "date": date(2026, 8, 18),
                "ticker": "A",
                "close": None,
                "open": 120.0,
                "is_tradable": False,
                "trading_value": 0.0,
            },
        ]
    )
    fills, unfilled = execution.resolve({"A": 0.95}, panel, decision_date=date(2026, 8, 14))
    assert unfilled == ()
    assert len(fills) == 1
    assert fills[0].execution_date == date(2026, 8, 18)
    assert fills[0].price == 120.0


def test_is_open_fillable_rejects_invalid_prices() -> None:
    import math

    from src.backtest.execution import is_open_fillable

    assert is_open_fillable(None) is False
    assert is_open_fillable(float("nan")) is False
    assert is_open_fillable(0.0) is False
    assert is_open_fillable(-1.0) is False
    assert is_open_fillable(100.0) is True
    assert is_open_fillable(math.inf) is False
