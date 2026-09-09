from datetime import date
import polars as pl
import pytest


def _one_row_panel(day: date, price: float) -> pl.DataFrame:
    return pl.DataFrame(
        {"date": [day], "ticker": ["A"], "open": [price]},
        schema={"date": pl.Date, "ticker": pl.String, "open": pl.Float64},
    )


def test_next_open_execution_uses_prices_from_the_panel_of_this_call() -> None:
    from src.backtest.execution import NextOpenExecution
    from src.core.calendar import get_calendar

    calendar = get_calendar()
    decision_date = date(2026, 1, 5)
    execution_date = calendar.next_session(decision_date)
    execution = NextOpenExecution(calendar)

    # When: panel A is resolved first, then a different panel B for the same date
    fills_a, unfilled_a = execution.resolve({"A": 1.0}, _one_row_panel(execution_date, 100.0), decision_date)
    fills_b, unfilled_b = execution.resolve({"A": 1.0}, _one_row_panel(execution_date, 999.0), decision_date)

    # Then: each call reflects its own panel (R10/R12)
    assert unfilled_a == () and unfilled_b == ()
    assert fills_a[0].price == pytest.approx(100.0)
    assert fills_b[0].price == pytest.approx(999.0)
    assert fills_b[0].execution_date == execution_date


def test_next_open_execution_cache_retains_only_the_active_panel() -> None:
    from src.backtest.execution import NextOpenExecution
    from src.core.calendar import get_calendar

    calendar = get_calendar()
    decision_date = date(2026, 1, 5)
    execution_date = calendar.next_session(decision_date)
    execution = NextOpenExecution(calendar)

    # Given: one panel resolved -> exactly one cached execution date
    panel_a = _one_row_panel(execution_date, 100.0)
    execution.resolve({"A": 1.0}, panel_a, decision_date)
    assert len(execution._open_prices) == 1

    # When: a different panel object is resolved
    panel_b = _one_row_panel(execution_date, 999.0)
    execution.resolve({"A": 1.0}, panel_b, decision_date)

    # Then: the cache is scoped to the active panel only - it does not grow per panel (R11)
    assert len(execution._open_prices) == 1
    assert execution._cache_panel is panel_b

    # Then: re-resolving the same panel reuses the cache without growing it
    execution.resolve({"A": 1.0}, panel_b, decision_date)
    assert len(execution._open_prices) == 1
