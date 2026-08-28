"""Engine integration checks for execution timing and costs."""
from __future__ import annotations

from dataclasses import replace
from datetime import date

import polars as pl

from src.alpha.baselines import BuyAndHoldBaseline
from src.backtest.costs import CostConfig, CostModel
from src.backtest.engine import BacktestConfig
from src.core.calendar import TradingCalendar
from src.portfolio.sizing import SizingScheme
from tests.unit.backtest.conftest import build_engine, panel_row


def test_cost_applied_on_execution_date_not_signal_date() -> None:
    cal = TradingCalendar()
    sessions = cal.sessions(date(2026, 1, 2), date(2026, 1, 9))
    rows = [
        panel_row(day=d, ticker="069500", close=100.0, mom_20=0.1, trading_value=50_000_000_000_000.0)
        for d in sessions
    ]
    panel = pl.DataFrame(rows)
    engine, _, filt = build_engine(panel, max_order_to_adv=1.0)
    strict_filt = replace(filt, max_order_to_adv=1.0)
    costs = CostConfig(commission_bps=100.0, slippage_bps=0.0, spread_bps=0.0)
    config = BacktestConfig(
        start=sessions[0],
        end=sessions[-1],
        capital=1_000_000_000.0,
        scheme=SizingScheme.TOP1,
        k=1,
        filters=strict_filt,
        costs=costs,
    )
    result = engine.run(BuyAndHoldBaseline(ticker="069500", name="B0"), panel, config)
    daily = result.daily.sort("date")
    first_day_equity = float(daily.filter(pl.col("date") == sessions[0]).select("equity").item())
    second_day_equity = float(daily.filter(pl.col("date") == sessions[1]).select("equity").item())
    expected_cost = CostModel(costs).charge(1_000_000_000.0)
    assert first_day_equity == 1_000_000_000.0
    assert second_day_equity == 1_000_000_000.0 - expected_cost


def test_engine_applies_participation_cap_before_fill() -> None:
    cal = TradingCalendar()
    sessions = cal.sessions(date(2026, 1, 2), date(2026, 1, 9))
    rows = [
        {
            "date": d,
            "ticker": "069500",
            "name": "KODEX 200",
            "close": 100.0,
            "open": 100.0,
            "is_tradable": True,
            "trading_value": 1_000_000_000.0,
            "underlying_index_name": "Index",
            "mom_20": 0.1,
        }
        for d in sessions
    ]
    panel = pl.DataFrame(rows)
    engine, _, filt = build_engine(panel, max_order_to_adv=0.01)
    strict_filt = replace(filt, max_order_to_adv=0.01)
    config = BacktestConfig(
        start=sessions[0],
        end=sessions[-1],
        capital=1_000_000_000.0,
        scheme=SizingScheme.TOP1,
        k=1,
        filters=strict_filt,
        costs=CostConfig(0.0, 0.0, 0.0),
    )
    result = engine.run(BuyAndHoldBaseline(ticker="069500", name="B0"), panel, config)
    assert result.trades.height >= 1
    first_weight = float(result.trades.sort("execution_date").select("weight").item(0, 0))
    assert first_weight <= 0.01 + 1e-9
