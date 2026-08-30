from __future__ import annotations

from datetime import date

import polars as pl

from src.backtest.costs import CostConfig
from src.backtest.engine import BacktestConfig
from src.portfolio.sizing import SizingScheme
from tests.unit.backtest.conftest import build_engine, panel_row


def test_engine_passes_theme_states_to_allocate() -> None:
    start = date(2026, 1, 2)
    end = date(2026, 1, 6)
    cal = __import__("src.core.calendar", fromlist=["TradingCalendar"]).TradingCalendar()
    sessions = cal.sessions(start, end)
    rows = [panel_row(day=d, ticker="069500", close=30000.0, mom_20=0.05) for d in sessions]
    panel = pl.DataFrame(rows)
    engine, _, filt = build_engine(panel, warmup_sessions=1, max_order_to_adv=1.0)
    config = BacktestConfig(
        start=start,
        end=end,
        capital=1_000_000_000.0,
        scheme=SizingScheme.TOP1,
        k=1,
        filters=filt,
        costs=CostConfig(0.0, 0.0, 0.0),
    )

    captured: list[dict[str, str] | None] = []

    class ThemeStateModel:
        name = "P12"

        def score(self, snapshot, context):  # type: ignore[no-untyped-def]
            return {"069500": 1.0}

        def theme_states_by_representative(self) -> dict[str, str]:
            return {"069500": "LEADING"}

        def allocate(self, scores, **kwargs):  # type: ignore[no-untyped-def]
            captured.append(kwargs.get("theme_states"))
            weights = dict.fromkeys(scores, 1.0)
            return type("Alloc", (), {"weights": weights})()

    model = ThemeStateModel()
    engine.run(model, panel, config)
    assert captured
    assert any(ts is not None and ts.get("069500") == "LEADING" for ts in captured)
