"""Universe-level participation / liquidity filter checks."""
from __future__ import annotations

from datetime import date

import polars as pl

from src.universe.provider import UniverseFilters, UniverseMode
from tests.unit.universe.conftest import build_universe, panel_row


def test_participation_rate_controls_required_adv_and_liquidity() -> None:
    """max_order_to_adv maps capital to minimum ADV for tradable names."""
    day = date(2026, 8, 27)
    filt_1pct = UniverseFilters(
        mode=UniverseMode.STRUCTURAL,
        warmup_sessions=1,
        capital=1_000_000_000,
        max_order_to_adv=0.01,
    )
    filt_5pct = UniverseFilters(
        mode=UniverseMode.STRUCTURAL,
        warmup_sessions=1,
        capital=1_000_000_000,
        max_order_to_adv=0.05,
    )
    assert filt_1pct.required_adv() == 100_000_000_000
    assert filt_5pct.required_adv() == 20_000_000_000

    panel = pl.DataFrame(
        [
            panel_row(day=day, ticker="069500", close=30_000.0, trading_value=30_000_000_000),
            panel_row(day=day, ticker="451060", close=20_000.0, trading_value=50_000_000_000),
        ]
    )
    universe, _, _ = build_universe(panel, adv_window=1)
    strict = universe.get(day, filt_1pct)
    loose = universe.get(day, filt_5pct)
    assert "451060" not in strict.tickers
    assert "451060" in loose.tickers
