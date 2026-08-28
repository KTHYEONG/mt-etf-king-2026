"""Execution-time participation cap (INV-11)."""
from __future__ import annotations

from src.backtest.liquidity import cap_target_weights_by_adv


def test_cap_limits_buy_to_adv_participation() -> None:
    capped = cap_target_weights_by_adv(
        target={"A": 1.0},
        current={},
        equity=1_000_000_000.0,
        adv_by_ticker={"A": 10_000_000_000.0},
        max_order_to_adv=0.01,
    )
    assert abs(capped["A"] - 0.1) < 1e-12


def test_cap_allows_full_target_when_within_limit() -> None:
    capped = cap_target_weights_by_adv(
        target={"A": 0.05},
        current={},
        equity=1_000_000_000.0,
        adv_by_ticker={"A": 10_000_000_000.0},
        max_order_to_adv=0.01,
    )
    assert capped["A"] == 0.05


def test_cap_partial_sell_when_exceeds_limit() -> None:
    capped = cap_target_weights_by_adv(
        target={"A": 0.0},
        current={"A": 1.0},
        equity=1_000_000_000.0,
        adv_by_ticker={"A": 10_000_000_000.0},
        max_order_to_adv=0.01,
    )
    assert abs(capped["A"] - 0.9) < 1e-12
