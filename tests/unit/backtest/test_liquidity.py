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


def test_sell_first_prevents_switch_overlap_gross_breach() -> None:
    from src.backtest.liquidity import cap_target_weights_by_adv, constrain_target_weights_sell_first
    from src.portfolio.constraints import gross_exposure

    equity = 1_000_000_000.0
    phi = 0.01
    max_gross = 1.90
    current = {"OLD": 0.90}
    target = {"OLD": 0.0, "NEW": 0.90}
    multiples = {"OLD": 2, "NEW": 2}
    adv = {"OLD": equity * 0.10 / phi, "NEW": equity * 0.90 / phi}
    parallel = cap_target_weights_by_adv(target, current, equity, adv, phi)
    assert gross_exposure(parallel, multiples) > max_gross + 1e-9
    constrained = constrain_target_weights_sell_first(
        target,
        current,
        equity,
        adv,
        phi,
        leverage_multiples=multiples,
        max_gross_exposure=max_gross,
    )
    assert gross_exposure(constrained, multiples) <= max_gross + 1e-9
    assert float(constrained.get("OLD", 0.0)) == 0.80
    assert abs(float(constrained.get("NEW", 0.0)) - 0.15) < 1e-6


def test_sell_first_cancels_buys_when_residual_budget_nonpositive() -> None:
    from src.backtest.liquidity import constrain_target_weights_sell_first
    from src.portfolio.constraints import gross_exposure

    equity = 1_000_000_000.0
    phi = 0.01
    current = {"OLD": 0.96}
    target = {"OLD": 0.0, "NEW": 0.95}
    multiples = {"OLD": 2, "NEW": 2}
    adv = {"OLD": equity * 0.01 / phi, "NEW": equity * 1.0 / phi}
    out = constrain_target_weights_sell_first(
        target,
        current,
        equity,
        adv,
        phi,
        leverage_multiples=multiples,
        max_gross_exposure=1.90,
    )
    assert "NEW" not in out or abs(float(out.get("NEW", 0.0))) < 1e-12
    assert abs(float(out["OLD"]) - 0.95) < 1e-9
    assert gross_exposure(out, multiples) >= 1.90 - 1e-9


def test_sell_first_preserves_within_budget_full_target() -> None:
    from src.backtest.liquidity import constrain_target_weights_sell_first

    equity = 1_000_000_000.0
    phi = 0.01
    current = {"OLD": 0.90}
    target = {"NEW": 0.90}
    multiples = {"OLD": 2, "NEW": 2}
    adv = {"OLD": equity * 1.0 / phi, "NEW": equity * 1.0 / phi}
    out = constrain_target_weights_sell_first(
        target,
        current,
        equity,
        adv,
        phi,
        leverage_multiples=multiples,
        max_gross_exposure=1.90,
    )
    assert abs(float(out.get("NEW", 0.0)) - 0.90) < 1e-9
    assert abs(float(out.get("OLD", 0.0))) < 1e-12
