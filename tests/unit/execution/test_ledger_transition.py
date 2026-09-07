# ruff: noqa
"""Facade fidelity for the src execution ledger transition split (P5)."""

from __future__ import annotations


def test_ledger_transition_canonical_home() -> None:
    import src.execution.ledger as facade
    from src.execution.ledger_transition import resolve_session_intent, transition_portfolio_state

    assert facade.transition_portfolio_state is transition_portfolio_state
    assert facade.resolve_session_intent is resolve_session_intent
    assert transition_portfolio_state.__module__ == "src.execution.ledger_transition"

def test_ledger_transition_wires_cash_accounting() -> None:
    from datetime import date
    from src.execution.ledger_state import PortfolioLedgerState
    from src.backtest.costs import CostConfig, CostModel
    from src.portfolio.intent import PortfolioIntent, CASH_INTENT, HOLD_INTENT
    from src.execution.cash_accounting import cash_order_transition
    kw = dict(prior_state=PortfolioLedgerState(cash=20.0, shares={"A":4.0,"B":4.0}), intent=PortfolioIntent(kind="target",weights={"B":0.8}), decision_date=date(2024,1,2), prev_closes={"A":10.0,"B":10.0}, opens={"B":10.0}, closes={"A":10.0,"B":10.0}, cost_model=CostModel(CostConfig(commission_bps=0.0)), adv_by_ticker={"A":1e6,"B":1e6}, max_order_to_adv=0.01, exposure_limits=(0.8,1.6,0.05), leverage_multiples={"A":1,"B":1}, execution=None, panel=None, lot_size=1)
    from src.execution.ledger_transition import transition_portfolio_state
    result=transition_portfolio_state(**kw)
    expected=cash_order_transition(**kw)
    assert result == expected
    assert result.equity_close == 100.0
