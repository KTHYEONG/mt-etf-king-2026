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


def test_aggregate_session_diagnostics_empty_reports_gross_metrics_unavailable() -> None:
    from src.execution.ledger_transition import aggregate_session_diagnostics

    # When: no session ever produced a transition
    diag = aggregate_session_diagnostics([], gross_limit=1.90)

    # Then: gross compliance is unmeasurable, not 'zero violations'
    assert diag.gross_violation_count is None
    assert diag.effective_gross_max is None
    assert diag.turnover_mean == 0.0
    assert diag.fill_count == 0
    assert diag.unfilled_count == 0


def test_aggregate_session_diagnostics_nonempty_reports_concrete_gross() -> None:
    from types import SimpleNamespace

    from src.execution.ledger_transition import aggregate_session_diagnostics

    # Given: two real sessions, one of which breached the execution gross limit
    sessions = [
        SimpleNamespace(
            execution_gross_violation=False,
            carry_gross_drift=False,
            delever_required_next_session=False,
            close_realized_gross=1.80,
            turnover_weight=0.5,
            fill_count=1,
            unfilled_count=0,
        ),
        SimpleNamespace(
            execution_gross_violation=True,
            carry_gross_drift=True,
            delever_required_next_session=True,
            close_realized_gross=2.00,
            turnover_weight=0.1,
            fill_count=1,
            unfilled_count=1,
        ),
    ]

    # When
    diag = aggregate_session_diagnostics(sessions, gross_limit=1.90)

    # Then
    assert diag.gross_violation_count == 1
    assert abs(float(diag.effective_gross_max) - 2.00) < 1e-9
    assert abs(float(diag.turnover_mean) - 0.3) < 1e-9
    assert diag.fill_count == 2
    assert diag.unfilled_count == 1
