"""Facade fidelity for the src execution ledger transition split (P5)."""

from __future__ import annotations


def test_ledger_transition_canonical_home() -> None:
    import src.execution.ledger as facade
    from src.execution.ledger_transition import resolve_session_intent, transition_portfolio_state

    assert facade.transition_portfolio_state is transition_portfolio_state
    assert facade.resolve_session_intent is resolve_session_intent
    assert transition_portfolio_state.__module__ == "src.execution.ledger_transition"
