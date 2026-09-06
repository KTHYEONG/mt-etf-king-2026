"""Facade fidelity for the src execution ledger split (P5)."""

from __future__ import annotations


def test_ledger_state_canonical_home() -> None:
    import src.execution.ledger as facade
    from src.execution.ledger_state import PortfolioLedgerState, is_priceless_session, ledger_state_from_weights

    assert facade.PortfolioLedgerState is PortfolioLedgerState
    assert facade.ledger_state_from_weights is ledger_state_from_weights
    assert facade.is_priceless_session is is_priceless_session
    assert PortfolioLedgerState.__module__ == "src.execution.ledger_state"
