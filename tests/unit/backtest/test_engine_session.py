"""Facade fidelity for the src backtest engine session split (P5)."""

from __future__ import annotations


def test_engine_session_canonical_home() -> None:
    import src.backtest.engine as facade
    from src.backtest.engine_session import bootstrap_prestart_intent, emit_session_trace

    assert bootstrap_prestart_intent.__module__ == "src.backtest.engine_session"
    assert emit_session_trace.__module__ == "src.backtest.engine_session"
    # explain_selection_drops is re-exported through the facade from its
    # canonical home in portfolio.selection (unchanged by the split).
    assert facade.explain_selection_drops.__module__ == "src.portfolio.selection"
