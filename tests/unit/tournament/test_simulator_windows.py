"""Facade fidelity for the src tournament simulator windows split (P5)."""

from __future__ import annotations


def test_simulator_windows_canonical_home() -> None:
    import src.tournament.simulator as facade
    from src.tournament.simulator_windows import resolve_prestart_intent, simulate_window_from_cache

    assert facade.simulate_window_from_cache is simulate_window_from_cache
    assert facade.resolve_prestart_intent is resolve_prestart_intent
    assert simulate_window_from_cache.__module__ == "src.tournament.simulator_windows"
