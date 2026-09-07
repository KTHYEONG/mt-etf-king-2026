"""Facade fidelity for the src tournament simulator windows split (P5)."""

from __future__ import annotations


def test_simulator_windows_canonical_home() -> None:
    import src.tournament.simulator as facade
    from src.tournament.simulator_windows import resolve_prestart_intent, simulate_window_from_cache

    assert facade.simulate_window_from_cache is simulate_window_from_cache
    assert facade.resolve_prestart_intent is resolve_prestart_intent
    assert simulate_window_from_cache.__module__ == "src.tournament.simulator_windows"


def test_simulator_windows_resolves_default_exposure_limits() -> None:
    from datetime import date
    from types import SimpleNamespace
    from src.tournament.simulator_windows import simulate_window_from_cache

    class Model:
        scores_path_independent = True
        def reset_trackers(self) -> None: pass

        def score(self, snapshot, context):
            return {}

    cache = SimpleNamespace(dates=(date(2026, 1, 2),), close_map={date(2026, 1, 2): {}}, scores={date(2026, 1, 2): {}}, open_map={}, adv_map={}, panel=None)
    result = simulate_window_from_cache(Model(), cache, 0, 1, 100.0, SimpleNamespace(max_position_weight=1.0, max_order_to_adv=0.01), None)
    assert result[0] == 0.0
