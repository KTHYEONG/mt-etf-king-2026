from __future__ import annotations


def test_walkforward_has_adaptive_dispatch() -> None:
    import src.tournament.champion.walkforward as module

    assert hasattr(module, "run_adaptive_specialist_research")
    assert hasattr(module, "run_champion_walk_forward")
