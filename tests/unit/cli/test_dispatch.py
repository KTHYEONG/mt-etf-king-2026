from __future__ import annotations


def test_convex_lottery_impulse_decide_handler() -> None:
    from src.cli.dispatch import STICKY_DECIDE_HANDLERS, family_of

    assert "convex.lottery_impulse" in STICKY_DECIDE_HANDLERS
    assert "sticky.fillable_mom60" in STICKY_DECIDE_HANDLERS
    assert family_of("P31") == "convex"
    assert family_of("P30") == "sticky"
