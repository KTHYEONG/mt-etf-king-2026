from __future__ import annotations


def test_p31_in_semantic_adoption_set_champion_unchanged() -> None:
    from src.cli.constants import CHAMPION_STRATEGY, STICKY_ADOPTION_MODELS
    from src.strategies.ids import STICKY_MOM60_RAW

    assert "convex.lottery_impulse" in STICKY_ADOPTION_MODELS
    assert "sticky.fillable_mom60" in STICKY_ADOPTION_MODELS
    assert CHAMPION_STRATEGY == STICKY_MOM60_RAW
