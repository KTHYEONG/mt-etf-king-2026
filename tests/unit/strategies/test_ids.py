from __future__ import annotations


def test_convex_lottery_impulse_id() -> None:
    from src.strategies.ids import CONVEX_LOTTERY_IMPULSE, STICKY_FILLABLE_MOM60

    assert CONVEX_LOTTERY_IMPULSE == "convex.lottery_impulse"
    assert CONVEX_LOTTERY_IMPULSE != STICKY_FILLABLE_MOM60
