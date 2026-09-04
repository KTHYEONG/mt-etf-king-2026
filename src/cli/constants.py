from __future__ import annotations

from typing import Final

from src.strategies.ids import STICKY_IMPULSE_CRASH, STICKY_MOM60_RAW

CHAMPION_STRATEGY: Final[str] = STICKY_MOM60_RAW
ANCHOR_STRATEGY: Final[str] = STICKY_IMPULSE_CRASH
STICKY_ADOPTION_MODELS: Final[frozenset[str]] = frozenset(
    {
        "sticky.leader_base",
        "sticky.impulse_crash",
        "sticky.family_peak_lock",
        "sticky.split_fill_lock",
        "sticky.mom60_peak_lock",
        "sticky.house_money",
        "sticky.mom60_concentrated",
        "sticky.mom60_raw",
        "sticky.mom60_hold",
        "sticky.mom60_abs_cash",
        "sticky.equity_mom60",
        "sticky.equity_mom60_vol",
        "sticky.fillable_mom60",
        "convex.lottery_impulse",
    }
)
