from __future__ import annotations

from typing import Final

from src.strategies.ids import STICKY_IMPULSE_CRASH, STICKY_MOM60_POST_CRASH_ANCHOR

# 2026-09-11 champion switch (ADR_20260911_POST_CRASH_CAMPAIGN_ANCHOR): sticky.mom60_raw (P27)
# is 0% P>30/P>50 in CRASH_REBOUND sleeve windows and the 2026 tournament opens in that sleeve;
# sticky.mom60_post_crash_anchor keeps P27's identical LOTTERY_ON/mom60 routing and adds a
# domestic index +2x anchor route for CRASH_REBOUND/held-anchor INACTIVE (see docs/decisions
# task POST_CRASH_CAMPAIGN_ANCHOR). ANCHOR_STRATEGY below is an unrelated pre-existing constant
# (sticky.impulse_crash, P21) -- do not confuse the two.
CHAMPION_STRATEGY: Final[str] = STICKY_MOM60_POST_CRASH_ANCHOR
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
        "sticky.mom60_runner_reversal",
        "sticky.mom60_inactive_participate",
        "sticky.mom60_post_crash_anchor",
    }
)
CONVEXITY_ADOPTION_MODELS: Final[frozenset[str]] = frozenset(
    {"portfolio.convexity_hold", "portfolio.convexity_rebalance", "portfolio.convexity_variant"}
)
LOTTERY_ADOPTION_MODELS: Final[frozenset[str]] = frozenset({"portfolio.lottery_exposure", "portfolio.lottery_rebalance"})
