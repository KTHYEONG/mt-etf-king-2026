from __future__ import annotations


def test_convex_lottery_impulse_decide_handler() -> None:
    from src.cli.commands.backtest import FAMILY_RUNNERS
    from src.cli.commands.backtest.families.convex import run_convex_family
    from src.cli.commands.backtest.families.sticky_house import EQUITY_GROUP_IDS, _hook_equity_group
    from src.strategies.registry import family_of

    assert "convex.lottery_impulse" in EQUITY_GROUP_IDS
    assert "sticky.fillable_mom60" in EQUITY_GROUP_IDS
    assert family_of("convex.lottery_impulse") == "convex"
    assert family_of("sticky.fillable_mom60") == "sticky"
    assert FAMILY_RUNNERS["convex"] is run_convex_family
    assert callable(_hook_equity_group)
