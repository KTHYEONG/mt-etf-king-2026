"""P4 CLI decomposition: portfolio_exposure.py hooks (P14/P16/P31)."""
from src.cli.commands.backtest.families import portfolio as portfolio_mod
from src.cli.commands.backtest.families.portfolio_exposure import (
    _hook_convexity,
    _hook_lottery_exposure,
    _hook_lottery_rebalance,
)


def test_p4_backtest_portfolio_exposure_hooks() -> None:
    assert callable(_hook_lottery_exposure)
    assert callable(_hook_lottery_rebalance)
    assert callable(_hook_convexity)
    assert portfolio_mod._PORTFOLIO_FORENSICS["portfolio.convexity_hold"] is _hook_convexity
    assert portfolio_mod._PORTFOLIO_FORENSICS["portfolio.lottery_exposure"] is _hook_lottery_exposure
