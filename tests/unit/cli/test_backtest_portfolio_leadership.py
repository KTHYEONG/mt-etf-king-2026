"""P4 CLI decomposition: portfolio_leadership.py hooks (P11/P12)."""
from src.cli.commands.backtest.families import portfolio as portfolio_mod
from src.cli.commands.backtest.families.portfolio_leadership import (
    _hook_leadership_confidence,
    _hook_leadership_policy,
)


def test_p4_backtest_portfolio_leadership_hooks() -> None:
    assert callable(_hook_leadership_policy)
    assert callable(_hook_leadership_confidence)
    assert portfolio_mod._PORTFOLIO_FORENSICS["portfolio.leadership_policy"] is _hook_leadership_policy
