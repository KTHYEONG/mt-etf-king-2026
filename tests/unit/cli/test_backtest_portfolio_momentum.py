"""P4 CLI decomposition: portfolio_momentum.py hooks (P08/P10)."""
from src.cli.commands.backtest.families import portfolio as portfolio_mod
from src.cli.commands.backtest.families.portfolio_momentum import (
    _hook_momentum_confidence,
    _hook_momentum_vehicle,
)


def test_p4_backtest_portfolio_momentum_hooks() -> None:
    assert callable(_hook_momentum_vehicle)
    assert callable(_hook_momentum_confidence)
    assert portfolio_mod._PORTFOLIO_FORENSICS["portfolio.momentum_vehicle"] is _hook_momentum_vehicle
