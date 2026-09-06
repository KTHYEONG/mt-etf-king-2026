"""P4 CLI decomposition: sticky cash hook table."""
from src.cli.commands.backtest.families.sticky_cash import CASH_HOOKS


def test_p4_backtest_sticky_cash_hooks() -> None:
    assert "sticky.mom60_hold" in CASH_HOOKS
    assert "sticky.mom60_abs_cash" in CASH_HOOKS
    assert all(callable(h) for h in CASH_HOOKS.values())
