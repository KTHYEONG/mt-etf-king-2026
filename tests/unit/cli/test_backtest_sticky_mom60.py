"""P4 CLI decomposition: sticky mom60 hook table."""
from src.cli.commands.backtest.families.sticky_mom60 import MOM60_HOOKS


def test_p4_backtest_sticky_mom60_hooks() -> None:
    assert "sticky.mom60_peak_lock" in MOM60_HOOKS
    assert "sticky.mom60_concentrated" in MOM60_HOOKS
    assert all(callable(h) for h in MOM60_HOOKS.values())
