"""P4 CLI decomposition: sticky split hook table."""
from src.cli.commands.backtest.families.sticky_split import SPLIT_HOOKS


def test_p4_backtest_sticky_split_hooks() -> None:
    assert "sticky.split_fill_lock" in SPLIT_HOOKS
    assert all(callable(h) for h in SPLIT_HOOKS.values())
