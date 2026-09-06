"""P4 CLI decomposition: sticky lock/split hook tables."""
from src.cli.commands.backtest.families.sticky_core import CORE_HOOKS
from src.cli.commands.backtest.families.sticky_locks import LOCK_HOOKS
from src.cli.commands.backtest.families.sticky_split import SPLIT_HOOKS


def test_p4_backtest_sticky_lock_hooks() -> None:
    assert "sticky.leader_base" in LOCK_HOOKS
    assert "sticky.family_peak_lock" in LOCK_HOOKS
    assert all(callable(h) for h in LOCK_HOOKS.values())


def test_p4_backtest_sticky_core_split_hooks() -> None:
    assert "sticky.impulse_crash" in CORE_HOOKS
    assert "sticky.split_fill_lock" in SPLIT_HOOKS
    assert all(callable(h) for h in CORE_HOOKS.values())
    assert all(callable(h) for h in SPLIT_HOOKS.values())
