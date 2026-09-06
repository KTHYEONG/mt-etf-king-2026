"""P4 CLI decomposition: sticky core hook table."""
from src.cli.commands.backtest.families.sticky_core import CORE_HOOKS


def test_p4_backtest_sticky_core_hooks_cover_crash_family() -> None:
    assert "sticky.impulse_crash" in CORE_HOOKS
    assert all(callable(h) for h in CORE_HOOKS.values())
