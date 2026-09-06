"""P4 CLI decomposition: sticky champion hook table (golden path)."""
from src.cli.commands.backtest.families.sticky_champion import CHAMPION_HOOKS


def test_p4_backtest_sticky_champion_hooks_cover_raw_path() -> None:
    assert "sticky.mom60_raw" in CHAMPION_HOOKS
    assert all(callable(h) for h in CHAMPION_HOOKS.values())
