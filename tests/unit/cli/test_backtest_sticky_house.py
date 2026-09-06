"""P4 CLI decomposition: sticky house-money hook table."""
from src.cli.commands.backtest.families.sticky_house import EQUITY_GROUP_IDS, HOUSE_HOOKS


def test_p4_backtest_sticky_house_hooks() -> None:
    assert "sticky.house_money" in HOUSE_HOOKS
    assert len(EQUITY_GROUP_IDS) > 0
    assert all(callable(h) for h in HOUSE_HOOKS.values())
