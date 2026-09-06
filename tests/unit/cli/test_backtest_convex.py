"""P4 CLI decomposition: convex family runner table."""
from src.cli.commands.backtest import FAMILY_RUNNERS
from src.cli.commands.backtest.families.convex import run_convex_family


def test_p4_backtest_convex_hooks_cover_impulse_models() -> None:
    assert callable(run_convex_family)
    assert FAMILY_RUNNERS["convex"] is run_convex_family
