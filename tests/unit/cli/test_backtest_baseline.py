"""P4 CLI decomposition: baseline/convex family runners."""
from src.cli.commands.backtest import FAMILY_RUNNERS
from src.cli.commands.backtest.families.baseline import run_baseline_family
from src.cli.commands.backtest.families.convex import run_convex_family


def test_p4_backtest_baseline_family_registered() -> None:
    assert callable(run_baseline_family)
    assert FAMILY_RUNNERS["baseline"] is run_baseline_family
    assert FAMILY_RUNNERS["champion"] is run_baseline_family


def test_p4_backtest_convex_family_registered() -> None:
    assert callable(run_convex_family)
    assert FAMILY_RUNNERS["convex"] is run_convex_family
