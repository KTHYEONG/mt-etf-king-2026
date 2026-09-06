def test_r3_no_cli_impl_module() -> None:
    from pathlib import Path

    impl = Path("src/cli/_impl.py")
    assert not impl.exists()
    assert not Path("src/cli/dispatch.py").exists()


def test_r3_impl_has_real_backtest_body() -> None:
    from pathlib import Path

    core = Path("src/cli/commands/backtest/_core.py").read_text(encoding="utf-8")
    prep = Path("src/cli/commands/backtest/_prep.py").read_text(encoding="utf-8")
    assert "TournamentSimulator" in prep
    assert "BacktestEngine" in prep
    assert "run_cells" in core
    assert sum(1 for _ in core.splitlines()) > 100


def test_r3_dispatch_family_of_sticky() -> None:
    from src.strategies.ids import STICKY_MOM60_RAW
    from src.strategies.registry import family_of

    assert family_of("sticky.mom60_raw") == "sticky"
    assert family_of(STICKY_MOM60_RAW) == "sticky"


def test_r3_normalize_cli_model_arg_sets_semantic() -> None:
    import argparse

    from src.cli.context import normalize_cli_model_arg
    from src.strategies.ids import STICKY_MOM60_RAW

    args = argparse.Namespace(model="sticky.mom60_raw")
    assert normalize_cli_model_arg(args) == STICKY_MOM60_RAW
    assert args.model == STICKY_MOM60_RAW


def test_r3_backtest_handler_registry_has_champion() -> None:
    from src.cli.commands.backtest import FAMILY_RUNNERS
    from src.cli.constants import CHAMPION_STRATEGY
    from src.strategies.registry import family_of

    assert family_of(CHAMPION_STRATEGY) in FAMILY_RUNNERS
    assert callable(FAMILY_RUNNERS[family_of(CHAMPION_STRATEGY)])
