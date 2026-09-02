def test_r3_no_cli_impl_module() -> None:
    from pathlib import Path

    impl = Path("src/cli/_impl.py")
    assert impl.exists()
    assert sum(1 for _ in impl.open(encoding="utf-8")) > 1000


def test_r3_impl_has_real_backtest_body() -> None:
    from pathlib import Path

    content = Path("src/cli/_impl.py").read_text(encoding="utf-8")
    assert "TournamentSimulator" in content
    assert "BacktestEngine" in content
    with Path("src/cli/_impl.py").open(encoding="utf-8") as handle:
        assert sum(1 for _ in handle) > 1000


def test_r3_dispatch_family_of_sticky() -> None:
    from src.cli.dispatch import family_of
    from src.strategies.ids import STICKY_MOM60_RAW

    assert family_of("P27") == "sticky"
    assert family_of(STICKY_MOM60_RAW) == "sticky"


def test_r3_normalize_cli_model_arg_sets_semantic() -> None:
    import argparse

    from src.cli.dispatch import normalize_cli_model_arg
    from src.strategies.ids import STICKY_MOM60_RAW

    args = argparse.Namespace(model="P27")
    assert normalize_cli_model_arg(args) == STICKY_MOM60_RAW
    assert args.model == STICKY_MOM60_RAW


def test_r3_backtest_handler_registry_has_champion() -> None:
    from src.cli.constants import CHAMPION_STRATEGY
    from src.cli.dispatch import STICKY_BACKTEST_HANDLERS

    assert CHAMPION_STRATEGY in STICKY_BACKTEST_HANDLERS
    assert callable(STICKY_BACKTEST_HANDLERS[CHAMPION_STRATEGY])
