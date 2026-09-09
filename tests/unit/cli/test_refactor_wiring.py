def test_cli_build_parser_subcommands_unchanged() -> None:
    from src.cli import build_parser

    parser = build_parser()
    sub_actions = [action for action in parser._actions if action.dest == "subcommand"]
    assert len(sub_actions) == 1
    choices = set(sub_actions[0].choices.keys())
    expected = {
        "config-check",
        "calendar",
        "ingest",
        "normalize",
        "universe",
        "features",
        "backtest",
        "forensics",
        "loyo",
        "replay",
        "decide",
        "daily-refresh",
        "storage-migrate",
        "champion-research",
        "feasibility-audit",
    }
    assert expected == choices


def test_cli_backtest_resolves_semantic_model_flag() -> None:
    import argparse

    from src.cli.commands.backtest import cmd_backtest
    from src.strategies.ids import STICKY_MOM60_RAW
    from src.strategies.registry import resolve_strategy_id

    assert resolve_strategy_id(STICKY_MOM60_RAW) == STICKY_MOM60_RAW
    assert resolve_strategy_id("STICKY.MOM60_RAW") == STICKY_MOM60_RAW
    args = argparse.Namespace(
        model=STICKY_MOM60_RAW,
        start="2018-01-02",
        end="2018-03-31",
        leverage_scenario="aggressive",
        eval_mode="adoption",
        protocol="single",
        stress_grid=False,
        commission_bps=None,
        slippage_bps=None,
        participation=None,
        forensics=False,
    )
    result = cmd_backtest(args)
    assert result in (0, 1)
    assert args.model == STICKY_MOM60_RAW


def test_championship_constants_use_semantic_ids() -> None:
    from src.cli.constants import ANCHOR_STRATEGY, CHAMPION_STRATEGY
    from src.strategies.ids import STICKY_IMPULSE_CRASH, STICKY_MOM60_RAW

    assert CHAMPION_STRATEGY == STICKY_MOM60_RAW
    assert ANCHOR_STRATEGY == STICKY_IMPULSE_CRASH
    assert "." in CHAMPION_STRATEGY
    assert not CHAMPION_STRATEGY.startswith("P")


def test_build_parser_subcommand_surface_unchanged() -> None:
    from src.cli.parser import SUBCOMMANDS, build_parser

    parser = build_parser()
    expected = {
        "config-check",
        "calendar",
        "ingest",
        "normalize",
        "universe",
        "features",
        "backtest",
        "forensics",
        "loyo",
        "replay",
        "decide",
        "daily-refresh",
        "storage-migrate",
        "champion-research",
        "feasibility-audit",
    }
    subparsers = [a for a in parser._actions if hasattr(a, "choices") and isinstance(getattr(a, "choices", None), dict)]
    assert subparsers, "no subparser action found"
    assert set(subparsers[0].choices) == expected
    assert set(SUBCOMMANDS) == expected

    args = parser.parse_args(["backtest", "--model", "sticky.mom60_raw", "--start", "2018-01-02", "--end", "2026-08-27"])
    assert args.model == "sticky.mom60_raw"
    assert args.leverage_scenario == "aggressive"
    assert args.eval_mode == "adoption"
