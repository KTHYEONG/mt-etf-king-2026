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
        "storage-migrate",
    }
    assert expected == choices


def test_cli_backtest_resolves_semantic_model_flag() -> None:
    import argparse

    from src.cli.commands.backtest import cmd_backtest
    from src.strategies.ids import STICKY_MOM60_RAW
    from src.strategies.registry import branch_model_key, resolve_strategy_id

    assert resolve_strategy_id("P27") == STICKY_MOM60_RAW
    assert branch_model_key(STICKY_MOM60_RAW) == "P27"
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
    assert args.model == "P27"


def test_championship_constants_use_semantic_ids() -> None:
    from src.cli.constants import ANCHOR_STRATEGY, CHAMPION_STRATEGY
    from src.strategies.ids import STICKY_IMPULSE_CRASH, STICKY_MOM60_RAW

    assert CHAMPION_STRATEGY == STICKY_MOM60_RAW
    assert ANCHOR_STRATEGY == STICKY_IMPULSE_CRASH
    assert "." in CHAMPION_STRATEGY
    assert not CHAMPION_STRATEGY.startswith("P")
