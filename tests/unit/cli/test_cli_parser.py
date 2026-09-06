"""P4 CLI decomposition: parser.py builds the mt-etf argument parser."""
import argparse

import src.cli.parser as parser_mod
from src.cli.parser import SUBCOMMANDS, build_parser


def test_p4_parser_subcommands_match_main_table() -> None:
    from src.cli.main import SUBCOMMANDS as MAIN_SUBCOMMANDS

    assert set(SUBCOMMANDS) == set(MAIN_SUBCOMMANDS)


def test_p4_parser_builds_and_routes_backtest() -> None:
    parser = build_parser()
    assert isinstance(parser, argparse.ArgumentParser)
    args = parser.parse_args(["backtest", "--model", "sticky.mom60_raw", "--start", "2018-01-02", "--end", "2018-03-30"])
    assert args.model == "sticky.mom60_raw"
    assert callable(getattr(args, "func", None))
    assert parser_mod.SUBCOMMANDS
