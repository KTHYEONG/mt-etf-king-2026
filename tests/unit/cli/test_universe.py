"""P4 CLI decomposition: universe.py moved verbatim from _impl.py."""
import argparse

from src.cli.commands.universe import cmd_universe


def test_p4_universe_missing_date_returns_1() -> None:
    assert cmd_universe(argparse.Namespace()) == 1


def test_p4_universe_invalid_mode_returns_1() -> None:
    args = argparse.Namespace(date="2026-08-27", mode="bogus")
    assert cmd_universe(args) == 1
