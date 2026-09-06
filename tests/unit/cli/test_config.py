"""P4 CLI decomposition: config.py (config_check/calendar) moved verbatim from _impl.py."""
import argparse

from src.cli.commands.config import cmd_calendar


def test_p4_config_calendar_bad_date_returns_1() -> None:
    assert cmd_calendar(argparse.Namespace(start="bogus", end="2026-08-27")) == 1
