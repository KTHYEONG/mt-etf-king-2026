"""Daily-refresh registration in the main dispatch registry."""

from __future__ import annotations


def test_main_registry_routes_daily_refresh() -> None:
    from src.cli.commands.pipeline import cmd_daily_refresh
    from src.cli.main import SUBCOMMANDS

    assert SUBCOMMANDS["daily-refresh"] is cmd_daily_refresh


def test_main_and_parser_registries_agree_on_daily_refresh() -> None:
    from src.cli.main import SUBCOMMANDS as MAIN_SUBCOMMANDS
    from src.cli.parser import SUBCOMMANDS

    assert "daily-refresh" in SUBCOMMANDS
    assert "daily-refresh" in MAIN_SUBCOMMANDS
