"""P4 CLI decomposition: main.py dispatches via SUBCOMMANDS table."""
from src.cli.main import SUBCOMMANDS, main


def test_p4_main_subcommands_cover_all_commands() -> None:
    assert set(SUBCOMMANDS) == {
        "config-check", "calendar", "ingest", "normalize", "universe",
        "features", "backtest", "forensics", "loyo", "replay", "decide",
        "storage-migrate", "champion-research",
    }


def test_p4_main_unknown_subcommand_returns_2() -> None:
    assert main(["no-such-command"]) == 2


def test_p4_main_no_args_returns_2() -> None:
    assert main([]) == 2
