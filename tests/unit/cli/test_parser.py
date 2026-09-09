# ruff: noqa
from __future__ import annotations


def test_parser_accepts_adaptive_specialists_mode() -> None:
    from src.cli.parser import build_parser

    args = build_parser().parse_args(["champion-research", "--start", "2018-01-02", "--end", "2026-08-27", "--candidate-mode", "adaptive_specialists"])
    assert args.candidate_mode == "adaptive_specialists"

def test_parser_accepts_explicit_frontier_artifact_paths() -> None:
    from src.cli.parser import build_parser
    args=build_parser().parse_args(["champion-research","--start","2018-01-02","--end","2026-08-27","--candidate-mode","capacity_frontier","--data-root","data","--output","results/championship_frontier"])
    assert args.candidate_mode == "capacity_frontier"
    assert args.data_root == "data"
    assert args.output == "results/championship_frontier"


def test_parser_daily_refresh_subcommand_defaults() -> None:
    from src.cli.parser import build_parser
    from src.cli.commands.pipeline import cmd_daily_refresh

    args = build_parser().parse_args(["daily-refresh"])

    assert args.dataset == "etf_daily"
    assert args.as_of is None
    assert args.lookback_days == 10
    assert args.decide is False
    assert args.output_dir == "results/decide_daily"
    assert args.func is cmd_daily_refresh


def test_parser_daily_refresh_accepts_overrides() -> None:
    from src.cli.parser import build_parser

    args = build_parser().parse_args(
        [
            "daily-refresh",
            "--dataset",
            "etf_daily",
            "--as-of",
            "2026-08-27",
            "--lookback-days",
            "3",
            "--decide",
            "--output-dir",
            "custom/dir",
        ]
    )

    assert args.dataset == "etf_daily"
    assert args.as_of == "2026-08-27"
    assert args.lookback_days == 3
    assert args.decide is True
    assert args.output_dir == "custom/dir"

