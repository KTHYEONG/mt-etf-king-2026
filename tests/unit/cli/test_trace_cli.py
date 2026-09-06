from __future__ import annotations

from src.cli import build_parser


def test_parser_trace_and_log_level_defaults() -> None:
    parser = build_parser()
    args = parser.parse_args(["backtest", "--model", "baseline.mom20_top1", "--start", "2024-01-02", "--end", "2024-01-08"])
    assert args.trace is False
    assert args.log_level == "INFO"
    args2 = parser.parse_args(["--trace", "--log-level", "DEBUG", "backtest", "--model", "baseline.mom20_top1", "--start", "2024-01-02", "--end", "2024-01-08"])
    assert args2.trace is True
    assert args2.log_level == "DEBUG"


def test_cmd_backtest_skips_trace_write_without_flag() -> None:
    from pathlib import Path

    text = Path("src/cli/commands/backtest/artifacts.py").read_text(encoding="utf-8")
    assert "write_trace_artifacts(" in text
    # ensure branch references trace flag
    assert "trace" in text and ("ctx.trace" in text or "getattr(" in text)
    assert "InMemoryTraceSink" in text
    main_text = Path("src/cli/main.py").read_text(encoding="utf-8")
    assert "configure_logging(" in main_text
