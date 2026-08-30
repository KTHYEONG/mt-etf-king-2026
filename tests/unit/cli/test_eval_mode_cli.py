from __future__ import annotations

import pytest

from src.cli import build_parser


def test_cli_eval_mode_default_adoption_for_portfolio() -> None:
    parser = build_parser()
    args = parser.parse_args(["backtest", "--model", "P11", "--start", "2026-01-02", "--end", "2026-01-08"])
    assert hasattr(args, "eval_mode")
    assert args.eval_mode == "adoption"
    args2 = parser.parse_args(
        ["backtest", "--model", "P11", "--start", "2026-01-02", "--end", "2026-01-08", "--eval-mode", "operational"]
    )
    assert args2.eval_mode == "operational"
    args3 = parser.parse_args(
        ["backtest", "--model", "B1", "--start", "2026-01-02", "--end", "2026-01-08", "--eval-mode", "adoption"]
    )
    assert args3.eval_mode == "adoption"


@pytest.mark.parametrize("scenario_id", ["test_cli_eval_mode_default_adoption_for_portfolio"])
def test_scenario_wrapper(scenario_id: str) -> None:
    test_cli_eval_mode_default_adoption_for_portfolio()
