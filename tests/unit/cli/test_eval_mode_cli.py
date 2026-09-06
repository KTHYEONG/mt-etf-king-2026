from __future__ import annotations

import pytest

from src.cli import build_parser


def test_cli_eval_mode_default_adoption_for_portfolio() -> None:
    parser = build_parser()
    args = parser.parse_args(["backtest", "--model", "portfolio.momentum_confidence", "--start", "2026-01-02", "--end", "2026-01-08"])
    assert hasattr(args, "eval_mode")
    assert args.eval_mode == "adoption"
    args2 = parser.parse_args(
        ["backtest", "--model", "portfolio.momentum_confidence", "--start", "2026-01-02", "--end", "2026-01-08", "--eval-mode", "operational"]
    )
    assert args2.eval_mode == "operational"
    args3 = parser.parse_args(
        ["backtest", "--model", "baseline.mom20_top1", "--start", "2026-01-02", "--end", "2026-01-08", "--eval-mode", "adoption"]
    )
    assert args3.eval_mode == "adoption"


def test_cli_accepts_p13_model() -> None:
    import inspect

    parser = build_parser()
    args = parser.parse_args(["backtest", "--model", "portfolio.leadership_confidence", "--start", "2026-01-02", "--end", "2026-01-08"])
    assert args.model == "portfolio.leadership_confidence"
    assert args.eval_mode == "adoption"
    # wiring literal check
    from src.cli.commands.backtest.families.portfolio import _hook_leadership_confidence

    cli_text = inspect.getsource(_hook_leadership_confidence)
    assert "resolve_adoption_vehicle_rate" in cli_text
    assert "vehicle_mult2_rate" in cli_text


def test_cli_accepts_p14_model() -> None:
    import inspect

    parser = build_parser()
    args = parser.parse_args(["backtest", "--model", "portfolio.lottery_exposure", "--start", "2026-01-02", "--end", "2026-01-08"])
    assert args.model == "portfolio.lottery_exposure"
    assert args.eval_mode == "adoption"
    from src.cli.commands.backtest.families.portfolio import _hook_lottery_exposure

    cli_text = inspect.getsource(_hook_lottery_exposure)
    assert "resolve_adoption_vehicle_rate" in cli_text
    assert "evaluate_adoption_gates" in cli_text
    assert "adoption_gate model=P14" in cli_text


@pytest.mark.parametrize(
    "scenario_id",
    ["test_cli_eval_mode_default_adoption_for_portfolio", "test_cli_accepts_p14_model"],
)
def test_scenario_wrapper(scenario_id: str) -> None:
    if scenario_id == "test_cli_eval_mode_default_adoption_for_portfolio":
        test_cli_eval_mode_default_adoption_for_portfolio()
    elif scenario_id == "test_cli_accepts_p14_model":
        test_cli_accepts_p14_model()
