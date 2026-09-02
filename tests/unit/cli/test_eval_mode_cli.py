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


def test_cli_accepts_p13_model() -> None:
    parser = build_parser()
    args = parser.parse_args(["backtest", "--model", "P13", "--start", "2026-01-02", "--end", "2026-01-08"])
    assert args.model == "P13"
    assert args.eval_mode == "adoption"
    # wiring literal check
    import pathlib

    cli_text = pathlib.Path("src/cli/_impl.py").read_text(encoding="utf-8")
    assert 'model_key == "P13"' in cli_text
    assert "resolve_adoption_vehicle_rate" in cli_text
    assert "measure_vehicle_activity_from_session_cache" in cli_text


def test_cli_accepts_p14_model() -> None:
    parser = build_parser()
    args = parser.parse_args(["backtest", "--model", "P14", "--start", "2026-01-02", "--end", "2026-01-08"])
    assert args.model == "P14"
    assert args.eval_mode == "adoption"
    import pathlib

    cli_text = pathlib.Path("src/cli/_impl.py").read_text(encoding="utf-8")
    assert 'model_key == "P14"' in cli_text
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
