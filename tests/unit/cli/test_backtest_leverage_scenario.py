"""SCENARIO-10-07"""
from pathlib import Path

import pytest

from src.cli import build_parser
from src.tournament.harness import resolve_leverage_scenario


def test_SCENARIO_10_07_cli_leverage_scenario() -> None:
    parser = build_parser()
    args = parser.parse_args(["backtest", "--model", "B1", "--start", "2026-01-02", "--end", "2026-01-08"])
    assert args.leverage_scenario == "aggressive"
    args2 = parser.parse_args(
        ["backtest", "--model", "B1", "--start", "2026-01-02", "--end", "2026-01-08", "--leverage-scenario", "conservative"]
    )
    assert args2.leverage_scenario == "conservative"
    args3 = parser.parse_args(
        ["backtest", "--model", "B1", "--start", "2026-01-02", "--end", "2026-01-08", "--leverage-scenario", "rules"]
    )
    assert args3.leverage_scenario == "rules"
    assert resolve_leverage_scenario("aggressive", None) is True
    assert resolve_leverage_scenario("conservative", None) is False
    with pytest.raises(SystemExit):
        parser.parse_args(
            ["backtest", "--model", "B1", "--start", "2026-01-02", "--end", "2026-01-08", "--leverage-scenario", "invalid"]
        )
    txt = Path("src/cli/_impl.py").read_text()
    assert "resolve_leverage_scenario" in txt
    assert "measure_vehicle_activity_from_allocate" in txt
    assert "measure_vehicle_activity_from_session_cache" in txt
    assert "resolve_adoption_vehicle_rate" in txt
    assert "preflight_features_span_ok" in txt
    assert "v_rate = 0.30" not in txt
    assert "fallback ensures gate not blocked" not in txt


@pytest.mark.parametrize("scenario_id", ["SCENARIO-10-07"])
def test_SCENARIO_hyphen_wrapper(scenario_id: str) -> None:
    test_SCENARIO_10_07_cli_leverage_scenario()
