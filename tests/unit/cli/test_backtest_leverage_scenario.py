"""SCENARIO-10-07"""
from pathlib import Path

import pytest

from src.cli import build_parser
from src.tournament.harness import resolve_leverage_scenario


def test_SCENARIO_10_07_cli_leverage_scenario() -> None:
    parser = build_parser()
    args = parser.parse_args(["backtest", "--model", "baseline.mom20_top1", "--start", "2026-01-02", "--end", "2026-01-08"])
    assert args.leverage_scenario == "aggressive"
    args2 = parser.parse_args(
        ["backtest", "--model", "baseline.mom20_top1", "--start", "2026-01-02", "--end", "2026-01-08", "--leverage-scenario", "conservative"]
    )
    assert args2.leverage_scenario == "conservative"
    args3 = parser.parse_args(
        ["backtest", "--model", "baseline.mom20_top1", "--start", "2026-01-02", "--end", "2026-01-08", "--leverage-scenario", "rules"]
    )
    assert args3.leverage_scenario == "rules"
    assert resolve_leverage_scenario("aggressive", None) is True
    assert resolve_leverage_scenario("conservative", None) is False
    with pytest.raises(SystemExit):
        parser.parse_args(
            ["backtest", "--model", "baseline.mom20_top1", "--start", "2026-01-02", "--end", "2026-01-08", "--leverage-scenario", "invalid"]
        )
    import inspect

    import src.cli.commands.backtest._core as _core_mod
    from src.cli.commands.backtest import _prep as _prep_mod
    from src.cli.commands.backtest.families.portfolio import _hook_lottery_rebalance

    txt = inspect.getsource(_core_mod) + inspect.getsource(_prep_mod) + inspect.getsource(_hook_lottery_rebalance)
    assert "resolve_leverage_scenario" in txt
    assert "resolve_adoption_vehicle_rate" in txt
    assert "preflight_features_span_ok" in txt
    assert "v_rate = 0.30" not in txt
    assert "fallback ensures gate not blocked" not in txt


@pytest.mark.parametrize("scenario_id", ["SCENARIO-10-07"])
def test_SCENARIO_hyphen_wrapper(scenario_id: str) -> None:
    test_SCENARIO_10_07_cli_leverage_scenario()
