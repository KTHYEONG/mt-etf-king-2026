from __future__ import annotations

from datetime import date
from pathlib import Path

from src.core.calendar import TradingCalendar
from src.universe.tournament import UNKNOWN, TournamentRules


def test_scenario_04_05_tournament_rules() -> None:
    """SCENARIO-04-05"""
    rules = TournamentRules.from_yaml(Path("configs/tournament.yaml"))
    assert rules.start_date == date(2026, 9, 21)
    assert rules.end_date == date(2026, 11, 13)
    assert rules.initial_capital == 1_000_000_000
    assert rules.sponsor_etf_only is True
    assert rules.leverage_allowed is UNKNOWN
    assert rules.horizon_sessions(TradingCalendar()) == 36
    assert rules.scenarios_for("leverage_allowed") == (True, False)
