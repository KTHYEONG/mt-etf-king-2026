from __future__ import annotations

from datetime import date
from pathlib import Path

from src.core.calendar import TradingCalendar
from src.universe.tournament import TournamentRules


def test_scenario_04_05_tournament_rules() -> None:
    rules = TournamentRules.from_yaml(Path("configs/tournament.yaml"))
    assert rules.start_date == date(2026, 9, 21)
    assert rules.end_date == date(2026, 11, 13)
    assert rules.initial_capital == 1_000_000_000
    assert rules.sponsor_etf_only is True
    assert rules.leverage_allowed is True
    assert rules.inverse_allowed is True
    assert rules.manifest_path is None
    assert rules.horizon_sessions(TradingCalendar()) == 36
    assert rules.scenarios_for("leverage_allowed") == (True,)




def test_tournament_rules_from_yaml_uses_cached_loader() -> None:
    from src.universe.tournament import TournamentRules, _load_tournament_yaml

    # Given: a clean cache
    _load_tournament_yaml.cache_clear()
    path = Path("configs/tournament.yaml")

    # When: from_yaml is called twice with the same path
    r1 = TournamentRules.from_yaml(path)
    info_after_first = _load_tournament_yaml.cache_info()
    r2 = TournamentRules.from_yaml(path)
    info_after_second = _load_tournament_yaml.cache_info()

    # Then: second call is served from cache, not a fresh file read
    assert info_after_first.misses == 1
    assert info_after_second.hits == 1
    assert info_after_second.misses == 1
    # And: parsed content is unaffected by caching
    assert r1.start_date == r2.start_date
    assert r1.initial_capital == r2.initial_capital
    assert r1.sponsor_etf_only == r2.sponsor_etf_only
    assert r1.stress_grid == r2.stress_grid
    # And: caching applies only to the raw YAML parse, not to the constructed
    # dataclass -- from_yaml still builds a fresh TournamentRules each call
    assert r1 is not r2



import pytest


def test_tournament_rules_from_yaml_missing_file_raises_every_call() -> None:
    from src.universe.tournament import TournamentRules, _load_tournament_yaml

    _load_tournament_yaml.cache_clear()
    bogus = Path("configs/__does_not_exist__.yaml")

    with pytest.raises(FileNotFoundError):
        TournamentRules.from_yaml(bogus)
    with pytest.raises(FileNotFoundError):
        TournamentRules.from_yaml(bogus)

    # And: the failing path was never cached as a hit
    assert _load_tournament_yaml.cache_info().hits == 0

