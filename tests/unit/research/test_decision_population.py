from datetime import date

import polars as pl
import pytest

from src.research.decision_population import PopulationError, WindowPolicy, build_decision_population


def test_build_decision_population_executable_count_formula() -> None:
    sessions = [date(2026, 1, d) for d in (2, 5, 6, 7, 8, 9)]
    panel = pl.DataFrame({"date": sessions, "ticker": ["X"] * len(sessions)})
    pop = build_decision_population(calendar_sessions=sessions, panel=panel, horizon=2, policy=WindowPolicy.EXECUTABLE)
    assert pop.n_windows == len(sessions) - 2 - 1
    w0 = pop.windows[0]
    assert w0.decision_date == sessions[0]
    assert w0.entry_date == sessions[1]
    assert w0.exit_date == sessions[1 + 2]
    assert w0.return_session_ids == (sessions[1], sessions[2])
    assert all(len(w.return_session_ids) == 2 for w in pop.windows)


def test_build_decision_population_calendar_naive_keeps_phantoms() -> None:
    calendar = [date(2026, 1, d) for d in (2, 5, 6, 7, 8)]
    panel = pl.DataFrame({"date": [date(2026, 1, 2), date(2026, 1, 5), date(2026, 1, 8)], "ticker": ["X", "X", "X"]})
    pop = build_decision_population(calendar_sessions=calendar, panel=panel, horizon=2, policy=WindowPolicy.CALENDAR_NAIVE)
    assert date(2026, 1, 6) in pop.sessions
    assert date(2026, 1, 7) in pop.sessions
    assert pop.n_windows == len(calendar) - 2 + 1
    assert pop.windows[0].decision_date == pop.windows[0].entry_date == calendar[0]


def test_build_decision_population_panel_naive_drops_phantoms() -> None:
    calendar = [date(2026, 1, d) for d in (2, 5, 6, 7)]
    panel = pl.DataFrame({"date": [date(2026, 1, 2), date(2026, 1, 5), date(2026, 1, 7)], "ticker": ["X", "X", "X"]})
    pop = build_decision_population(calendar_sessions=calendar, panel=panel, horizon=2, policy=WindowPolicy.PANEL_NAIVE)
    assert date(2026, 1, 6) not in pop.sessions
    assert pop.n_windows == len(pop.sessions) - 2 + 1
    reasons = {e.reason for e in pop.exclusions}
    assert "PHANTOM_SESSION" in reasons
    assert date(2026, 1, 6) in {e.date for e in pop.exclusions}


def test_build_decision_population_fail_closed_empty_or_bad_horizon() -> None:
    sessions = [date(2026, 1, 2), date(2026, 1, 5)]
    panel = pl.DataFrame({"date": sessions, "ticker": ["X", "X"]})
    with pytest.raises(PopulationError):
        build_decision_population(calendar_sessions=sessions, panel=panel, horizon=0, policy=WindowPolicy.EXECUTABLE)
    with pytest.raises(PopulationError):
        build_decision_population(calendar_sessions=[], panel=panel, horizon=2, policy=WindowPolicy.EXECUTABLE)
    with pytest.raises(PopulationError):
        build_decision_population(calendar_sessions=sessions, panel=pl.DataFrame({"date": []}), horizon=2, policy=WindowPolicy.EXECUTABLE)
