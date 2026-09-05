from datetime import date

import polars as pl


def test_resolve_session_grid_drops_phantom_sessions() -> None:
    from src.backtest.session_grid import resolve_session_grid

    # Given: calendar claims 2026-06-03 is a session but the panel observes nothing that day
    sessions = [date(2026, 6, 2), date(2026, 6, 3), date(2026, 6, 4)]
    panel = pl.DataFrame(
        {
            "date": [date(2026, 6, 2), date(2026, 6, 4), date(2026, 6, 5)],
            "ticker": ["122630", "122630", "122630"],
            "close": [100.0, 110.0, 120.0],
        }
    )

    # When
    grid = resolve_session_grid(sessions, panel)

    # Then: phantom dropped, non-calendar panel date not added
    assert grid.sessions == (date(2026, 6, 2), date(2026, 6, 4))
    assert grid.phantom == (date(2026, 6, 3),)
    assert date(2026, 6, 5) not in grid.sessions


def test_engine_run_excludes_phantom_sessions() -> None:
    from src.backtest.session_grid import resolve_session_grid

    # Given: the panel misses two mid-range calendar sessions
    calendar_sessions = [date(2026, 6, d) for d in (1, 2, 3, 4, 5)]
    panel = pl.DataFrame(
        {
            "date": [date(2026, 6, 1), date(2026, 6, 2), date(2026, 6, 4)],
            "ticker": ["X", "X", "X"],
            "close": [1.0, 1.0, 1.0],
            "open": [1.0, 1.0, 1.0],
        }
    )

    # When
    grid = resolve_session_grid(calendar_sessions, panel)

    # Then: engine-facing grid never contains a priceless session
    panel_dates = set(panel["date"].to_list())
    assert set(grid.sessions) <= panel_dates
    assert grid.phantom == (date(2026, 6, 3), date(2026, 6, 5))


def test_phantom_session_labels_reports_dropped_dates() -> None:
    from src.backtest.session_grid import phantom_session_labels

    sessions = [date(2026, 6, 2), date(2026, 6, 3), date(2026, 6, 4)]
    panel = pl.DataFrame(
        {
            "date": [date(2026, 6, 2), date(2026, 6, 4)],
            "ticker": ["122630", "122630"],
            "close": [100.0, 110.0],
        }
    )
    assert phantom_session_labels(sessions, panel) == ["2026-06-03"]


def test_resolve_session_grid_handles_invalid_min_rows() -> None:
    from src.backtest.session_grid import resolve_session_grid

    sessions = [date(2026, 6, 2)]
    panel = pl.DataFrame({"date": [date(2026, 6, 2)], "ticker": ["X"], "close": [1.0]})
    grid = resolve_session_grid(sessions, panel, min_rows="bad")  # type: ignore[arg-type]
    assert grid.sessions == (date(2026, 6, 2),)


def test_resolve_session_grid_tolerates_bad_panel_dates() -> None:
    from src.backtest.session_grid import resolve_session_grid

    sessions = [date(2026, 6, 2)]
    panel = pl.DataFrame({"date": ["bad"], "ticker": ["X"], "close": [1.0]})
    grid = resolve_session_grid(sessions, panel)
    assert grid.phantom == (date(2026, 6, 2),)


def test_phantom_session_labels_returns_empty_on_failure() -> None:
    from unittest.mock import patch

    from src.backtest.session_grid import phantom_session_labels

    with patch("src.backtest.session_grid.resolve_session_grid", side_effect=RuntimeError("boom")):
        assert phantom_session_labels([date(2026, 6, 2)], pl.DataFrame()) == []


def test_resolve_session_grid_swallows_panel_iteration_errors() -> None:
    from unittest.mock import patch

    from src.backtest.session_grid import resolve_session_grid

    sessions = [date(2026, 6, 2)]
    panel = pl.DataFrame({"date": [date(2026, 6, 2)], "ticker": ["X"], "close": [1.0]})
    with patch.object(pl.DataFrame, "__getitem__", side_effect=RuntimeError("boom")):
        grid = resolve_session_grid(sessions, panel)
    assert grid.phantom == (date(2026, 6, 2),)
