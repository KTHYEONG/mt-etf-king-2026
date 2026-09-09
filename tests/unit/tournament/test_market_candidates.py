from datetime import date, timedelta
from types import SimpleNamespace
from unittest.mock import patch

import polars as pl
import pytest


def test_market_candidates_by_session_fails_closed_on_minimal_panel() -> None:
    from src.tournament.attainability import _MARKET_CANDIDATES_CACHE, market_candidates_by_session

    _MARKET_CANDIDATES_CACHE.clear()

    # Given: the minimal schema used by legacy unit tests (no name/trading_value/is_tradable)
    sessions = [date(2026, 1, 5), date(2026, 1, 6)]
    minimal = pl.DataFrame(
        {"date": sessions, "ticker": ["A", "A"], "open": [1.0, 1.0], "close": [1.0, 1.0]},
        schema={"date": pl.Date, "ticker": pl.String, "open": pl.Float64, "close": pl.Float64},
    )

    # Then: fail-closed to an empty map, never a widened universe (R4)
    assert market_candidates_by_session(sessions, minimal) == {}
    # Then: a non-DataFrame panel is equally fail-closed
    assert market_candidates_by_session(sessions, object()) == {}
    # Then: no sessions short-circuits without touching the cache (R5)
    assert market_candidates_by_session([], minimal) == {}
    assert _MARKET_CANDIDATES_CACHE == {}


def test_market_candidates_by_session_memoizes_on_content_key() -> None:
    from src.tournament.attainability import _MARKET_CANDIDATES_CACHE, market_candidates_by_session

    _MARKET_CANDIDATES_CACHE.clear()

    # Given: a full-schema panel that the oracle builder can process
    sessions = [date(2026, 1, 5) + timedelta(days=i) for i in range(8)]
    rows: list[dict[str, object]] = [
        {"date": day, "ticker": "111", "name": "KODEX 반도체", "open": 100.0, "trading_value": 200_000_000.0, "is_tradable": True}
        for day in sessions
    ]
    panel = pl.DataFrame(
        rows,
        schema={"date": pl.Date, "ticker": pl.String, "name": pl.String, "open": pl.Float64, "trading_value": pl.Float64, "is_tradable": pl.Boolean},
    )

    # When: called twice with identical inputs
    with patch(
        "src.research.executable_oracle.market_wide_session_candidates",
        return_value=dict.fromkeys(sessions, ()),
    ) as spy:
        first = market_candidates_by_session(sessions, panel)
        second = market_candidates_by_session(sessions, panel)

    # Then: the oracle was built exactly once and both calls agree
    assert spy.call_count == 1
    assert first == second

    # Then: the memo key is content-derived, never an object identity (R5)
    assert list(_MARKET_CANDIDATES_CACHE) == [(sessions[0], sessions[-1], len(sessions), panel.height)]
    assert id(panel) not in {part for key in _MARKET_CANDIDATES_CACHE for part in key if isinstance(part, int)}


def test_backtest_attainability_payload_emits_market_fields_from_full_panel() -> None:
    from src.tournament.attainability import _MARKET_CANDIDATES_CACHE, backtest_attainability_payload

    _MARKET_CANDIDATES_CACHE.clear()

    # Given: 70 sessions so the 60-session history floor is cleared near the tail
    sessions = [date(2026, 1, 5) + timedelta(days=i) for i in range(70)]
    rows: list[dict[str, object]] = []
    for day in sessions:
        for ticker, name in (("111", "KODEX 반도체"), ("555", "KODEX 2차전지")):
            rows.append({"date": day, "ticker": ticker, "name": name, "open": 100.0, "close": 100.0, "trading_value": 200_000_000.0, "is_tradable": True})
    panel = pl.DataFrame(
        rows,
        schema={"date": pl.Date, "ticker": pl.String, "name": pl.String, "open": pl.Float64, "close": pl.Float64, "trading_value": pl.Float64, "is_tradable": pl.Boolean},
    )
    # Given: the model scored only one of the two market-eligible tickers
    shared_cache = SimpleNamespace(
        scores={day: {"111": 1.0} for day in sessions},
        universes={},
        open_map={day: {"111": 100.0, "555": 100.0} for day in sessions},
    )

    # When
    payload = backtest_attainability_payload(
        calendar=SimpleNamespace(sessions=lambda start, end: list(sessions)),
        panel=panel,
        engine=None,
        model=None,
        case_config=SimpleNamespace(start=sessions[0], end=sessions[-1]),
        rolling=SimpleNamespace(returns=[0.0] * len(sessions)),
        horizon=2,
        shared_cache=shared_cache,
    )

    # Then: the model-conditional family is untouched (R6/R8)
    assert payload["breadth_mean"] == pytest.approx(1.0)

    # Then: the market family is emitted from the market-wide universe
    # 67 windows are emitted; 7 of them (decision index 60..66) see both eligible tickers
    assert payload["breadth_mean_market"] == pytest.approx(14.0 / 67.0)
    assert sorted(payload["attainability_market"]) == ["0.3", "0.4", "0.5", "0.6"]
    assert sorted(payload["capture_market"]) == ["0.3", "0.4", "0.5", "0.6"]
    assert sorted(payload["n_attainable_market"]) == ["0.3", "0.4", "0.5", "0.6"]

    # Then: win_bar is now computed on the market basis (R7)
    assert payload["win_bar_is_production_gate"] is False


def test_backtest_attainability_payload_market_fields_degrade_on_minimal_panel() -> None:
    from src.tournament.attainability import _MARKET_CANDIDATES_CACHE, backtest_attainability_payload

    _MARKET_CANDIDATES_CACHE.clear()

    # Given: the legacy minimal panel schema (no name/trading_value/is_tradable)
    sessions = [date(2026, 1, 5), date(2026, 1, 6), date(2026, 1, 7), date(2026, 1, 8)]
    panel = pl.DataFrame({"date": sessions}, schema={"date": pl.Date})
    shared_cache = SimpleNamespace(
        scores={day: {"A": 1.0, "B": 0.5} for day in sessions},
        universes={},
        open_map={
            sessions[0]: {"A": 100.0, "B": 100.0},
            sessions[1]: {"A": 100.0, "B": 100.0},
            sessions[2]: {"A": 120.0, "B": 110.0},
            sessions[3]: {"A": 150.0, "B": 110.0},
        },
    )

    # When
    payload = backtest_attainability_payload(
        calendar=SimpleNamespace(sessions=lambda start, end: list(sessions)),
        panel=panel,
        engine=None,
        model=None,
        case_config=SimpleNamespace(start=sessions[0], end=sessions[-1]),
        rolling=SimpleNamespace(returns=[0.60, 0.05, 0.0, 0.0]),
        horizon=2,
        shared_cache=shared_cache,
    )

    # Then: model-conditional family still computed from the session cache (R6)
    assert payload["breadth_mean"] == pytest.approx(2.0)
    assert payload["n_attainable"]["0.4"] == 1

    # Then: market family degrades rather than raising (R4)
    assert payload["breadth_mean_market"] == pytest.approx(0.0)
    assert payload["n_attainable_market"] == {"0.3": 0, "0.4": 0, "0.5": 0, "0.6": 0}
    assert payload["capture_market"] == {"0.3": None, "0.4": None, "0.5": None, "0.6": None}
