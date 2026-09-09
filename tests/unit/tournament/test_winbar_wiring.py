from datetime import date
from types import SimpleNamespace

import polars as pl
import pytest


def test_backtest_attainability_payload_emits_winbar_keys() -> None:
    from src.tournament.attainability import backtest_attainability_payload
    from src.tournament.winbar import WIN_BAR_ORACLE_RATIO, WIN_BAR_RANK

    # Given: 4-session grid backed by a real panel so resolve_session_grid keeps every session
    sessions = [date(2026, 1, 5), date(2026, 1, 6), date(2026, 1, 7), date(2026, 1, 8)]
    panel = pl.DataFrame({"date": sessions}, schema={"date": pl.Date})
    shared_cache = SimpleNamespace(
        scores={s: {"A": 1.0, "B": 0.5} for s in sessions},
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

    # Then: pre-existing attainability keys are untouched
    assert payload["breadth_mean"] == pytest.approx(2.0, abs=1e-12)
    assert payload["n_attainable"]["0.4"] == 1

    # Then: winbar keys are merged in from the market-wide universe (R7).
    # The minimal panel has no oracle-eligible column, so the market map is
    # fail-closed to {} and no win-bar window clears the evidence floor.
    assert payload["win_bar_rank"] == WIN_BAR_RANK
    assert payload["win_bar_is_production_gate"] is False
    assert payload["n_win_bar_windows"]["ratio"] == 0
    assert payload["n_win_bar_windows"]["rank"] == 0
    assert payload["win_bar_median"]["rank"] is None
    assert payload["win_bar_median"]["ratio"] is None
