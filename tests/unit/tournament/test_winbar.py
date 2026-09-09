from datetime import date

import pytest


def test_window_win_bars_emits_rank_and_ratio_bars() -> None:
    from src.tournament.winbar import window_win_bars

    # Given: 4 sessions, horizon 2 -> only i=0 is emittable (entry=1, exit=3)
    sessions = [date(2026, 1, 5), date(2026, 1, 6), date(2026, 1, 7), date(2026, 1, 8)]
    open_by_session = [
        {"A": 100.0, "B": 100.0, "C": 100.0},
        {"A": 100.0, "B": 100.0, "C": 100.0},
        {"A": 150.0, "B": 120.0, "C": 105.0},
        {"A": 200.0, "B": 150.0, "C": 110.0},
    ]
    candidates = {sessions[0]: ["A", "B", "C"]}

    # When
    bars = window_win_bars(open_by_session, sessions, candidates, horizon=2, rank=2, oracle_ratio=0.621)

    # Then: returns are A=+100%, B=+50%, C=+10% -> rank2 bar = 0.50, ratio bar = 1.00 * 0.621
    assert len(bars) == 1
    assert bars[0].window_start == sessions[1]
    assert bars[0].breadth == 3
    assert bars[0].bar_rank == pytest.approx(0.50, abs=1e-12)
    assert bars[0].bar_ratio == pytest.approx(0.621, abs=1e-12)

def test_window_win_bars_rank_bar_none_when_breadth_below_rank() -> None:
    from src.tournament.winbar import window_win_bars

    # Given: B has a zero exit open -> excluded; only A contributes
    sessions = [date(2026, 1, 5), date(2026, 1, 6), date(2026, 1, 7), date(2026, 1, 8)]
    open_by_session = [
        {"A": 100.0, "B": 100.0},
        {"A": 100.0, "B": 100.0},
        {"A": 150.0, "B": 120.0},
        {"A": 200.0, "B": 0.0},
    ]
    candidates = {sessions[0]: ["A", "B"]}

    # When
    bars = window_win_bars(open_by_session, sessions, candidates, horizon=2, rank=2, oracle_ratio=0.5)

    # Then
    assert len(bars) == 1
    assert bars[0].breadth == 1
    assert bars[0].bar_rank is None
    assert bars[0].bar_ratio == pytest.approx(0.50, abs=1e-12)

def test_window_win_bars_rejects_invalid_rank_and_ratio() -> None:
    from src.tournament.winbar import window_win_bars

    sessions = [date(2026, 1, 5), date(2026, 1, 6), date(2026, 1, 7), date(2026, 1, 8)]
    open_by_session = [{"A": 100.0}, {"A": 100.0}, {"A": 150.0}, {"A": 200.0}]
    candidates = {sessions[0]: ["A"]}

    # Then: fail-closed on invalid rank
    with pytest.raises(ValueError, match=r".*"):
        window_win_bars(open_by_session, sessions, candidates, horizon=2, rank=0)

    # Then: fail-closed on ratio outside (0.0, 1.0]
    with pytest.raises(ValueError, match=r".*"):
        window_win_bars(open_by_session, sessions, candidates, horizon=2, oracle_ratio=0.0)
    with pytest.raises(ValueError, match=r".*"):
        window_win_bars(open_by_session, sessions, candidates, horizon=2, oracle_ratio=1.5)

    # Then: non-positive horizon is a no-op, not an error
    assert window_win_bars(open_by_session, sessions, candidates, horizon=0) == ()
    assert window_win_bars(open_by_session, [], candidates, horizon=2) == ()

def test_win_bar_exceedance_excludes_none_bars_and_enforces_floor() -> None:
    from src.tournament.winbar import WindowWinBar, win_bar_exceedance

    # Given: 3 windows, the third has no identifiable bar
    d = [date(2026, 1, i) for i in (5, 6, 7)]
    bars = [
        WindowWinBar(window_start=d[0], breadth=6, bar_rank=0.50, bar_ratio=0.62),
        WindowWinBar(window_start=d[1], breadth=6, bar_rank=0.20, bar_ratio=0.30),
        WindowWinBar(window_start=d[2], breadth=0, bar_rank=None, bar_ratio=None),
    ]
    returns = [0.60, 0.10, 0.90]

    # When
    rank_rate, rank_n = win_bar_exceedance(returns, bars, model="rank", min_bar_windows=1)
    ratio_rate, ratio_n = win_bar_exceedance(returns, bars, model="ratio", min_bar_windows=1)
    blocked, blocked_n = win_bar_exceedance(returns, bars, model="rank", min_bar_windows=5)

    # Then: window 3 is excluded from both numerator and denominator
    assert rank_n == 2
    assert rank_rate == pytest.approx(0.5, abs=1e-12)
    assert ratio_n == 2
    assert ratio_rate == pytest.approx(0.0, abs=1e-12)
    assert blocked is None and blocked_n == 2

    # Then: unknown model is fail-closed
    with pytest.raises(ValueError, match=r".*"):
        win_bar_exceedance(returns, bars, model="oracle")

def test_build_winbar_summary_emits_both_models() -> None:
    from src.tournament.winbar import build_winbar_summary

    # Given: one emittable window, returns A=+100%, B=+50%, C=+10%
    sessions = [date(2026, 1, 5), date(2026, 1, 6), date(2026, 1, 7), date(2026, 1, 8)]
    open_map = {
        sessions[0]: {"A": 100.0, "B": 100.0, "C": 100.0},
        sessions[1]: {"A": 100.0, "B": 100.0, "C": 100.0},
        sessions[2]: {"A": 150.0, "B": 120.0, "C": 105.0},
        sessions[3]: {"A": 200.0, "B": 150.0, "C": 110.0},
    }
    candidates = {sessions[0]: ["A", "B", "C"]}

    # When: strategy returned +60% on that window
    payload = build_winbar_summary(
        sessions=sessions,
        open_map=open_map,
        candidates_by_session=candidates,
        window_returns=[0.60, 0.0, 0.0, 0.0],
        horizon=2,
        rank=2,
        oracle_ratio=0.621,
        min_bar_windows=1,
    )

    # Then: exact schema (R12)
    assert set(payload) == {
        "win_bar_exceedance",
        "n_win_bar_windows",
        "win_bar_median",
        "win_bar_rank",
        "win_bar_oracle_ratio",
        "win_bar_is_production_gate",
    }
    # Then: 0.60 clears the rank bar 0.50 but not the ratio bar 0.621 (R11 - both models emitted)
    assert payload["win_bar_exceedance"]["rank"] == pytest.approx(1.0, abs=1e-12)
    assert payload["win_bar_exceedance"]["ratio"] == pytest.approx(0.0, abs=1e-12)
    assert payload["n_win_bar_windows"] == {"rank": 1, "ratio": 1}
    assert payload["win_bar_median"]["rank"] == pytest.approx(0.50, abs=1e-12)
    assert payload["win_bar_median"]["ratio"] == pytest.approx(0.621, abs=1e-12)
    assert payload["win_bar_rank"] == 2
    assert payload["win_bar_oracle_ratio"] == pytest.approx(0.621, abs=1e-12)
    assert payload["win_bar_is_production_gate"] is False

def test_build_winbar_summary_median_survives_evidence_floor() -> None:
    from src.tournament.winbar import build_winbar_summary

    sessions = [date(2026, 1, 5), date(2026, 1, 6), date(2026, 1, 7), date(2026, 1, 8)]
    open_map = {
        sessions[0]: {"A": 100.0, "B": 100.0},
        sessions[1]: {"A": 100.0, "B": 100.0},
        sessions[2]: {"A": 150.0, "B": 120.0},
        sessions[3]: {"A": 200.0, "B": 150.0},
    }
    candidates = {sessions[0]: ["A", "B"]}

    # When: evidence floor of 30 windows cannot be met by a single window
    payload = build_winbar_summary(
        sessions=sessions,
        open_map=open_map,
        candidates_by_session=candidates,
        window_returns=[0.60, 0.0, 0.0, 0.0],
        horizon=2,
        rank=2,
        oracle_ratio=0.5,
        min_bar_windows=30,
    )

    # Then: exceedance withheld, medians still reported
    assert payload["win_bar_exceedance"]["rank"] is None
    assert payload["win_bar_exceedance"]["ratio"] is None
    assert payload["n_win_bar_windows"] == {"rank": 1, "ratio": 1}
    assert payload["win_bar_median"]["rank"] == pytest.approx(0.50, abs=1e-12)
    assert payload["win_bar_median"]["ratio"] == pytest.approx(0.50, abs=1e-12)
