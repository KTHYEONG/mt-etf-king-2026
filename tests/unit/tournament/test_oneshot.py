def test_oneshot_anchor_starts_sep21_and_horizon_fit() -> None:
    from datetime import date, timedelta

    from src.tournament.distribution import oneshot_anchor_starts

    y2024 = [date(2024, 9, 20)] + [date(2024, 9, 23) + timedelta(days=i) for i in range(40)]
    y2025_short = [date(2025, 9, 22) + timedelta(days=i) for i in range(10)]
    sessions = y2024 + y2025_short
    starts = oneshot_anchor_starts(sessions, month=9, day=21, horizon=36)
    assert starts == (date(2024, 9, 23),)
    assert oneshot_anchor_starts([], month=9, day=21, horizon=36) == ()
    assert oneshot_anchor_starts(y2024, month=9, day=21, horizon=0) == ()


def test_oneshot_window_returns_compounds_and_skips_short() -> None:
    from datetime import date

    from src.tournament.distribution import oneshot_window_returns

    sessions = [date(2024, 9, 23), date(2024, 9, 24), date(2024, 9, 25)]
    daily = [0.10, 0.10, 0.10]
    rows = oneshot_window_returns(daily, sessions, (date(2024, 9, 23),), horizon=2)
    assert rows == ((2024, date(2024, 9, 23), 0.21),)
    assert oneshot_window_returns(daily, sessions, (date(2024, 9, 23),), horizon=4) == ()
    assert oneshot_window_returns([0.1], sessions, (date(2024, 9, 23),), horizon=2) == ()
    assert oneshot_window_returns(daily, sessions, (date(2024, 9, 30),), horizon=2) == ()
