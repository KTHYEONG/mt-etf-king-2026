def test_locked_window_returns_freezes_at_lock() -> None:
    from src.backtest.metrics import window_returns
    from src.tournament.distribution import locked_window_returns

    rets = [0.25, 0.25, -0.50]
    unlocked = window_returns(rets, 3)
    locked = locked_window_returns(rets, 3, 0.40)
    assert len(locked) == 1
    assert unlocked[0] < 0.0
    assert locked[0] >= 0.40 - 1e-12
    assert abs(locked[0] - (1.25 * 1.25 - 1.0)) < 1e-12


def test_locked_window_returns_fail_closed_empty() -> None:
    from src.tournament.distribution import locked_window_returns

    assert locked_window_returns([], 36, 0.40) == []
    assert locked_window_returns([0.01, 0.01], 0, 0.40) == []
    assert locked_window_returns([0.01, 0.01, 0.01], 3, 0.0) == []
    assert locked_window_returns([0.01] * 3, 3, float("nan")) == []


def test_peak_lock_active_threshold() -> None:
    from src.tournament.policy import peak_lock_active

    assert peak_lock_active(1.40e9, 1.0e9, 0.40) is True
    assert peak_lock_active(1.39e9, 1.0e9, 0.40) is False
    assert peak_lock_active(1.0e9, 0.0, 0.40) is False
    assert peak_lock_active(float("nan"), 1.0e9, 0.40) is False


def test_locked_window_returns_lock50_preserves_championship_zone() -> None:
    from src.tournament.distribution import locked_window_returns
    from src.tournament.policy import peak_lock_active

    rets = [0.45, 0.10, -0.20]
    lock40 = locked_window_returns(rets, 3, 0.40)
    lock50 = locked_window_returns(rets, 3, 0.50)
    assert len(lock40) == 1 and len(lock50) == 1
    assert lock40[0] + 1e-12 >= 0.40
    assert lock40[0] < 0.4782
    assert lock50[0] + 1e-12 >= 0.50
    assert lock50[0] >= 0.4782
    assert peak_lock_active(1.50e9, 1.0e9, 0.50) is True
    assert peak_lock_active(1.49e9, 1.0e9, 0.50) is False
    assert peak_lock_active(1.40e9, 1.0e9, 0.40) is True
