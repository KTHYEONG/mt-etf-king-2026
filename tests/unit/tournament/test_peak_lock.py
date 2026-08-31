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
