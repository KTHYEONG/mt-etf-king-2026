def test_championship_lock_returns_freeze_when_trail_zero() -> None:
    from src.tournament.distribution import championship_lock_returns, locked_window_returns

    rets = [0.25, 0.25, -0.50]
    a = championship_lock_returns(rets, 3, 0.40, 0.0)
    b = locked_window_returns(rets, 3, 0.40)
    assert a == b
    assert a[0] >= 0.40 - 1e-12
    gap = [0.45, 0.10, -0.20]
    freeze50 = championship_lock_returns(gap, 3, 0.50, 0.0)
    freeze40 = championship_lock_returns(gap, 3, 0.40, -1.0)
    assert freeze50[0] + 1e-12 >= 0.50
    assert freeze50[0] >= 0.4782
    assert freeze40[0] < 0.4782
    assert championship_lock_returns(gap, 3, 0.50, float('nan')) == locked_window_returns(gap, 3, 0.50)


def test_championship_lock_returns_hysteresis_skips_arm_touch() -> None:
    from src.tournament.distribution import championship_lock_returns, locked_window_returns

    rets = [0.40, 0.12, -0.18]
    freeze = locked_window_returns(rets, 3, 0.40)
    trail = championship_lock_returns(rets, 3, 0.40, 0.08)
    assert len(trail) == 1
    assert freeze[0] + 1e-12 >= 0.40
    assert freeze[0] < 0.50
    assert trail[0] > freeze[0] + 1e-9
    eq = 1.40 * 1.12
    stop = max(1.40, eq - 0.08)
    assert abs(trail[0] - (stop - 1.0)) < 1e-9 or trail[0] + 1e-12 >= 0.50


def test_championship_lock_returns_fail_closed() -> None:
    from src.tournament.distribution import championship_lock_returns

    assert championship_lock_returns([], 36, 0.50) == []
    assert championship_lock_returns([0.01, 0.01], 0, 0.50) == []
    assert championship_lock_returns([0.01] * 3, 3, 0.0) == []
    assert championship_lock_returns([0.01] * 3, 3, float('nan')) == []
    assert championship_lock_returns([0.01] * 3, 3, -0.1, 0.08) == []


def test_evaluate_p24_adoption_gates_requires_p50() -> None:
    from src.tournament.distribution import evaluate_adoption_gates, evaluate_p24_adoption_gates

    base_ok = evaluate_adoption_gates(0.10, 0.05, 0.09, 0.05, -0.20, -0.25, 0.90)
    assert base_ok[0] == 'PASS'
    fail50 = evaluate_p24_adoption_gates(0.10, 0.05, 0.09, 0.05, 0.01, 0.02, -0.20, -0.25, 0.90)
    assert fail50[0] == 'FAIL'
    assert 'p_gt_50' in fail50[1]
    pass50 = evaluate_p24_adoption_gates(0.10, 0.05, 0.09, 0.05, 0.06, 0.02, -0.20, -0.25, 0.90)
    assert pass50[0] == 'PASS'
    assert pass50[1] == []
    vehicle = evaluate_p24_adoption_gates(0.10, 0.05, 0.09, 0.05, 0.06, 0.02, -0.20, -0.25, 0.0)
    assert 'vehicle_activity' in vehicle[1]
