def test_house_money_should_cash_does_not_freeze_early() -> None:
    from datetime import date

    from src.tournament.policy import house_money_should_cash, remaining_sessions

    assert house_money_should_cash(0.80, 20, 0.50, 5) is False
    assert house_money_should_cash(0.50, 6, 0.50, 5) is False
    assert house_money_should_cash(0.80, 5, 0.50, 5) is True
    assert house_money_should_cash(0.80, 0, 0.50, 5) is True
    assert house_money_should_cash(0.20, 1, 0.50, 5) is False
    assert house_money_should_cash(float('nan'), 1, 0.50, 5) is False
    assert house_money_should_cash(0.80, 1, float('nan'), 5) is False
    assert remaining_sessions(date(2026, 10, 7), date(2026, 11, 13), None) == 10**9

    class _Cal:
        def next_session(self, day: date, offset: int = 1) -> date:
            from datetime import timedelta
            return day + timedelta(days=offset)

        def session_count(self, start: date, end: date) -> int:
            return (end - start).days + 1

    rem = remaining_sessions(date(2026, 11, 13), date(2026, 11, 13), _Cal())
    assert rem == 0


def test_house_money_ratchet_lets_continuation_past_arm() -> None:
    from src.tournament.distribution import (
        championship_lock_returns,
        continuation_capture,
        house_money_ratchet_returns,
        overlay_right_tail_stats,
    )

    cont = [0.60] + [0.10] * 7
    freeze = championship_lock_returns(cont, 8, 0.50, 0.0)
    ratchet = house_money_ratchet_returns(cont, 8, 0.50, 2)
    eq = 1.0
    for r in cont:
        eq *= 1.0 + r
    unlocked = [eq - 1.0]
    assert abs(freeze[0] - 0.60) < 1e-9
    assert ratchet[0] > 1.50
    assert abs(ratchet[0] - (1.60 * (1.10 ** 5) - 1.0)) < 1e-9
    assert unlocked[0] > ratchet[0]
    stats_f = overlay_right_tail_stats(freeze)
    stats_r = overlay_right_tail_stats(ratchet)
    assert stats_f['p_gt_80'] == 0.0
    assert stats_r['p_gt_80'] == 1.0
    cap = continuation_capture(unlocked, freeze, ratchet, 0.50, eps=0.10)
    assert cap > 0.9


def test_house_money_ratchet_emergency_floor_and_never_arm() -> None:
    from src.tournament.distribution import championship_lock_returns, house_money_ratchet_returns

    crash = [0.60, -0.40, 0.10, 0.0, 0.0, 0.0, 0.0, 0.0]
    rt = house_money_ratchet_returns(crash, 8, 0.50, 0)
    fz = championship_lock_returns(crash, 8, 0.50, 0.0)
    assert abs(rt[0] - 0.50) < 1e-9
    assert fz[0] + 1e-12 >= 0.50
    never = [0.01] * 8
    assert house_money_ratchet_returns(never, 8, 0.50, 2) == championship_lock_returns(never, 8, 0.50, 0.0)
    assert house_money_ratchet_returns([], 36, 0.50, 5) == []
    assert house_money_ratchet_returns([0.01] * 3, 0, 0.50, 5) == []
    assert house_money_ratchet_returns([0.01] * 3, 3, 0.0, 5) == []
    assert house_money_ratchet_returns([0.01] * 3, 3, float('nan'), 5) == []


def test_evaluate_p25_adoption_gates_rejects_right_tail_censor() -> None:
    from src.tournament.distribution import evaluate_p24_adoption_gates, evaluate_p25_adoption_gates

    censored = evaluate_p25_adoption_gates(
        0.50, 0.50, 0.0, 0.0, 0.40, 0.0, 0.03, 0.90
    )
    assert censored[0] == 'FAIL'
    assert 'right_tail_censored' in censored[1]
    assert 'continuation' in censored[1]
    ok = evaluate_p25_adoption_gates(
        1.57, 0.60, 0.40, 0.0, 0.40, 0.97, 0.03, 0.90
    )
    assert ok[0] == 'PASS'
    assert ok[1] == []
    ruin = evaluate_p25_adoption_gates(
        1.57, 0.60, 0.40, 0.0, 0.40, 0.97, 0.08, 0.90
    )
    assert 'ruin' in ruin[1]
    vehicle = evaluate_p25_adoption_gates(
        1.57, 0.60, 0.40, 0.0, 0.40, 0.97, 0.03, 0.0
    )
    assert 'vehicle_activity' in vehicle[1]
    p24 = evaluate_p24_adoption_gates(0.10, 0.05, 0.09, 0.05, 0.01, 0.02, -0.20, -0.25, 0.90)
    assert 'p_gt_50' in p24[1]
    assert 'p_gt_50' not in ok[1]
    assert 'p_gt_40' not in ok[1]
