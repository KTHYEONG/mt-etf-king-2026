def test_classify_activation_state_missing_or_nonfinite_is_uncertain() -> None:
    from datetime import date

    from src.research.activation_state import ActivationState, classify_activation_state

    decision = date(2020, 1, 10)
    prior = date(2020, 1, 2)
    empty = classify_activation_state(decision_date=decision, mom60_by_date={})
    assert empty.state is ActivationState.UNCERTAIN
    assert empty.mom60 is None
    assert empty.n_history == 0
    assert empty.q1 is None and empty.q2 is None
    absent = classify_activation_state(decision_date=decision, mom60_by_date={prior: 0.2}, min_history=1)
    assert absent.state is ActivationState.UNCERTAIN
    assert absent.mom60 is None
    assert absent.n_history == 1
    for bad in (None, float("nan"), float("inf"), float("-inf")):
        snap = classify_activation_state(decision_date=decision, mom60_by_date={decision: bad}, min_history=1)
        assert snap.state is ActivationState.UNCERTAIN
        assert snap.mom60 is None



def test_classify_activation_state_rejects_future_dates() -> None:
    from datetime import date

    import pytest

    from src.features.pit import PitViolationError
    from src.research.activation_state import classify_activation_state

    decision = date(2020, 1, 2)
    with pytest.raises(PitViolationError):
        classify_activation_state(
            decision_date=decision,
            mom60_by_date={decision: 0.1, date(2020, 1, 3): 0.2},
            min_history=1,
        )
    with pytest.raises(ValueError, match="min_history"):
        classify_activation_state(decision_date=decision, mom60_by_date={decision: 0.1}, min_history=0)



def test_classify_activation_state_insufficient_history_is_uncertain() -> None:
    from datetime import date

    from src.research.activation_state import ACTIVATION_TERCILE_MIN_HISTORY, ActivationState, classify_activation_state

    days = [date(2020, 1, d) for d in range(1, 10)]
    series = {day: 0.01 * idx for idx, day in enumerate(days)}
    snap = classify_activation_state(decision_date=days[-1], mom60_by_date=series, min_history=20)
    assert snap.state is ActivationState.UNCERTAIN
    assert snap.n_history == 9
    assert snap.q1 is None and snap.q2 is None
    assert ACTIVATION_TERCILE_MIN_HISTORY == 252
    default = classify_activation_state(decision_date=days[-1], mom60_by_date=series)
    assert default.state is ActivationState.UNCERTAIN
    assert default.n_history == 9



def test_classify_activation_state_expanding_tercile_on_off_middle() -> None:
    from datetime import date

    import pytest

    from src.research.activation_state import ActivationState, classify_activation_state

    def linear_quantile(values: list[float], q: float) -> float:
        ordered = sorted(values)
        pos = q * (len(ordered) - 1)
        lo = int(pos)
        hi = min(lo + 1, len(ordered) - 1)
        weight = pos - lo
        return ordered[lo] * (1.0 - weight) + ordered[hi] * weight

    days = [date(2020, 1, d) for d in (2, 3, 6, 7, 8, 9)]
    values = [0.0, 0.1, 0.2, 0.3, -0.5, 0.15]
    series = dict(zip(days, values, strict=True))
    on = classify_activation_state(decision_date=days[3], mom60_by_date={d: series[d] for d in days[:4]}, min_history=4)
    hist_on = values[:4]
    assert on.state is ActivationState.AGGRESSIVE_ON
    assert on.mom60 == 0.3
    assert on.n_history == 4
    assert on.q1 == pytest.approx(linear_quantile(hist_on, 1.0 / 3.0))
    assert on.q2 == pytest.approx(linear_quantile(hist_on, 2.0 / 3.0))
    off = classify_activation_state(decision_date=days[4], mom60_by_date={d: series[d] for d in days[:5]}, min_history=4)
    assert off.state is ActivationState.AGGRESSIVE_OFF
    assert off.mom60 == -0.5
    mid = classify_activation_state(decision_date=days[5], mom60_by_date=series, min_history=4)
    assert mid.state is ActivationState.UNCERTAIN
    assert mid.mom60 == 0.15



def test_classify_activation_state_ignores_audit_hvr_true_reversal() -> None:
    from datetime import date

    from src.research.activation_state import ActivationState, classify_activation_state
    from src.research.audit_regime import AUDIT_REGIME_IS_ACTIVATION_GATE, audit_regime_label

    days = [date(2020, 2, d) for d in (3, 4, 5, 6, 7)]
    series = dict(zip(days, [0.40, 0.30, 0.20, 0.10, -0.22], strict=True))
    snap = classify_activation_state(decision_date=days[-1], mom60_by_date=series, min_history=4)
    assert snap.state is ActivationState.AGGRESSIVE_OFF
    assert AUDIT_REGIME_IS_ACTIVATION_GATE is False
    assert audit_regime_label(mom60=-0.22, mom20=0.23, rv20=0.90, dd60=-0.26) == "high_vol_reversal"



def test_classify_activation_series_is_expanding_not_fullsample() -> None:
    from datetime import date

    from src.research.activation_state import ActivationState, classify_activation_series, classify_activation_state

    days = [date(2020, 3, d) for d in range(2, 9)]
    values = [0.0, 1.0, 2.0, 3.0, 100.0, 100.0, 100.0]
    series = dict(zip(days, values, strict=True))
    snaps = classify_activation_series(series, min_history=4)
    assert tuple(item.as_of for item in snaps) == tuple(days)
    early = snaps[3]
    assert early.as_of == days[3]
    assert early.state is ActivationState.AGGRESSIVE_ON
    single = classify_activation_state(decision_date=days[3], mom60_by_date={d: series[d] for d in days[:4]}, min_history=4)
    assert single.state is ActivationState.AGGRESSIVE_ON
    assert early.state == single.state



def test_activation_conditional_table_length_mismatch_and_zero_oracle_capture() -> None:
    import pytest
    import polars as pl

    from src.research.activation_state import ActivationState, activation_conditional_table

    with pytest.raises(ValueError, match="length"):
        activation_conditional_table(
            states=(ActivationState.AGGRESSIVE_ON,),
            terminal_returns=(0.1, 0.2),
            oracle_returns=(0.1, 0.2),
        )
    table = activation_conditional_table(
        states=(ActivationState.AGGRESSIVE_OFF, ActivationState.AGGRESSIVE_OFF),
        terminal_returns=(0.10, 0.11),
        oracle_returns=(-0.10, -0.05),
    )
    assert table.height == 3
    assert table["state"].to_list() == ["AGGRESSIVE_ON", "UNCERTAIN", "AGGRESSIVE_OFF"]
    off = table.filter(pl.col("state") == "AGGRESSIVE_OFF")
    assert off["n_windows"][0] == 2
    assert off["n_effective"][0] == pytest.approx(2.0 / 36.0)
    assert off["p27_p50"][0] == 0.0
    assert off["capture_50"][0] == 0.0
    assert off["executable_oracle_p50"][0] == 0.0
    on = table.filter(pl.col("state") == "AGGRESSIVE_ON")
    assert on["n_windows"][0] == 0
    assert on["p27_p50"][0] == 0.0
    mixed = activation_conditional_table(
        states=(ActivationState.AGGRESSIVE_ON, ActivationState.AGGRESSIVE_ON),
        terminal_returns=(0.60, 0.10),
        oracle_returns=(0.55, 0.51),
    )
    on_hit = mixed.filter(pl.col("state") == "AGGRESSIVE_ON")
    assert on_hit["p27_p50"][0] == pytest.approx(0.5)
    assert on_hit["executable_oracle_p50"][0] == pytest.approx(1.0)
    assert on_hit["capture_50"][0] == pytest.approx(0.5)



def test_activation_constants_are_research_only() -> None:
    from src.features.regime import RegimeState
    from src.research.activation_state import ACTIVATION_STATE_IS_PRODUCTION_GATE, ActivationState
    from src.research.audit_regime import AUDIT_REGIME_IS_ACTIVATION_GATE

    assert ACTIVATION_STATE_IS_PRODUCTION_GATE is False
    assert AUDIT_REGIME_IS_ACTIVATION_GATE is False
    assert {member.value for member in ActivationState} == {"AGGRESSIVE_ON", "UNCERTAIN", "AGGRESSIVE_OFF"}
    assert set(ActivationState) != set(RegimeState)
    assert "AGGRESSIVE_ON" not in {member.value for member in RegimeState}

