

def test_annualize_daily_realized_vol_fail_closed_none_nonfinite_negative() -> None:
    from src.research.audit_regime import annualize_daily_realized_vol

    assert annualize_daily_realized_vol(None) is None
    assert annualize_daily_realized_vol(float("nan")) is None
    assert annualize_daily_realized_vol(float("inf")) is None
    assert annualize_daily_realized_vol(float("-inf")) is None
    assert annualize_daily_realized_vol(-0.01) is None


def test_annualize_daily_realized_vol_matches_hvr_vector() -> None:
    import math

    from src.research.audit_regime import AUDIT_RV20_SESSIONS_PER_YEAR, annualize_daily_realized_vol

    assert AUDIT_RV20_SESSIONS_PER_YEAR == 252.0
    daily = 0.05644660723119608
    got = annualize_daily_realized_vol(daily)
    assert got is not None
    assert math.isclose(got, daily * (252.0 ** 0.5), rel_tol=0.0, abs_tol=1e-12)
    assert got > 0.25


def test_classify_championship_sleeve_uncertain_on_missing_or_nonfinite() -> None:
    from datetime import date

    from src.tournament.championship_regime import ChampionshipSleeve, classify_championship_sleeve

    decision = date(2026, 8, 27)
    for mom60, mom20, rv in (
        (None, 0.23, 0.056),
        (-0.22, None, 0.056),
        (-0.22, 0.23, None),
        (float("nan"), 0.23, 0.056),
        (-0.22, float("inf"), 0.056),
        (-0.22, 0.23, float("nan")),
    ):
        snap = classify_championship_sleeve(decision_date=decision, mom60=mom60, mom20=mom20, rv20_daily=rv)
        assert snap.sleeve is ChampionshipSleeve.UNCERTAIN
        assert snap.as_of == decision


def test_classify_championship_sleeve_crash_rebound_on_rebound_vector() -> None:
    from datetime import date

    from src.research.audit_regime import annualize_daily_realized_vol, audit_regime_label
    from src.tournament.championship_regime import ChampionshipSleeve, classify_championship_sleeve

    snap = classify_championship_sleeve(decision_date=date(2026, 8, 27), mom60=-0.22, mom20=0.23, rv20_daily=0.05644660723119608)
    assert snap.sleeve is ChampionshipSleeve.CRASH_REBOUND
    assert snap.mom60 == -0.22
    assert snap.mom20 == 0.23
    assert snap.rv20_annualized == annualize_daily_realized_vol(0.05644660723119608)
    assert audit_regime_label(mom60=-0.22, mom20=0.23, rv20=snap.rv20_annualized, dd60=-0.26) == "high_vol_reversal"
    high_vol_only = classify_championship_sleeve(decision_date=date(2020, 3, 23), mom60=0.10, mom20=0.02, rv20_daily=0.40)
    assert high_vol_only.sleeve is not ChampionshipSleeve.CRASH_REBOUND


def test_classify_championship_sleeve_lottery_on_when_mom60_hot() -> None:
    from datetime import date

    from src.tournament.championship_regime import ChampionshipSleeve, LOTTERY_ON_MOM60_MIN, classify_championship_sleeve

    assert LOTTERY_ON_MOM60_MIN == 0.08
    snap = classify_championship_sleeve(decision_date=date(2025, 9, 22), mom60=0.12, mom20=0.04, rv20_daily=0.01)
    assert snap.sleeve is ChampionshipSleeve.LOTTERY_ON


def test_classify_championship_sleeve_inactive_when_weak_trend() -> None:
    from datetime import date

    from src.tournament.championship_regime import ChampionshipSleeve, classify_championship_sleeve

    snap = classify_championship_sleeve(decision_date=date(2019, 6, 3), mom60=0.02, mom20=0.01, rv20_daily=0.01)
    assert snap.sleeve is ChampionshipSleeve.INACTIVE


def test_classify_championship_sleeve_from_maps_rejects_future_dates() -> None:
    from datetime import date

    import pytest

    from src.features.pit import PitViolationError
    from src.tournament.championship_regime import classify_championship_sleeve_from_maps

    decision = date(2020, 1, 2)
    future = date(2020, 1, 3)
    with pytest.raises(PitViolationError):
        classify_championship_sleeve_from_maps(
            decision_date=decision,
            mom60_by_date={decision: -0.22, future: 0.1},
            mom20_by_date={decision: 0.23},
            rv20_daily_by_date={decision: 0.056},
        )


def test_classify_championship_sleeve_series_is_expanding_pit() -> None:
    from datetime import date

    from src.tournament.championship_regime import classify_championship_sleeve_from_maps, classify_championship_sleeve_series

    days = [date(2020, 3, d) for d in (2, 3, 4, 5)]
    mom60 = dict(zip(days, [0.12, 0.10, -0.22, -0.22], strict=True))
    mom20 = dict(zip(days, [0.04, 0.04, 0.23, 0.23], strict=True))
    rv = dict(zip(days, [0.01, 0.01, 0.056, 0.056], strict=True))
    snaps = classify_championship_sleeve_series(mom60_by_date=mom60, mom20_by_date=mom20, rv20_daily_by_date=rv)
    assert tuple(item.as_of for item in snaps) == tuple(days)
    cut = {days[0]: mom60[days[0]], days[1]: mom60[days[1]]}
    cut20 = {days[0]: mom20[days[0]], days[1]: mom20[days[1]]}
    cutrv = {days[0]: rv[days[0]], days[1]: rv[days[1]]}
    early = classify_championship_sleeve_from_maps(decision_date=days[1], mom60_by_date=cut, mom20_by_date=cut20, rv20_daily_by_date=cutrv)
    assert snaps[1].sleeve == early.sleeve
    assert snaps[1].as_of == days[1]


def test_abs_mom_rebound_bypass_allowed_stays_false_while_gate_false() -> None:
    from src.features.regime import RegimeState
    from src.research.activation_state import ACTIVATION_STATE_IS_PRODUCTION_GATE
    from src.research.audit_regime import AUDIT_REGIME_IS_ACTIVATION_GATE
    from src.tournament.championship_regime import (
        CHAMPIONSHIP_SLEEVE_IS_PRODUCTION_GATE,
        ChampionshipSleeve,
        abs_mom_rebound_bypass_allowed,
    )

    assert CHAMPIONSHIP_SLEEVE_IS_PRODUCTION_GATE is False
    assert ACTIVATION_STATE_IS_PRODUCTION_GATE is False
    assert AUDIT_REGIME_IS_ACTIVATION_GATE is False
    assert abs_mom_rebound_bypass_allowed(sleeve=ChampionshipSleeve.CRASH_REBOUND.value, config_enabled=True) is False
    assert abs_mom_rebound_bypass_allowed(sleeve=ChampionshipSleeve.CRASH_REBOUND.value, config_enabled=False) is False
    assert abs_mom_rebound_bypass_allowed(sleeve=None, config_enabled=True) is False
    assert ChampionshipSleeve.CRASH_REBOUND.value not in {member.value for member in RegimeState}


def test_sleeve_conditional_table_length_mismatch_and_zero_oracle() -> None:
    import pytest
    import polars as pl

    from src.tournament.championship_regime import ChampionshipSleeve, sleeve_conditional_table

    with pytest.raises(ValueError, match="length"):
        sleeve_conditional_table(sleeves=(ChampionshipSleeve.CRASH_REBOUND,), terminal_returns=(0.1, 0.2), oracle_returns=(0.1, 0.2))
    table = sleeve_conditional_table(
        sleeves=(ChampionshipSleeve.INACTIVE, ChampionshipSleeve.INACTIVE),
        terminal_returns=(0.10, 0.11),
        oracle_returns=(-0.10, -0.05),
    )
    assert table.height == 4
    assert table["sleeve"].to_list() == ["CRASH_REBOUND", "LOTTERY_ON", "INACTIVE", "UNCERTAIN"]
    inactive = table.filter(pl.col("sleeve") == "INACTIVE")
    assert inactive["n_windows"][0] == 2
    assert inactive["n_effective"][0] == pytest.approx(2.0 / 36.0)
    assert inactive["p27_p50"][0] == 0.0
    assert inactive["capture_50"][0] == 0.0
    mixed = sleeve_conditional_table(
        sleeves=(ChampionshipSleeve.CRASH_REBOUND, ChampionshipSleeve.CRASH_REBOUND),
        terminal_returns=(0.60, 0.10),
        oracle_returns=(0.55, 0.51),
    )
    hit = mixed.filter(pl.col("sleeve") == "CRASH_REBOUND")
    assert hit["p27_p50"][0] == pytest.approx(0.5)
    assert hit["executable_oracle_p50"][0] == pytest.approx(1.0)
    assert hit["capture_50"][0] == pytest.approx(0.5)


def test_championship_sleeve_not_regime_state_and_gate_false() -> None:
    from src.alpha.base import DecisionContext
    from src.features.regime import RegimeState
    from src.tournament.championship_regime import CHAMPIONSHIP_SLEEVE_IS_PRODUCTION_GATE, ChampionshipSleeve

    assert CHAMPIONSHIP_SLEEVE_IS_PRODUCTION_GATE is False
    assert {member.value for member in ChampionshipSleeve} == {"CRASH_REBOUND", "LOTTERY_ON", "INACTIVE", "UNCERTAIN"}
    assert set(ChampionshipSleeve) != set(RegimeState)
    from datetime import date
    from src.universe.tournament import TournamentRules

    rules = TournamentRules(
        name="t",
        start_date=date(2026, 9, 21),
        end_date=date(2026, 11, 13),
        initial_capital=1_000_000_000,
        category="autonomous",
        leverage_allowed=True,
        inverse_allowed=True,
        max_weight=1.0,
        cash_allowed=True,
        sponsor_etf_only=True,
        manifest_path=None,
        issuer_whitelist=None,
        commission_bps=3.0,
        slippage_bps=5.0,
        max_order_to_adv=0.01,
        stress_grid=(0.01, 0.02, 0.05),
    )
    ctx = DecisionContext(decision_date=date(2026, 8, 27), regime=None, capital=1.0e9, held={}, rules=rules)
    assert ctx.championship_sleeve is None


def test_classify_p27_regime_missing_or_invalid_is_uncertain() -> None:
    from dataclasses import replace
    from datetime import date

    from src.tournament.championship_regime import P27RegimeInputs, P27RegimeState, classify_p27_regime

    decision = date(2026, 8, 27)
    missing = classify_p27_regime(decision_date=decision, inputs=None)
    assert missing.state is P27RegimeState.UNCERTAIN
    assert missing.confidence == 0.0
    assert missing.components == {"beta_direction": 0.0, "trend_quality": 0.0, "opportunity": 0.0, "risk_stability": 0.0}
    complete = P27RegimeInputs(
        kospi_mom20=0.1, kospi_mom60=0.2, kosdaq_mom20=0.1, kosdaq_mom60=0.2,
        kospi_eff60=0.5, kosdaq_eff60=0.5, kospi_eff60_prior_median=0.1, kosdaq_eff60_prior_median=0.1,
        leader_mom20=0.2, leader_mom60=0.3, breadth20=0.8, breadth60=0.8,
        kospi_dd60=-0.02, kospi_rv20_daily=0.01,
    )
    for invalid in (replace(complete, leader_mom60=None), replace(complete, breadth20=1.1), replace(complete, kospi_rv20_daily=-0.01)):
        snap = classify_p27_regime(decision_date=decision, inputs=invalid)
        assert snap.state is P27RegimeState.UNCERTAIN
        assert snap.confidence == 0.0


def test_classify_p27_regime_on_components_and_confidence() -> None:
    from datetime import date

    import pytest

    from src.tournament.championship_regime import P27RegimeInputs, P27RegimeState, classify_p27_regime

    inputs = P27RegimeInputs(
        kospi_mom20=0.1, kospi_mom60=0.2, kosdaq_mom20=0.1, kosdaq_mom60=0.2,
        kospi_eff60=0.5, kosdaq_eff60=0.3, kospi_eff60_prior_median=0.1, kosdaq_eff60_prior_median=0.0,
        leader_mom20=0.2, leader_mom60=0.3, breadth20=0.8, breadth60=0.7,
        kospi_dd60=-0.20, kospi_rv20_daily=0.01,
    )
    snap = classify_p27_regime(decision_date=date(2026, 1, 2), inputs=inputs)
    assert snap.state is P27RegimeState.ON
    assert snap.components == {"beta_direction": 1.0, "trend_quality": 1.0, "opportunity": 1.0, "risk_stability": 0.5}
    assert snap.confidence == pytest.approx(0.875)


def test_p27_regime_off_uncertain_and_conditional_table() -> None:
    from dataclasses import replace
    from datetime import date

    import polars as pl
    import pytest

    from src.tournament.championship_regime import (
        P27RegimeInputs,
        P27RegimeState,
        classify_p27_regime,
        p27_regime_conditional_table,
    )

    base = P27RegimeInputs(
        kospi_mom20=-0.1, kospi_mom60=-0.2, kosdaq_mom20=-0.1, kosdaq_mom60=-0.2,
        kospi_eff60=-0.5, kosdaq_eff60=-0.3, kospi_eff60_prior_median=0.1, kosdaq_eff60_prior_median=0.0,
        leader_mom20=-0.2, leader_mom60=-0.3, breadth20=0.2, breadth60=0.3,
        kospi_dd60=-0.20, kospi_rv20_daily=0.03,
    )
    off = classify_p27_regime(decision_date=date(2026, 1, 2), inputs=base)
    mixed = classify_p27_regime(decision_date=date(2026, 1, 2), inputs=replace(base, kospi_mom60=0.2))
    assert off.state is P27RegimeState.OFF
    assert mixed.state is P27RegimeState.UNCERTAIN
    table = p27_regime_conditional_table(
        states=(P27RegimeState.ON, P27RegimeState.ON, P27RegimeState.OFF),
        confidences=(1.0, 0.5, 0.0),
        terminal_returns=(0.60, -0.30, 0.30),
        overlap_horizon=2,
    )
    on = table.filter(pl.col("state") == "ON")
    off_row = table.filter(pl.col("state") == "OFF")
    assert table["state"].to_list() == ["ON", "UNCERTAIN", "OFF"]
    assert on["n_effective"][0] == pytest.approx(1.0)
    assert on["mean_confidence"][0] == pytest.approx(0.75)
    assert on["p50"][0] == pytest.approx(0.5)
    assert on["ruin25"][0] == pytest.approx(0.5)
    assert off_row["p30"][0] == 0.0
    with pytest.raises(ValueError, match="length"):
        p27_regime_conditional_table(states=(P27RegimeState.ON,), confidences=(), terminal_returns=(0.1,))

def test_p27_regime_conditional_table_rejects_bad_overlap_and_values() -> None:
    import pytest
    from src.tournament.championship_regime import P27RegimeState, p27_regime_conditional_table
    with pytest.raises(ValueError, match="overlap"):
        p27_regime_conditional_table(states=(P27RegimeState.ON,), confidences=(0.5,), terminal_returns=(0.1,), overlap_horizon=0)
    with pytest.raises(ValueError, match="finite"):
        p27_regime_conditional_table(states=(P27RegimeState.ON,), confidences=(0.5,), terminal_returns=(float("nan"),), overlap_horizon=2)
    with pytest.raises(ValueError, match="confidence"):
        p27_regime_conditional_table(states=(P27RegimeState.ON,), confidences=(1.5,), terminal_returns=(0.1,), overlap_horizon=2)
