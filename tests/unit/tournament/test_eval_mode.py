from __future__ import annotations

from datetime import date

import polars as pl
import pytest

from src.core.calendar import TradingCalendar
from src.portfolio.policy import PortfolioPolicy
from src.portfolio.sizing import ConfidenceSizingConfig
from src.tournament.eval_mode import EvalMode, resolve_eval_flags


def test_resolve_eval_flags_adoption_disables_state() -> None:
    model = PortfolioPolicy()
    flags = resolve_eval_flags(model, EvalMode.ADOPTION)
    assert flags.state_enabled is False
    assert flags.path_dependent is False
    assert model.state_enabled is False
    assert model.path_dependent is False
    from src.tournament.simulator import model_requires_path_dependent

    assert model_requires_path_dependent(model) is False
    # also string variant
    model2 = PortfolioPolicy()
    flags2 = resolve_eval_flags(model2, "adoption")
    assert flags2.state_enabled is False
    assert flags2.path_dependent is False
    assert model_requires_path_dependent(model2) is False


def test_resolve_eval_flags_operational_enables_state() -> None:
    model = PortfolioPolicy()
    flags = resolve_eval_flags(model, EvalMode.OPERATIONAL)
    assert flags.state_enabled is True
    assert flags.path_dependent is True
    model2 = PortfolioPolicy()
    flags2 = resolve_eval_flags(model2, "operational")
    assert flags2.state_enabled is True
    assert flags2.path_dependent is True


def test_simulate_window_vehicle_rate_aggressive() -> None:
    from src.portfolio.sizing import ConfidenceSizingConfig
    from src.tournament.distribution import measure_vehicle_activity_from_allocate
    from src.universe.instruments import InstrumentAttributes, InstrumentMaster
    from src.universe.taxonomy import Taxonomy

    panel = pl.DataFrame(
        [
            {"date": date(2026, 1, 2), "ticker": "T1", "name": "KODEX 200", "underlying_index_name": "KOSPI 200"},
            {"date": date(2026, 1, 3), "ticker": "T1", "name": "KODEX 200", "underlying_index_name": "KOSPI 200"},
            {"date": date(2026, 1, 2), "ticker": "T2", "name": "KODEX 레버리지", "underlying_index_name": "KOSPI 200"},
            {"date": date(2026, 1, 3), "ticker": "T2", "name": "KODEX 레버리지", "underlying_index_name": "KOSPI 200"},
        ]
    )
    taxonomy = Taxonomy(rules=[])
    master = InstrumentMaster.build(panel, taxonomy, {})
    a = master.attributes["T1"]
    b = master.attributes["T2"]
    fk = a.leverage_family_key
    attrs = {
        "T1": InstrumentAttributes(
            ticker=a.ticker,
            name=a.name,
            issuer=a.issuer,
            leverage_multiple=1,
            leverage_family_key=fk,
            is_synthetic=a.is_synthetic,
            is_hedged=a.is_hedged,
            is_active=a.is_active,
            index_key=a.index_key,
            theme=a.theme,
            first_seen=a.first_seen,
            last_seen=a.last_seen,
            left_censored=a.left_censored,
            confidence=a.confidence,
        ),
        "T2": InstrumentAttributes(
            ticker=b.ticker,
            name=b.name,
            issuer=b.issuer,
            leverage_multiple=2,
            leverage_family_key=fk,
            is_synthetic=b.is_synthetic,
            is_hedged=b.is_hedged,
            is_active=b.is_active,
            index_key=b.index_key,
            theme=b.theme,
            first_seen=b.first_seen,
            last_seen=b.last_seen,
            left_censored=b.left_censored,
            confidence=b.confidence,
        ),
    }
    master2 = InstrumentMaster(attributes=attrs, panel_start=master.panel_start)
    cfg = ConfidenceSizingConfig()
    policy = PortfolioPolicy(sizing_config=cfg, master=master2, state_enabled=False)
    policy.name = "P11"  # type: ignore[attr-defined]
    # high confidence scores -> w_top ~1.0 -> confidence_low False -> should pick +2x
    high_scores = {"T1": 1.0, "T2": 0.1}
    # monkey patch score not needed; we use measure_vehicle_activity which uses allocate with seed derived from master
    # For this test we provide explicit score_seed with high confidence T1
    sessions = [date(2026, 1, 2), date(2026, 1, 3), date(2026, 1, 4), date(2026, 1, 5)]
    # Build regimes: all RISK_ON
    from src.features.regime import RegimeSnapshot, RegimeState

    regimes = {}
    for d in sessions:
        regimes[d] = RegimeSnapshot(as_of=d, state=RegimeState.RISK_ON, score=0.9, components={})
    # Need policy to use high scores internally; override score_seed via direct call
    # measure_vehicle_activity_from_allocate uses score_seed; we provide high conf seed
    # But policy allocate uses scores param; with high spread, confidence gate should be low=False
    # Ensure policy allocate with high scores yields leverage
    # We set up policy master and will test via measure function with custom seed
    rate = measure_vehicle_activity_from_allocate(policy, sessions, regimes, leverage_allowed=True, score_seed=high_scores)
    assert rate >= 0.5


def test_run_rolling_leverage_propagation_single_protocol() -> None:
    import unittest.mock as mock

    from src.backtest.costs import CostConfig
    from src.backtest.engine import BacktestConfig
    from src.backtest.session_cache import build_session_cache
    from src.portfolio.sizing import SizingScheme
    from src.tournament.simulator import TournamentSimulator
    from tests.unit.backtest.conftest import build_engine, panel_row

    cal = TradingCalendar()
    sessions = cal.sessions(date(2026, 1, 2), date(2026, 1, 20))
    panel = pl.DataFrame([panel_row(day=d, ticker="069500", close=30000.0, mom_20=0.2) for d in sessions])
    engine, cal2, filt = build_engine(panel)
    config = BacktestConfig(
        start=sessions[0],
        end=sessions[-1],
        capital=1_000_000_000.0,
        scheme=SizingScheme.TOP1,
        k=1,
        filters=filt,
        costs=CostConfig(0.0, 0.0, 0.0),
    )
    policy = PortfolioPolicy(sizing_config=ConfidenceSizingConfig())
    policy.name = "P11"  # type: ignore[attr-defined]
    policy.scores_path_independent = True
    resolve_eval_flags(policy, EvalMode.OPERATIONAL)

    def _score(snapshot, ctx):  # type: ignore[no-untyped-def]
        return {"069500": 1.0}

    policy.score = _score  # type: ignore[attr-defined]
    sim = TournamentSimulator(engine, cal2)
    captured: list[dict[str, object]] = []

    def _capture_build(*args, **kwargs):  # type: ignore[no-untyped-def]
        captured.append(dict(kwargs))
        return build_session_cache(*args, **kwargs)

    with mock.patch("src.backtest.session_cache.build_session_cache", side_effect=_capture_build):
        rolling = sim.run_rolling(
            policy,
            panel,
            config,
            horizon=5,
            path_dependent=True,
            leverage_allowed=True,
            inverse_allowed=False,
        )
    assert len(rolling.returns) > 0
    assert captured, "build_session_cache not invoked"
    assert captured[-1].get("leverage_allowed") is True
    assert captured[-1].get("inverse_allowed") is False


def test_resolve_eval_flags_adoption_keeps_path_dependent_for_sticky() -> None:
    from src.alpha.baselines import BASELINES
    from src.portfolio.policy import PortfolioPolicy
    from src.tournament.eval_mode import EvalMode, resolve_eval_flags
    from src.tournament.simulator import model_requires_path_dependent

    sticky = BASELINES["P27"]()
    flags = resolve_eval_flags(sticky, EvalMode.ADOPTION)
    assert flags.path_dependent is True
    assert flags.state_enabled is False
    assert sticky.path_dependent is True
    assert sticky.scores_path_independent is False
    assert model_requires_path_dependent(sticky) is True

    policy = PortfolioPolicy()
    policy.scores_path_independent = True
    pflags = resolve_eval_flags(policy, EvalMode.ADOPTION)
    assert pflags.path_dependent is False
    assert pflags.state_enabled is False
    assert policy.path_dependent is False
    assert model_requires_path_dependent(policy) is False


@pytest.mark.parametrize("scenario_id", ["test_resolve_eval_flags_adoption_disables_state", "test_resolve_eval_flags_operational_enables_state", "test_simulate_window_vehicle_rate_aggressive", "test_run_rolling_leverage_propagation_single_protocol"])
def test_scenario_wrapper(scenario_id: str) -> None:
    if scenario_id == "test_resolve_eval_flags_adoption_disables_state":
        test_resolve_eval_flags_adoption_disables_state()
    elif scenario_id == "test_resolve_eval_flags_operational_enables_state":
        test_resolve_eval_flags_operational_enables_state()
    elif scenario_id == "test_simulate_window_vehicle_rate_aggressive":
        test_simulate_window_vehicle_rate_aggressive()
    elif scenario_id == "test_run_rolling_leverage_propagation_single_protocol":
        test_run_rolling_leverage_propagation_single_protocol()


def test_resolve_path_dependent_mode_sticky_returns_fast() -> None:
    from src.alpha.baselines import BASELINES
    from src.tournament.eval_mode import EvalMode, resolve_path_dependent_mode

    p27 = BASELINES["P27"]()
    p21 = BASELINES["P21"]()
    assert p27.scores_path_independent is False
    assert resolve_path_dependent_mode(p27, mode=EvalMode.ADOPTION) == "fast"
    assert resolve_path_dependent_mode(p21, mode=EvalMode.ADOPTION) == "fast"
