"""SCENARIO-10-06 SCENARIO-10-09"""
from datetime import date

import pytest

from src.tournament.distribution import (
    ReturnDistribution,
    b1_gate_anchors_from_distribution,
    measure_vehicle_activity_from_allocate,
    preflight_features_span_ok,
    score_seed_for_vehicle_probe,
    vehicle_activity_rate,
)
from src.universe.instruments import InstrumentAttributes, InstrumentMaster
from src.universe.taxonomy import Taxonomy
import polars as pl


def test_SCENARIO_10_06_vehicle_rate() -> None:
    assert abs(vehicle_activity_rate([2, 2, 1, 2], [True, True, True, False]) - 2 / 3) < 1e-9
    assert vehicle_activity_rate([2, 1], [False, False]) == 0.0
    assert vehicle_activity_rate([], []) == 0.0
    assert vehicle_activity_rate([2, 2], [True, True]) == 1.0


def test_SCENARIO_10_09_measure_vehicle_activity() -> None:
    panel = pl.DataFrame([
        {"date": date(2026, 1, 2), "ticker": "A1", "name": "KODEX 200", "underlying_index_name": "KOSPI 200"},
        {"date": date(2026, 1, 2), "ticker": "A2", "name": "KODEX 레버리지", "underlying_index_name": "KOSPI 200"},
    ])
    master = InstrumentMaster.build(panel, Taxonomy(rules=[]), {})
    a1 = master.attributes["A1"]
    a2 = master.attributes["A2"]
    fk = a1.leverage_family_key
    master2 = InstrumentMaster(
        attributes={
            "A1": a1,
            "A2": InstrumentAttributes(
                ticker=a2.ticker,
                name=a2.name,
                issuer=a2.issuer,
                leverage_multiple=2,
                leverage_family_key=fk,
                is_synthetic=False,
                is_hedged=False,
                is_active=True,
                index_key=a2.index_key,
                theme=a2.theme,
                first_seen=a1.first_seen,
                last_seen=a1.last_seen,
                left_censored=False,
                confidence=a1.confidence,
            ),
        },
        panel_start=master.panel_start,
    )
    from src.portfolio.policy import PortfolioPolicy
    from src.portfolio.sizing import ConfidenceSizingConfig

    policy = PortfolioPolicy(sizing_config=ConfidenceSizingConfig(), master=master2, state_enabled=False)

    class _Snap:
        state = "RISK_ON"

    rate = measure_vehicle_activity_from_allocate(
        policy,
        [date(2026, 1, 2), date(2026, 1, 3)],
        {date(2026, 1, 2): _Snap(), date(2026, 1, 3): _Snap()},
        True,
        {"A1": 1.0, "A2": 0.1},
    )
    assert rate == 1.0
    seed = score_seed_for_vehicle_probe(master2)
    assert len(seed) >= 2
    rate_probe = measure_vehicle_activity_from_allocate(
        policy,
        [date(2026, 1, 2), date(2026, 1, 3)],
        {date(2026, 1, 2): _Snap(), date(2026, 1, 3): _Snap()},
        True,
    )
    assert rate_probe == 1.0


def test_b1_gate_anchors_from_distribution() -> None:
    dist = ReturnDistribution(
        name="B1",
        horizon=36,
        returns=(0.1, 0.2),
        n_windows=2,
        n_effective=1,
        quantiles={0.5: 0.15},
        exceedance={0.3: 0.05, 0.4: 0.04},
        cvar_05=-0.31,
        right_tail_score=0.2,
    )
    p30, p40, cvar = b1_gate_anchors_from_distribution(dist)
    assert abs(p30 - 0.05) < 1e-12
    assert abs(p40 - 0.04) < 1e-12
    assert abs(cvar - (-0.31)) < 1e-12


def test_SCENARIO_10_09_preflight_span() -> None:
    assert preflight_features_span_ok(date(2018, 1, 2), date(2026, 8, 27), date(2018, 1, 2), date(2026, 8, 27))
    assert not preflight_features_span_ok(date(2024, 1, 2), date(2024, 1, 31), date(2018, 1, 2), date(2026, 8, 27))


@pytest.mark.parametrize("scenario_id", ["SCENARIO-10-06", "SCENARIO-10-09"])
def test_SCENARIO_hyphen_wrapper(scenario_id: str) -> None:
    if scenario_id == "SCENARIO-10-06":
        test_SCENARIO_10_06_vehicle_rate()
    elif scenario_id == "SCENARIO-10-09":
        test_SCENARIO_10_09_measure_vehicle_activity()
        test_SCENARIO_10_09_preflight_span()
        test_b1_gate_anchors_from_distribution()
