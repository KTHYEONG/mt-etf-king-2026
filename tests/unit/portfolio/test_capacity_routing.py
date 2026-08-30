# ruff: noqa
from datetime import date

import polars as pl

from src.portfolio.exposure import CapacityContext, ExposureSelector, VehicleRoute
from src.universe.instruments import InstrumentAttributes, InstrumentMaster
from src.universe.taxonomy import Taxonomy


def _master_with_family() -> InstrumentMaster:
    panel = pl.DataFrame(
        [
            {"date": date(2026, 1, 2), "ticker": "T1", "name": "KODEX 200", "underlying_index_name": "KOSPI 200"},
            {"date": date(2026, 1, 2), "ticker": "T2", "name": "KODEX 레버리지", "underlying_index_name": "KOSPI 200"},
            {"date": date(2026, 1, 3), "ticker": "T1", "name": "KODEX 200", "underlying_index_name": "KOSPI 200"},
            {"date": date(2026, 1, 3), "ticker": "T2", "name": "KODEX 레버리지", "underlying_index_name": "KOSPI 200"},
        ]
    )
    taxonomy = Taxonomy(rules=[])
    master = InstrumentMaster.build(panel, taxonomy, {})
    attrs = dict(master.attributes)
    a = attrs["T1"]
    b = attrs["T2"]
    new_b = InstrumentAttributes(
        ticker=b.ticker,
        name=b.name,
        issuer=b.issuer,
        leverage_multiple=2,
        leverage_family_key=a.leverage_family_key,
        is_synthetic=b.is_synthetic,
        is_hedged=b.is_hedged,
        is_active=b.is_active,
        index_key=b.index_key,
        theme=b.theme,
        first_seen=b.first_seen,
        last_seen=b.last_seen,
        left_censored=b.left_censored,
        confidence=b.confidence,
    )
    new_a = InstrumentAttributes(
        ticker=a.ticker,
        name=a.name,
        issuer=a.issuer,
        leverage_multiple=1,
        leverage_family_key=a.leverage_family_key,
        is_synthetic=a.is_synthetic,
        is_hedged=a.is_hedged,
        is_active=a.is_active,
        index_key=a.index_key,
        theme=a.theme,
        first_seen=a.first_seen,
        last_seen=a.last_seen,
        left_censored=a.left_censored,
        confidence=a.confidence,
    )
    return InstrumentMaster(attributes={"T1": new_a, "T2": new_b}, panel_start=master.panel_start)


def test_capacity_context_uses_execution_date_adv() -> None:
    master = _master_with_family()
    # Different ADV maps for t and t+1
    adv_t = {"T1": 100_000_000.0, "T2": 50_000_000.0}
    adv_t1 = {"T1": 200_000_000_000.0, "T2": 150_000_000_000.0}
    equity = 1_000_000_000.0
    participation = 0.01
    # CapacityContext should contain only t+1 value
    cap = CapacityContext(equity=equity, participation=participation, adv_by_ticker=dict(adv_t1), current_weights={})
    # available_delta equals ADV(t+1)*participation/equity
    sel = ExposureSelector(master)
    cap_for_t2 = sel._cap_for("T2", cap)
    assert cap_for_t2 is not None
    expected = adv_t1["T2"] * participation / equity
    assert abs(cap_for_t2 - expected) < 1e-12
    # Ensure t value not used
    assert abs(cap_for_t2 - adv_t["T2"] * participation / equity) > 1e-12
    # VehicleRoute available_delta check via select
    route = sel.select_capacity_aware("T1", regime="RISK_ON", leverage_allowed=True, inverse_allowed=None, confidence_low=False, capacity=cap)
    # With large cap, should select 2x
    assert route.vehicle_ticker == "T2"
    assert abs(route.available_delta - expected) < 1e-12
    assert route.required_delta == 1.0


def test_capacity_router_demotes_unfillable_2x_to_1x() -> None:
    master = _master_with_family()
    equity = 1_000_000_000.0
    participation = 0.01
    # Make T2 cap too small (adv small), T1 cap large enough
    # cap = adv*part/equity => need req 1.0 > cap2 but req 1.0 <= cap1
    adv = {"T1": 200_000_000_000.0, "T2": 10_000_000.0}  # T2 cap = 10M*0.01/1B=0.0001 <1.0 ; T1 cap =200B*0.01/1B=2.0 >1.0
    cap = CapacityContext(equity=equity, participation=participation, adv_by_ticker=adv, current_weights={})
    sel = ExposureSelector(master)
    route = sel.select_capacity_aware("T1", regime="RISK_ON", leverage_allowed=True, inverse_allowed=None, confidence_low=False, capacity=cap)
    assert route.vehicle_ticker == "T1"
    assert route.multiple == 1
    assert route.reason == "CAPACITY_DEMOTE"
    assert route.required_delta == 1.0


def test_capacity_router_defers_entry_until_same_family_unwind() -> None:
    master = _master_with_family()
    equity = 1_000_000_000.0
    participation = 0.01
    # Held T1 weight 0.5, but cap for T1 small so cannot unwind fully
    adv = {"T1": 10_000_000.0, "T2": 200_000_000_000.0}  # held T1 cap 0.0001 <0.5 => cannot unwind
    current = {"T1": 0.5}
    cap = CapacityContext(equity=equity, participation=participation, adv_by_ticker=adv, current_weights=current)
    sel = ExposureSelector(master)
    route = sel.select_capacity_aware("T1", regime="RISK_ON", leverage_allowed=True, inverse_allowed=None, confidence_low=False, capacity=cap)
    assert route.reason == "UNWIND_DEFER"
    # No second family vehicle receives positive weight: vehicle remains held
    assert route.vehicle_ticker == "T1"
    # alternative check via pick_vehicle should not open second vehicle
    out = sel.pick_vehicle(["T1"], regime="RISK_ON", leverage_allowed=True, confidence_low=False, capacity=cap)
    assert out["T1"] == "T1"


def test_capacity_router_denies_2x_when_leverage_gate_blocks() -> None:
    master = _master_with_family()
    equity = 1_000_000_000.0
    participation = 0.01
    adv = {"T1": 200_000_000_000.0, "T2": 200_000_000_000.0}
    cap = CapacityContext(equity=equity, participation=participation, adv_by_ticker=adv, current_weights={})
    sel = ExposureSelector(master)
    route = sel.select_capacity_aware(
        "T1",
        regime="RISK_ON",
        leverage_allowed=False,
        inverse_allowed=None,
        confidence_low=False,
        capacity=cap,
    )
    assert route.vehicle_ticker == "T1"
    assert route.multiple == 1
    assert route.vehicle_ticker != "T2"
    route_unknown = sel.select_capacity_aware(
        "T1",
        regime="RISK_ON",
        leverage_allowed=None,
        inverse_allowed=None,
        confidence_low=False,
        capacity=cap,
    )
    assert route_unknown.vehicle_ticker == "T1"
    assert route_unknown.multiple == 1


def test_capacity_router_unknown_adv_fails_to_1x_or_hold() -> None:
    master = _master_with_family()
    equity = 1_000_000_000.0
    participation = 0.01
    # Missing or non-positive +2x ADV never selects +2x; executable +1x is selected
    adv_missing = {"T1": 200_000_000_000.0}  # T2 missing
    cap = CapacityContext(equity=equity, participation=participation, adv_by_ticker=adv_missing, current_weights={})
    sel = ExposureSelector(master)
    route = sel.select_capacity_aware("T1", regime="RISK_ON", leverage_allowed=True, inverse_allowed=None, confidence_low=False, capacity=cap)
    assert route.vehicle_ticker == "T1"
    assert route.multiple == 1
    # non-positive
    adv_zero = {"T1": 200_000_000_000.0, "T2": 0.0}
    cap2 = CapacityContext(equity=equity, participation=participation, adv_by_ticker=adv_zero, current_weights={})
    route2 = sel.select_capacity_aware("T1", regime="RISK_ON", leverage_allowed=True, inverse_allowed=None, confidence_low=False, capacity=cap2)
    assert route2.vehicle_ticker == "T1"
    assert route2.vehicle_ticker != "T2"
    # otherwise the existing vehicle is retained with no new family member: if +1x also not executable
    adv_none = {"T1": 1_000.0, "T2": 0.0}  # both tiny caps <1.0
    cap3 = CapacityContext(equity=equity, participation=participation, adv_by_ticker=adv_none, current_weights={"T1": 0.5})
    route3 = sel.select_capacity_aware("T1", regime="RISK_ON", leverage_allowed=True, inverse_allowed=None, confidence_low=False, capacity=cap3)
    # Should retain existing? vehicle stays T1 due to unwind defer or hold
    assert route3.vehicle_ticker == "T1"
