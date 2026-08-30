from __future__ import annotations

from datetime import date

import polars as pl
import pytest

from src.portfolio.policy import PortfolioPolicy
from src.portfolio.sizing import ConfidenceSizingConfig, confidence_vehicle_gate
from src.universe.instruments import InstrumentAttributes, InstrumentMaster
from src.universe.taxonomy import Taxonomy


def test_confidence_vehicle_gate_threshold() -> None:
    cfg = ConfidenceSizingConfig(w_max=1.0)
    assert confidence_vehicle_gate(w_top=0.34, config=cfg, vehicle_conf_min=0.85) is True
    assert confidence_vehicle_gate(w_top=0.90, config=cfg, vehicle_conf_min=0.85) is False
    # edge: w_top exactly at threshold -> False (not less)
    assert confidence_vehicle_gate(w_top=0.85, config=cfg, vehicle_conf_min=0.85) is False
    # w_max scaling
    cfg2 = ConfidenceSizingConfig(w_max=2.0)
    assert confidence_vehicle_gate(w_top=1.6, config=cfg2, vehicle_conf_min=0.85) is True  # 1.6 <1.7
    assert confidence_vehicle_gate(w_top=1.8, config=cfg2, vehicle_conf_min=0.85) is False


def test_allocate_passes_confidence_low_to_pick_vehicle() -> None:
    panel = pl.DataFrame(
        [
            {"date": date(2026, 1, 2), "ticker": "T1", "name": "KODEX 200", "underlying_index_name": "KOSPI 200"},
            {"date": date(2026, 1, 2), "ticker": "T2", "name": "KODEX 레버리지", "underlying_index_name": "KOSPI 200"},
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

    captured: list[bool | None] = []
    orig_pick = master2  # keep ref
    from src.portfolio.exposure import ExposureSelector

    orig_select = ExposureSelector.select
    orig_pick_vehicle = ExposureSelector.pick_vehicle

    def wrapped_pick_vehicle(self, tickers, leverage_allowed=None, confidence_low=False, regime=None, inverse_allowed=None):  # type: ignore[no-untyped-def]
        captured.append(bool(confidence_low))
        return orig_pick_vehicle(self, tickers, leverage_allowed=leverage_allowed, confidence_low=confidence_low, regime=regime, inverse_allowed=inverse_allowed)

    # patch
    import unittest.mock as mock

    with mock.patch.object(ExposureSelector, "pick_vehicle", wrapped_pick_vehicle):
        # low-spread scores -> confidence_low should be True
        low_scores = {"T1": 0.03, "T2": 0.029}
        # high-spread scores -> False
        high_scores = {"T1": 1.0, "T2": 0.1}
        # need to ensure allocate picks confidence correctly
        # First low
        captured.clear()
        dec_low = policy.allocate(low_scores, regime="RISK_ON", leverage_allowed=True)
        assert captured, "pick_vehicle not called"
        assert captured[-1] is True, f"expected confidence_low True for low spread, got {captured[-1]}"

        captured.clear()
        dec_high = policy.allocate(high_scores, regime="RISK_ON", leverage_allowed=True)
        assert captured[-1] is False, f"expected confidence_low False for high spread, got {captured[-1]}"


@pytest.mark.parametrize("scenario_id", ["test_confidence_vehicle_gate_threshold", "test_allocate_passes_confidence_low_to_pick_vehicle"])
def test_scenario_wrapper(scenario_id: str) -> None:
    if scenario_id == "test_confidence_vehicle_gate_threshold":
        test_confidence_vehicle_gate_threshold()
    elif scenario_id == "test_allocate_passes_confidence_low_to_pick_vehicle":
        test_allocate_passes_confidence_low_to_pick_vehicle()
