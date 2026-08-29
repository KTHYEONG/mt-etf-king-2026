"""SCENARIO-08-16 SCENARIO-08-17"""
from datetime import date

import polars as pl

from src.portfolio.constraints import leverage_gate
from src.portfolio.exposure import ExposureSelector
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
    # force same family
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


def test_SCENARIO_08_16_exposure_low_confidence() -> None:  # noqa: N802
    master = _master_with_family()
    sel = ExposureSelector(master)
    fk = master.attributes["T1"].leverage_family_key
    chosen = sel.select(fk, leverage_allowed=True, confidence_low=True)
    assert chosen == "T1"
    # verify leverage_multiple ==1
    assert master.attributes[chosen].leverage_multiple == 1


def test_SCENARIO_08_17_leverage_gate_unknown() -> None:  # noqa: N802
    # leverage_allowed=None -> UNKNOWN -> deny leveraged
    assert leverage_gate("T2", None, None, False) is False
    # confidence_low also forces deny
    assert leverage_gate("T2", None, True, True) is False
    # allowed case
    assert leverage_gate("T1", None, True, False) is True


import pytest


@pytest.mark.parametrize("scenario_id", ["SCENARIO-08-16", "SCENARIO-08-17"])
def test_SCENARIO_hyphen_wrapper(scenario_id: str) -> None:  # noqa: N802
    if scenario_id == "SCENARIO-08-16":
        test_SCENARIO_08_16_exposure_low_confidence()
    if scenario_id == "SCENARIO-08-17":
        test_SCENARIO_08_17_leverage_gate_unknown()
