"""SCENARIO-08-06"""
from datetime import date

import polars as pl

from src.portfolio.selection import select_positions
from src.universe.instruments import InstrumentAttributes, InstrumentMaster
from src.universe.taxonomy import Taxonomy


def _master_two_same_family() -> InstrumentMaster:
    panel = pl.DataFrame(
        [
            {"date": date(2026, 1, 2), "ticker": "A", "name": "KODEX 200", "underlying_index_name": "KOSPI 200"},
            {"date": date(2026, 1, 2), "ticker": "B", "name": "KODEX 레버리지", "underlying_index_name": "KOSPI 200"},
            {"date": date(2026, 1, 3), "ticker": "A", "name": "KODEX 200", "underlying_index_name": "KOSPI 200"},
            {"date": date(2026, 1, 3), "ticker": "B", "name": "KODEX 레버리지", "underlying_index_name": "KOSPI 200"},
        ]
    )
    taxonomy = Taxonomy(rules=[])
    master = InstrumentMaster.build(panel, taxonomy, {})
    # Force same family key for both
    # Manually override B's family to match A's
    attrs = dict(master.attributes)
    a_attr = attrs["A"]
    b_attr = attrs["B"]
    # b has leverage 2, but set family same as A
    new_b = InstrumentAttributes(
        ticker=b_attr.ticker,
        name=b_attr.name,
        issuer=b_attr.issuer,
        leverage_multiple=b_attr.leverage_multiple,
        leverage_family_key=a_attr.leverage_family_key,
        is_synthetic=b_attr.is_synthetic,
        is_hedged=b_attr.is_hedged,
        is_active=b_attr.is_active,
        index_key=b_attr.index_key,
        theme=b_attr.theme,
        first_seen=b_attr.first_seen,
        last_seen=b_attr.last_seen,
        left_censored=b_attr.left_censored,
        confidence=b_attr.confidence,
    )
    # Also set theme same to test precedence? But need family dedup first
    # Use same theme for simplicity
    new_master = InstrumentMaster(attributes={"A": a_attr, "B": new_b}, panel_start=master.panel_start)
    return new_master


def test_SCENARIO_08_06_family_dedup() -> None:  # noqa: N802
    master = _master_two_same_family()
    scores = {"A": 0.5, "B": 0.4}
    sel = select_positions(scores, master, max_per_theme=2, max_per_family=1)
    assert len(sel) == 1
    assert sel[0] == "A"
    # family dedup precedes theme dedup: even with different themes, family limits
    scores2 = {"A": 0.9, "B": 0.8}
    sel2 = select_positions(scores2, master, max_per_theme=1, max_per_family=1)
    assert len(sel2) == 1


import pytest


@pytest.mark.parametrize("scenario_id", ["SCENARIO-08-06"])
def test_SCENARIO_hyphen_wrapper(scenario_id: str) -> None:  # noqa: N802
    if scenario_id == "SCENARIO-08-06":
        test_SCENARIO_08_06_family_dedup()
