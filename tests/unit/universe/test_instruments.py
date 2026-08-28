from __future__ import annotations

from datetime import date
from pathlib import Path

import polars as pl

from src.universe.instruments import (
    InstrumentMaster,
    load_sponsor_brand_map,
    resolve_issuer,
    resolve_leverage,
)
from src.universe.taxonomy import Taxonomy, normalize_index_key


def test_scenario_04_01_resolve_leverage() -> None:
    """SCENARIO-04-01"""
    assert resolve_leverage("KODEX 200선물인버스2X") == (-2, resolve_leverage("KODEX 200선물인버스2X")[1])
    assert resolve_leverage("KODEX 200선물인버스2X")[0] == -2
    assert resolve_leverage("KIWOOM 미국달러선물인버스2X")[0] == -2
    assert resolve_leverage("ACE 인버스")[0] == -1
    assert resolve_leverage("HANARO 200선물인버스")[0] == -1
    assert resolve_leverage("ACE 레버리지")[0] == 2
    assert resolve_leverage("TIGER 미국필라델피아반도체레버리지(합성)")[0] == 2
    assert resolve_leverage("ACE 삼성전자단일종목레버리지")[0] == 2
    assert resolve_leverage("KODEX 200")[0] == 1
    assert resolve_leverage("1Q 200액티브")[0] == 1
    assert resolve_leverage("KODEX 200선물인버스2X")[0] != 2


def test_scenario_04_19_low_confidence_on_contradictory_name() -> None:
    """SCENARIO-04-19: INV-19 — 모순 토큰은 Confidence.LOW."""
    from src.universe.instruments import Confidence

    lev, conf = resolve_leverage("ACE 인버스 레버리지")
    assert conf is Confidence.LOW
    assert lev == -1


def test_scenario_04_02_instrument_master_and_issuer() -> None:
    """SCENARIO-04-02"""
    d0 = date(2026, 8, 13)
    d1 = date(2026, 8, 14)
    d2 = date(2026, 8, 18)
    rows = [
        {"date": d0, "ticker": "A", "name": "KODEX 200", "underlying_index_name": "코스피 200"},
        {"date": d1, "ticker": "A", "name": "KODEX 200", "underlying_index_name": "코스피 200"},
        {"date": d2, "ticker": "A", "name": "KODEX 200", "underlying_index_name": "코스피 200"},
        {"date": d1, "ticker": "B", "name": "ACE 인버스", "underlying_index_name": "코스피 200"},
        {"date": d2, "ticker": "B", "name": "ACE 인버스", "underlying_index_name": "코스피 200"},
        {"date": d0, "ticker": "C", "name": "WON 200", "underlying_index_name": "코스피 200"},
        {"date": d1, "ticker": "C", "name": "WON 200", "underlying_index_name": "코스피 200"},
    ]
    panel = pl.DataFrame(rows)
    taxonomy = Taxonomy(rules=[])
    brand_map = load_sponsor_brand_map(Path("configs/sponsor_brands.yaml"))
    master = InstrumentMaster.build(panel, taxonomy, brand_map)

    assert master.attributes["A"].first_seen == d0
    assert master.attributes["A"].last_seen == d2
    assert master.attributes["A"].left_censored is True
    assert master.attributes["B"].first_seen == d1
    assert master.attributes["B"].left_censored is False
    assert master.attributes["C"].last_seen == d1

    assert brand_map["KODEX"] == "삼성자산운용"
    assert resolve_issuer("KODEX 200", brand_map) == "삼성자산운용"
    assert resolve_issuer("WON 200", brand_map) == "UNKNOWN"

    expected_key = normalize_index_key("코스피 200")
    assert master.attributes["A"].leverage_family_key == expected_key
    assert master.attributes["A"].index_key == expected_key
