"""SCENARIO-10-01 SCENARIO-10-02 SCENARIO-10-03"""
from datetime import date
import polars as pl
from src.portfolio.selection import family_canonical_scores
from src.universe.instruments import InstrumentAttributes, InstrumentMaster
from src.universe.taxonomy import Taxonomy
import pytest

def _master_same_family(t1="T1", t2="T2", lev1=1, lev2=2, synth2=False):
    panel = pl.DataFrame([
        {"date": date(2026,1,2), "ticker": t1, "name": "KODEX 200", "underlying_index_name": "KOSPI 200"},
        {"date": date(2026,1,2), "ticker": t2, "name": "KODEX 레버리지", "underlying_index_name": "KOSPI 200"},
    ])
    tax = Taxonomy(rules=[])
    master = InstrumentMaster.build(panel, tax, {})
    a = master.attributes[t1]
    b = master.attributes[t2]
    # force same family
    new_a = InstrumentAttributes(ticker=a.ticker, name=a.name, issuer=a.issuer, leverage_multiple=lev1, leverage_family_key="FAM", is_synthetic=a.is_synthetic, is_hedged=a.is_hedged, is_active=a.is_active, index_key=a.index_key, theme=a.theme, first_seen=a.first_seen, last_seen=a.last_seen, left_censored=a.left_censored, confidence=a.confidence)
    new_b = InstrumentAttributes(ticker=b.ticker, name=b.name, issuer=b.issuer, leverage_multiple=lev2, leverage_family_key="FAM", is_synthetic=synth2, is_hedged=b.is_hedged, is_active=b.is_active, index_key=b.index_key, theme=b.theme, first_seen=b.first_seen, last_seen=b.last_seen, left_censored=b.left_censored, confidence=b.confidence)
    return InstrumentMaster(attributes={t1: new_a, t2: new_b}, panel_start=master.panel_start)

def _master_two_families():
    panel = pl.DataFrame([
        {"date": date(2026,1,2), "ticker": "A1", "name": "KODEX 200", "underlying_index_name": "KOSPI 200"},
        {"date": date(2026,1,2), "ticker": "A2", "name": "KODEX 레버리지", "underlying_index_name": "KOSPI 200"},
        {"date": date(2026,1,2), "ticker": "B1", "name": "KODEX 코스닥", "underlying_index_name": "KOSDAQ 150"},
        {"date": date(2026,1,2), "ticker": "B2", "name": "KODEX 코스닥 레버리지", "underlying_index_name": "KOSDAQ 150"},
    ])
    tax = Taxonomy(rules=[])
    master = InstrumentMaster.build(panel, tax, {})
    # build custom families
    attrs = {}
    for t, fam, lev in [("A1","FAMA",1), ("A2","FAMA",2), ("B1","FAMB",1), ("B2","FAMB",2)]:
        base_attr = master.attributes[t]
        attrs[t] = InstrumentAttributes(ticker=t, name=base_attr.name, issuer=base_attr.issuer, leverage_multiple=lev, leverage_family_key=fam, is_synthetic=False, is_hedged=False, is_active=True, index_key=fam, theme="Theme", first_seen=base_attr.first_seen, last_seen=base_attr.last_seen, left_censored=False, confidence=base_attr.confidence)
    return InstrumentMaster(attributes=attrs, panel_start=master.panel_start)

def test_SCENARIO_10_01_canonical_plus1_preferred() -> None:
    master = _master_same_family("C1","C2",1,2)
    scores = {"C1": 0.1, "C2": 0.9}
    out = family_canonical_scores(scores, master)
    assert "C1" in out and "C2" not in out
    assert out["C1"] == 0.1

def test_SCENARIO_10_02_two_families_plus1() -> None:
    master = _master_two_families()
    scores = {"A1": 0.5, "A2": 0.9, "B1": 0.4, "B2": 0.8}
    out = family_canonical_scores(scores, master)
    assert len(out) == 2
    assert set(out.keys()) == {"A1","B1"}
    for k in out:
        assert master.attributes[k].leverage_multiple == 1
    # order preserved: A1 higher than B1
    assert out["A1"] > out["B1"]

def test_SCENARIO_10_03_only_leveraged() -> None:
    master = _master_same_family("L1","L2",2, -2 if False else 2)
    # create family with only 2x members: two tickers both +2 but one synthetic
    from src.universe.instruments import InstrumentAttributes
    panel = pl.DataFrame([
        {"date": date(2026,1,2), "ticker": "X1", "name": "ETF X", "underlying_index_name": "IDX"},
        {"date": date(2026,1,2), "ticker": "X2", "name": "ETF X 레버리지", "underlying_index_name": "IDX"},
    ])
    tax = Taxonomy(rules=[])
    m = InstrumentMaster.build(panel, tax, {})
    # override both to same family NO +1
    a = m.attributes["X1"]
    b = m.attributes["X2"]
    attrs = {
        "X1": InstrumentAttributes(ticker="X1", name=a.name, issuer=a.issuer, leverage_multiple=2, leverage_family_key="FAMX", is_synthetic=False, is_hedged=False, is_active=True, index_key="FAMX", theme="T", first_seen=a.first_seen, last_seen=a.last_seen, left_censored=False, confidence=a.confidence),
        "X2": InstrumentAttributes(ticker="X2", name=b.name, issuer=b.issuer, leverage_multiple=2, leverage_family_key="FAMX", is_synthetic=True, is_hedged=False, is_active=True, index_key="FAMX", theme="T", first_seen=b.first_seen, last_seen=b.last_seen, left_censored=False, confidence=b.confidence),
    }
    master2 = InstrumentMaster(attributes=attrs, panel_start=m.panel_start)
    scores = {"X1": 0.2, "X2": 0.9}
    out = family_canonical_scores(scores, master2)
    assert len(out) == 1
    assert "X1" in out  # non-synthetic preferred
    assert family_canonical_scores({}, master2) == {}

@pytest.mark.parametrize("scenario_id", ["SCENARIO-10-01","SCENARIO-10-02","SCENARIO-10-03"])
def test_SCENARIO_hyphen_wrapper(scenario_id: str) -> None:
    if scenario_id == "SCENARIO-10-01":
        test_SCENARIO_10_01_canonical_plus1_preferred()
    elif scenario_id == "SCENARIO-10-02":
        test_SCENARIO_10_02_two_families_plus1()
    elif scenario_id == "SCENARIO-10-03":
        test_SCENARIO_10_03_only_leveraged()
