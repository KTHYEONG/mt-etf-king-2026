"""SCENARIO-08-09 SCENARIO-B2-03 SCENARIO-B2-04 SCENARIO-09-05 SCENARIO-09-06 SCENARIO-09-07"""
from datetime import date

import polars as pl

from src.portfolio.policy import PortfolioPolicy
from src.portfolio.sizing import ConfidenceSizingConfig
from src.universe.instruments import InstrumentAttributes, InstrumentMaster
from src.universe.taxonomy import Taxonomy


def test_SCENARIO_08_09_portfolio_policy_path_dependent() -> None:  # noqa: N802
    assert PortfolioPolicy.path_dependent is True
    p = PortfolioPolicy(sizing_config=ConfidenceSizingConfig())
    assert p.path_dependent is True
    dec = p.allocate({"A": 0.5, "B": 0.3, "C": 0.2})
    assert hasattr(dec, "weights")
    assert sum(dec.weights.values()) <= 1.0 + 1e-9
    # empty scores -> empty weights
    dec2 = p.allocate({})
    assert dec2.weights == {}


def test_SCENARIO_B2_03_breakdown_and_reset() -> None:  # noqa: N802
    # SCENARIO-B2-03: After allocate with theme_states forcing BREAKDOWN on top ticker, that ticker weight==0.0 and rationale contains 'state=EXIT' or 'state=WATCH'; reset_trackers clears
    p = PortfolioPolicy(sizing_config=ConfidenceSizingConfig())
    scores = {"AAA": 0.10, "BBB": 0.05, "CCC": 0.02}
    # force BREAKDOWN on AAA
    dec = p.allocate(scores, theme_states={"AAA": "BREAKDOWN"})
    assert dec.weights.get("AAA", 1.0) == 0.0
    assert "AAA" in dec.rationale  # type: ignore
    r = dec.rationale["AAA"]  # type: ignore
    assert "state=EXIT" in r or "state=WATCH" in r
    # reset should clear trackers
    p.reset_trackers()
    # next allocate should start HOLD - allocate without breakdown should give positive weight for AAA if top
    dec2 = p.allocate(scores)
    # AAA should now have positive weight (since HOLD)
    assert dec2.weights.get("AAA", 0) > 0


def test_SCENARIO_B2_04_cooldown_reenter() -> None:  # noqa: N802
    # SCENARIO-B2-04: Sequence EXIT then WATCH for cooldown-1 sessions then RECOVERY at >=3 yields RE_ENTER then HOLD
    p = PortfolioPolicy(sizing_config=ConfidenceSizingConfig())
    scores = {"TICK": 0.08, "OTHER": 0.01}
    # Step1: BREAKDOWN -> EXIT
    dec1 = p.allocate(scores, theme_states={"TICK": "BREAKDOWN"})
    assert dec1.weights["TICK"] == 0.0
    assert "state=EXIT" in dec1.rationale["TICK"]  # type: ignore
    # Step2: WATCH for cooldown-1 sessions (2 sessions) with non-RECOVERY
    for _ in range(2):
        dec = p.allocate(scores, theme_states={"TICK": "OVERHEATED"})
        assert dec.weights["TICK"] == 0.0
        # should be WATCH
        assert "state=WATCH" in dec.rationale["TICK"]  # type: ignore
    # Step3: RECOVERY at sessions_since_exit>=3 -> RE_ENTER
    dec3 = p.allocate(scores, theme_states={"TICK": "RECOVERY"})
    # RE_ENTER should have positive weight (m=1)
    assert dec3.weights["TICK"] > 0
    assert "state=RE_ENTER" in dec3.rationale["TICK"]  # type: ignore
    # Next session after RE_ENTER should be HOLD with positive weight
    dec4 = p.allocate(scores, theme_states={"TICK": "LEADING"})
    assert dec4.weights["TICK"] > 0
    assert "state=HOLD" in dec4.rationale["TICK"]  # type: ignore


import pytest


def _family_master() -> InstrumentMaster:
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
    attrs = {
        "T1": InstrumentAttributes(
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
        ),
        "T2": InstrumentAttributes(
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
        ),
    }
    return InstrumentMaster(attributes=attrs, panel_start=master.panel_start)


def test_SCENARIO_09_05_vehicle_remap_risk_on() -> None:  # noqa: N802
    master = _family_master()
    p = PortfolioPolicy(sizing_config=ConfidenceSizingConfig(), master=master, state_enabled=False)
    dec = p.allocate({"T1": 0.90, "T2": 0.01}, regime="RISK_ON", leverage_allowed=True)
    assert "T2" in dec.weights
    assert "T1" not in dec.weights
    assert dec.vehicles is not None
    assert dec.vehicles.get("T1") == "T2"
    assert dec.rationale is not None
    for why in dec.rationale.values():
        assert "vehicle=" in why
        assert "mult=" in why


def test_SCENARIO_09_06_unknown_leverage_plus_one_only() -> None:  # noqa: N802
    master = _family_master()
    p = PortfolioPolicy(sizing_config=ConfidenceSizingConfig(), master=master, state_enabled=False, max_gross_exposure=1.60)
    dec = p.allocate({"T1": 0.90, "T2": 0.01}, regime="RISK_ON", leverage_allowed=None)
    for ticker in dec.weights:
        assert master.attributes[ticker].leverage_multiple == 1
    assert dec.gross is not None
    assert dec.gross <= 1.60 + 1e-9


def test_SCENARIO_09_07_liquidity_demote_after_vehicle() -> None:  # noqa: N802
    master = _family_master()
    p = PortfolioPolicy(sizing_config=ConfidenceSizingConfig(), master=master, state_enabled=False)
    capital = 1_000_000_000.0
    participation = 0.01
    # T2 ADV tiny -> max weight ~0.001; T1 ADV large enough for full weight
    adv = {"T1": 1e12, "T2": 1e6}
    dec = p.allocate(
        {"T1": 0.90},
        capital=capital,
        adv=adv,
        participation=participation,
        regime="RISK_ON",
        leverage_allowed=True,
    )
    assert "T1" in dec.weights
    assert "T2" not in dec.weights
    max_w = adv["T1"] * participation / capital
    assert dec.weights["T1"] <= max_w + 1e-6


@pytest.mark.parametrize(
    "scenario_id",
    ["SCENARIO-08-09", "SCENARIO-B2-03", "SCENARIO-B2-04", "SCENARIO-09-05", "SCENARIO-09-06", "SCENARIO-09-07"],
)
def test_SCENARIO_hyphen_wrapper(scenario_id: str) -> None:  # noqa: N802
    if scenario_id == "SCENARIO-08-09":
        test_SCENARIO_08_09_portfolio_policy_path_dependent()
    if scenario_id == "SCENARIO-B2-03":
        test_SCENARIO_B2_03_breakdown_and_reset()
    if scenario_id == "SCENARIO-B2-04":
        test_SCENARIO_B2_04_cooldown_reenter()
    if scenario_id == "SCENARIO-09-05":
        test_SCENARIO_09_05_vehicle_remap_risk_on()
    if scenario_id == "SCENARIO-09-06":
        test_SCENARIO_09_06_unknown_leverage_plus_one_only()
    if scenario_id == "SCENARIO-09-07":
        test_SCENARIO_09_07_liquidity_demote_after_vehicle()
