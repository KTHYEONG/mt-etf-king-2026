from __future__ import annotations

from datetime import date

import polars as pl
import pytest
from unittest import mock

from src.portfolio.constraints import gross_exposure
from src.portfolio.policy import PortfolioPolicy
from src.portfolio.sizing import (
    ConfidenceSizingConfig,
    LotteryExposureConfig,
    lottery_active,
    lottery_concentration_weights,
)
from src.universe.instruments import InstrumentAttributes, InstrumentMaster
from src.universe.taxonomy import Taxonomy


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
    return InstrumentMaster(attributes=attrs, panel_start=master.panel_start)


def test_lottery_active_fail_closed() -> None:
    cfg = LotteryExposureConfig(enabled=True)
    assert lottery_active("RISK_ON", True, cfg) is True
    assert lottery_active("STRONG_RISK_ON", True, cfg) is True
    assert lottery_active("RISK_ON", False, cfg) is False
    assert lottery_active("RISK_ON", None, cfg) is False
    assert lottery_active(None, True, cfg) is False
    assert lottery_active("NEUTRAL", True, cfg) is False
    assert lottery_active("RISK_OFF", True, cfg) is False
    assert lottery_active("RISK_ON", True, None) is False
    assert lottery_active("RISK_ON", True, LotteryExposureConfig(enabled=False)) is False


def test_lottery_concentration_weights_top1() -> None:
    cfg = LotteryExposureConfig(w_top=1.0)
    assert lottery_concentration_weights({}, cfg) == {}
    assert lottery_concentration_weights({"B": 0.10, "A": 0.10}, LotteryExposureConfig(w_top=1.0)) == {"A": 1.0}
    result = lottery_concentration_weights({"Z": 0.20, "A": 0.05}, cfg)
    assert set(result.keys()) == {"Z"}
    assert result["Z"] == pytest.approx(1.0, abs=1e-9)
    # No second name
    assert len(result) == 1


def test_allocate_lottery_top1_plus2_gross_two() -> None:
    master = _family_master()
    policy = PortfolioPolicy(
        lottery_config=LotteryExposureConfig(enabled=True),
        sizing_config=ConfidenceSizingConfig(),
        master=master,
        state_enabled=False,
    )
    dec = policy.allocate({"T1": 1.0, "T2": 0.1}, regime="RISK_ON", leverage_allowed=True)
    # final weights have exactly one positive key which is T2
    positive = {k: v for k, v in dec.weights.items() if v > 1e-9}
    assert len(positive) == 1
    assert "T2" in positive
    assert positive["T2"] == pytest.approx(1.0, abs=1e-9)
    # gross
    assert gross_exposure(dec.weights, {"T2": 2}) == pytest.approx(2.0, abs=1e-9)
    assert dec.gross == pytest.approx(2.0, abs=1e-9)
    # rationale contains lottery=1 and vehicle= and mult=
    assert dec.rationale is not None
    joined = " ".join(dec.rationale.values())
    assert "lottery=1" in joined
    assert "vehicle=" in joined
    assert "mult=" in joined


def test_allocate_lottery_skips_vehicle_conf_gate() -> None:
    master = _family_master()
    cfg = LotteryExposureConfig(enabled=True)
    policy = PortfolioPolicy(sizing_config=ConfidenceSizingConfig(), master=master, state_enabled=False, lottery_config=cfg)
    from src.portfolio.exposure import ExposureSelector

    captured: list[bool] = []
    orig = ExposureSelector.pick_vehicle

    def wrapped(self, tickers, leverage_allowed=None, confidence_low=False, regime=None, inverse_allowed=None):  # type: ignore[no-untyped-def]
        captured.append(bool(confidence_low))
        return orig(self, tickers, leverage_allowed=leverage_allowed, confidence_low=confidence_low, regime=regime, inverse_allowed=inverse_allowed)

    with mock.patch.object(ExposureSelector, "pick_vehicle", wrapped):
        low_scores = {"T1": 0.030, "T2": 0.029}
        captured.clear()
        dec = policy.allocate(low_scores, regime="RISK_ON", leverage_allowed=True)
        assert captured, "pick_vehicle not called"
        assert captured[-1] is False
        # final positive vehicle is T2
        positive = {k: v for k, v in dec.weights.items() if v > 1e-9}
        assert "T2" in positive

    # contrast: without lottery, confidence_low True
    policy2 = PortfolioPolicy(sizing_config=ConfidenceSizingConfig(), master=master, state_enabled=False)
    captured2: list[bool] = []

    def wrapped2(self, tickers, leverage_allowed=None, confidence_low=False, regime=None, inverse_allowed=None):  # type: ignore[no-untyped-def]
        captured2.append(bool(confidence_low))
        return orig(self, tickers, leverage_allowed=leverage_allowed, confidence_low=confidence_low, regime=regime, inverse_allowed=inverse_allowed)

    with mock.patch.object(ExposureSelector, "pick_vehicle", wrapped2):
        captured2.clear()
        dec2 = policy2.allocate(low_scores, regime="RISK_ON", leverage_allowed=True)
        assert captured2[-1] is True


def test_allocate_lottery_off_preserves_gross_cap_160() -> None:
    master = _family_master()
    policy_off = PortfolioPolicy(
        sizing_config=ConfidenceSizingConfig(),
        master=master,
        state_enabled=False,
        lottery_config=LotteryExposureConfig(enabled=False),
    )
    dec = policy_off.allocate({"T1": 1.0, "T2": 0.1}, regime="RISK_ON", leverage_allowed=True)
    # if final vehicle is T2 then gross <=1.60
    if "T2" in dec.weights and dec.weights["T2"] > 1e-9:
        assert gross_exposure(dec.weights, {"T2": 2}) <= 1.60 + 1e-9
    # rationale contains lottery=0 when attached disabled
    assert dec.rationale is not None
    joined = " ".join(dec.rationale.values())
    assert "lottery=0" in joined
    # if lottery_config is None, do not require lottery key
    policy_none = PortfolioPolicy(sizing_config=ConfidenceSizingConfig(), master=master, state_enabled=False, lottery_config=None)
    dec2 = policy_none.allocate({"T1": 1.0, "T2": 0.1}, regime="RISK_ON", leverage_allowed=True)
    # no exception and weights exist
    assert dec2.weights is not None


def test_allocate_lottery_risk_off_does_not_force_plus2() -> None:
    master = _family_master()
    policy = PortfolioPolicy(
        sizing_config=ConfidenceSizingConfig(),
        master=master,
        state_enabled=False,
        lottery_config=LotteryExposureConfig(enabled=True),
    )
    dec = policy.allocate({"T1": 1.0, "T2": 0.1}, regime="RISK_OFF", leverage_allowed=True)
    # lottery_active is False so path is confidence/tail, final positive keys have leverage 1
    for ticker in dec.weights:
        if dec.weights[ticker] > 1e-9:
            assert master.attributes[ticker].leverage_multiple == 1
    assert dec.rationale is not None
    joined = " ".join(dec.rationale.values())
    assert "lottery=0" in joined
    # also verify lottery_active false
    assert lottery_active("RISK_OFF", True, LotteryExposureConfig(enabled=True)) is False


def test_allocate_lottery_suppress_trim_keeps_exit() -> None:
    master = _family_master()
    policy = PortfolioPolicy(
        sizing_config=ConfidenceSizingConfig(),
        master=master,
        state_enabled=True,
        lottery_config=LotteryExposureConfig(enabled=True, suppress_trim=True),
    )
    # (a) TRIM coerced to HOLD
    # Infer via theme_states OVERHEATED? With lottery suppress_trim, TRIM should be coerced to HOLD so weight >0
    # Need to find a way to force TRIM: use infer_theme_proxy via OVERHEATED? Simpler: pass theme_states that cause TRIM for top ticker
    # The policy state logic: with theme_states {"T1": "OVERHEATED"} and previous HOLD, transition to TRIM, but suppress_trim should coerce to HOLD
    # So allocate with OVERHEATED on top canonical should yield weight >0 (not 0.5)
    # We allocate high-spread scores so top is T1, vehicle maps to T2
    dec_trim = policy.allocate({"T1": 1.0, "T2": 0.1}, regime="RISK_ON", leverage_allowed=True, theme_states={"T1": "OVERHEATED"})
    # check weight >0 for T2 (vehicle)
    # if TRIM suppressed, weight should be 1.0 not 0.5
    # Find positive weight
    pos_weights = {k: v for k, v in dec_trim.weights.items() if v > 1e-9}
    assert len(pos_weights) == 1
    w_val = next(iter(pos_weights.values()))
    # with trim suppressed, weight should be ~1.0 (before cap). Without suppression it would be 0.5.
    assert w_val == pytest.approx(1.0, abs=1e-9) or w_val > 0.75

    # (b) BREAKDOWN should remain 0.0 and rationale contains state=EXIT or WATCH
    policy2 = PortfolioPolicy(
        sizing_config=ConfidenceSizingConfig(),
        master=master,
        state_enabled=True,
        lottery_config=LotteryExposureConfig(enabled=True, suppress_trim=True),
    )
    dec_exit = policy2.allocate({"T1": 1.0, "T2": 0.1}, regime="RISK_ON", leverage_allowed=True, theme_states={"T1": "BREAKDOWN"})
    assert dec_exit.weights.get("T2", dec_exit.weights.get("T1", 1.0)) == pytest.approx(0.0, abs=1e-9)
    joined = " ".join(dec_exit.rationale.values()) if dec_exit.rationale else ""
    assert "state=EXIT" in joined or "state=WATCH" in joined


def test_lottery_config_from_yaml_fail_closed() -> None:
    cfg = LotteryExposureConfig.from_yaml({})
    assert cfg.enabled is False
    assert cfg.max_gross == pytest.approx(2.0)
    assert cfg.w_top == pytest.approx(1.0)
    assert cfg.suppress_vehicle_gate is True
    assert cfg.suppress_trim is True

    cfg2 = LotteryExposureConfig.from_yaml({"enabled": True, "max_gross": 2.0, "w_top": 1.0, "risk_on_regimes": ["RISK_ON", "STRONG_RISK_ON"]})
    assert cfg2.enabled is True
    assert cfg2.max_gross == pytest.approx(2.0)

    cfg3 = LotteryExposureConfig.from_yaml({"enabled": "x", "max_gross": "bad"})
    # should not raise
    assert cfg3.max_gross == pytest.approx(2.0)
    # enabled 'x' -> bool('x') True or fallback, but not raise
    assert isinstance(cfg3.enabled, bool)


def test_p13_allocate_unchanged_without_lottery() -> None:
    p = PortfolioPolicy(sizing_config=ConfidenceSizingConfig(), lottery_config=None)
    assert p.allocate({}).weights == {}
    # low-spread two-name scores without master: confidence_weights path len >=1 and if 2 names present top weight <0.85
    scores = {"A": 0.03, "B": 0.029}
    dec = p.allocate(scores)
    assert len(dec.weights) >= 1
    if len(dec.weights) == 2:
        top_w = max(dec.weights.values())
        assert top_w < 0.85
