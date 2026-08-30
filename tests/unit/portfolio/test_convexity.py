from datetime import date
from src.portfolio.convexity import ConvexityHoldConfig, convexity_active, convexity_should_exit, resolve_convexity_vehicle
from src.universe.instruments import Confidence, InstrumentAttributes, InstrumentMaster


def test_convexity_active_fail_closed_and_crisis() -> None:
    scores = {"A": 0.20, "B": 0.10}
    on = ConvexityHoldConfig(enabled=True)
    assert convexity_active(True, "NEUTRAL", scores, on) is True
    assert convexity_active(True, "RISK_OFF", scores, on) is True
    assert convexity_active(True, "RISK_ON", scores, on) is True
    assert convexity_active(True, None, scores, on) is True
    assert convexity_active(True, "STRONG_RISK_OFF", scores, on) is False
    assert convexity_active(False, "NEUTRAL", scores, on) is False
    assert convexity_active(None, "NEUTRAL", scores, on) is False
    assert convexity_active(True, "NEUTRAL", scores, None) is False
    assert convexity_active(True, "NEUTRAL", scores, ConvexityHoldConfig(enabled=False)) is False
    assert convexity_active(True, "NEUTRAL", {}, on) is False
    parsed = ConvexityHoldConfig.from_yaml({})
    assert parsed.enabled is False
    assert convexity_active(True, "NEUTRAL", scores, parsed) is False


def test_convexity_active_respects_min_gap() -> None:
    tight = {"A": 0.10, "B": 0.06}
    wide = {"A": 0.10, "B": 0.04}
    cfg = ConvexityHoldConfig(enabled=True, min_gap=0.05)
    assert convexity_active(True, "NEUTRAL", tight, cfg) is False
    assert convexity_active(True, "NEUTRAL", wide, cfg) is True
    zero = ConvexityHoldConfig(enabled=True, min_gap=0.0)
    assert convexity_active(True, "NEUTRAL", tight, zero) is True


def test_convexity_should_exit_score_drop_and_crisis() -> None:
    cfg = ConvexityHoldConfig(enabled=True, score_drop_pct=0.30)
    assert convexity_should_exit(0.20, 0.20, "NEUTRAL", cfg) is False
    assert convexity_should_exit(0.16, 0.20, "NEUTRAL", cfg) is False
    assert convexity_should_exit(0.14, 0.20, "NEUTRAL", cfg) is True
    assert convexity_should_exit(0.20, 0.20, "STRONG_RISK_OFF", cfg) is True
    assert convexity_should_exit(0.20, None, "NEUTRAL", cfg) is False
    assert convexity_should_exit(None, 0.20, "NEUTRAL", cfg) is False
    assert convexity_should_exit(0.01, 0.20, "NEUTRAL", ConvexityHoldConfig(enabled=False)) is False


def _attr(ticker: str, fam: str, lev: int, conf: Confidence = Confidence.HIGH) -> InstrumentAttributes:
    d = date(2024, 1, 2)
    return InstrumentAttributes(ticker=ticker, name=ticker, issuer="삼성자산운용", leverage_multiple=lev, leverage_family_key=fam, is_synthetic=False, is_hedged=False, is_active=True, index_key="kospi 200", theme="EQUITY", first_seen=d, last_seen=d, left_censored=False, confidence=conf)


def test_resolve_convexity_vehicle_ignores_regime() -> None:
    master = InstrumentMaster(attributes={"T1": _attr("T1", "FAM", 1), "T2": _attr("T2", "FAM", 2)}, panel_start=date(2024, 1, 2))
    assert resolve_convexity_vehicle("T1", master, leverage_allowed=True, confidence_low=False) == "T2"
    assert resolve_convexity_vehicle("T1", master, leverage_allowed=False, confidence_low=False) == "T1"
    assert resolve_convexity_vehicle("T1", master, leverage_allowed=True, confidence_low=True) == "T1"
    assert resolve_convexity_vehicle("T1", None, leverage_allowed=True) == "T1"


def test_allocate_skips_capacity_demote_on_convexity() -> None:
    from src.portfolio.convexity import ConvexityHoldConfig
    from src.portfolio.policy import PortfolioPolicy
    from src.portfolio.sizing import ConfidenceSizingConfig
    from src.universe.instruments import Confidence, InstrumentAttributes, InstrumentMaster

    def _attr2(ticker: str, fam: str, lev: int) -> InstrumentAttributes:
        d = date(2024, 1, 2)
        return InstrumentAttributes(ticker=ticker, name=ticker, issuer="삼성자산운용", leverage_multiple=lev, leverage_family_key=fam, is_synthetic=False, is_hedged=False, is_active=True, index_key="kospi 200", theme="EQUITY", first_seen=d, last_seen=d, left_censored=False, confidence=Confidence.HIGH)

    master = InstrumentMaster(attributes={"T1": _attr2("T1", "FAM", 1), "T2": _attr2("T2", "FAM", 2)}, panel_start=date(2024, 1, 2))
    policy = PortfolioPolicy(master=master, sizing_config=ConfidenceSizingConfig(), lottery_config=None, convexity_config=ConvexityHoldConfig(enabled=True, skip_capacity_route=True, max_gross=2.0, w_top=1.0))
    tiny_adv = {"T1": 1.0, "T2": 1.0}
    dec = policy.allocate({"T1": 0.25, "X": 0.01}, capital=1_000_000_000.0, adv=tiny_adv, participation=0.01, current_weights={}, regime="NEUTRAL", leverage_allowed=True)
    pos = {k: v for k, v in dec.weights.items() if abs(float(v)) > 1e-9}
    assert "T2" in pos
    assert pos["T2"] > 0.5
    assert "T1" not in pos or abs(pos.get("T1", 0.0)) <= 1e-9
    off = policy.allocate({"T1": 0.25}, capital=1_000_000_000.0, adv=tiny_adv, participation=0.01, current_weights={}, regime="RISK_OFF", leverage_allowed=True)
    off_pos = {k: v for k, v in off.weights.items() if abs(float(v)) > 1e-9}
    assert "T2" in off_pos
