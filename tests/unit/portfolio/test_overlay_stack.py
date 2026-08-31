def test_lottery_active_min_gap_fail_closed() -> None:
    from src.portfolio.sizing import LotteryExposureConfig, lottery_active

    scores = {"A": 0.12, "B": 0.10}
    cfg = LotteryExposureConfig(enabled=True, min_gap=0.05)
    assert lottery_active("RISK_ON", True, cfg, scores) is False
    wide = {"A": 0.12, "B": 0.04}
    assert lottery_active("RISK_ON", True, cfg, wide) is True
    legacy = LotteryExposureConfig(enabled=True, min_gap=0.0)
    assert lottery_active("RISK_ON", True, legacy, scores) is True
    assert lottery_active("RISK_ON", True, cfg, None) is False

def test_resolve_overlay_sizing_branch_lottery_first() -> None:
    from src.portfolio.convexity import ConvexityHoldConfig
    from src.portfolio.sizing import LotteryExposureConfig, resolve_overlay_sizing_branch

    scores = {"T1": 0.30, "T2": 0.10}
    lot = LotteryExposureConfig(enabled=True, w_top=1.0)
    conv = ConvexityHoldConfig(enabled=True, w_top=1.0)
    weights, lottery_branch, convexity_sizing = resolve_overlay_sizing_branch(
        scores,
        lottery_on=True,
        convexity_on=True,
        lottery_config=lot,
        convexity_config=conv,
    )
    assert lottery_branch is True
    assert convexity_sizing is False
    assert weights == {"T1": 1.0}

def test_resolve_overlay_sizing_branch_convexity_only() -> None:
    from src.portfolio.convexity import ConvexityHoldConfig
    from src.portfolio.sizing import resolve_overlay_sizing_branch

    scores = {"T1": 0.30, "T2": 0.10}
    conv = ConvexityHoldConfig(enabled=True, w_top=0.8)
    weights, lottery_branch, convexity_sizing = resolve_overlay_sizing_branch(
        scores,
        lottery_on=False,
        convexity_on=True,
        lottery_config=None,
        convexity_config=conv,
    )
    assert lottery_branch is False
    assert convexity_sizing is True
    assert weights == {"T1": 0.8}

def test_allocate_lottery_plus_convexity_uses_lottery_weights() -> None:
    from datetime import date

    from src.portfolio.convexity import ConvexityHoldConfig
    from src.portfolio.policy import PortfolioPolicy
    from src.portfolio.sizing import ConfidenceSizingConfig, LotteryExposureConfig
    from src.universe.instruments import Confidence, InstrumentAttributes, InstrumentMaster

    def attr(ticker: str, fam: str, lev: int) -> InstrumentAttributes:
        d = date(2024, 1, 2)
        return InstrumentAttributes(
            ticker=ticker,
            name=ticker,
            issuer="삼성자산운용",
            leverage_multiple=lev,
            leverage_family_key=fam,
            is_synthetic=False,
            is_hedged=False,
            is_active=True,
            index_key="kospi 200",
            theme="EQUITY",
            first_seen=d,
            last_seen=d,
            left_censored=False,
            confidence=Confidence.HIGH,
        )

    master = InstrumentMaster(
        attributes={"T1": attr("T1", "FAM", 1), "T2": attr("T2", "FAM", 2), "X": attr("X", "XFAM", 1)},
        panel_start=date(2024, 1, 2),
    )
    policy = PortfolioPolicy(
        master=master,
        sizing_config=ConfidenceSizingConfig(k=3),
        lottery_config=LotteryExposureConfig(enabled=True, w_top=1.0, suppress_vehicle_gate=True),
        convexity_config=ConvexityHoldConfig(enabled=True, w_top=1.0, skip_capacity_route=True, max_gross=2.0),
    )
    scores = {"T1": 0.30, "X": 0.05, "Y": 0.04}
    dec = policy.allocate(
        scores,
        capital=1_000_000_000.0,
        adv={"T1": 1e10, "T2": 1e10, "X": 1e10},
        participation=0.01,
        current_weights={},
        regime="STRONG_RISK_ON",
        leverage_allowed=True,
    )
    pos = {k: v for k, v in dec.weights.items() if abs(float(v)) > 1e-9}
    assert "T2" in pos
    assert pos["T2"] >= 0.99
    assert sum(pos.values()) <= 1.0 + 1e-9
