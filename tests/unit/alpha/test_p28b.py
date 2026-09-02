def test_apply_abs_mom_cash_returns_cash_when_all_nonpos() -> None:
    from src.alpha.sticky import StickyLeaderConfig, apply_abs_mom_cash
    from src.portfolio.intent import CASH_INTENT, PortfolioIntent

    cfg = StickyLeaderConfig(mom_col="mom_60")
    scores = {"AAA": -0.10, "BBB": 0.0}
    out = apply_abs_mom_cash(scores, cfg)
    assert isinstance(out, PortfolioIntent)
    assert out.kind == CASH_INTENT.kind


def test_apply_abs_mom_cash_passes_when_max_positive() -> None:
    from src.alpha.sticky import StickyLeaderConfig, apply_abs_mom_cash
    from src.portfolio.intent import PortfolioIntent

    cfg = StickyLeaderConfig(mom_col="mom_60")
    scores = {"AAA": -0.10, "BBB": 0.05}
    out = apply_abs_mom_cash(scores, cfg)
    assert not isinstance(out, PortfolioIntent)
    assert out == {"AAA": -0.10, "BBB": 0.05}


def test_apply_abs_mom_cash_preserves_cash_intent() -> None:
    from src.alpha.sticky import StickyLeaderConfig, apply_abs_mom_cash
    from src.portfolio.intent import CASH_INTENT

    cfg = StickyLeaderConfig()
    out = apply_abs_mom_cash(CASH_INTENT, cfg)
    assert out is CASH_INTENT


def test_p28b_factory_enables_hold_and_abs_mom() -> None:
    from src.alpha.baselines import BASELINES
    from src.alpha.sticky import StickyLeaderModel

    assert "P28B" in BASELINES
    p27 = BASELINES["P27"]()
    p28b = BASELINES["P28B"]()
    assert isinstance(p28b, StickyLeaderModel)
    assert p28b.name == "P28B"
    assert bool(p28b.config.same_leader_hold) is True
    assert str(p28b.config.mom_col) == str(p27.config.mom_col) == "mom_60"
    assert float(p28b.config.min_gap) == float(p27.config.min_gap) == 0.04
    assert int(p28b.config.min_hold) == int(p27.config.min_hold) == 2


def test_p28b_score_cash_when_all_mom_nonpos() -> None:
    from datetime import date

    import polars as pl

    from src.alpha.base import DecisionContext
    from src.alpha.baselines import BASELINES
    from src.portfolio.intent import CASH_INTENT, PortfolioIntent
    from src.universe.tournament import TournamentRules

    snap = pl.DataFrame(
        {
            "ticker": ["LEV1", "LEV2"],
            "name": ["KODEX 레버리지", "KODEX 코스닥150레버리지"],
            "mom_60": [-0.05, -0.02],
            "mom_5": [0.0, 0.0],
            "volume_expansion": [0.1, 0.1],
            "drawdown_20": [0.0, 0.0],
        }
    )
    rules = TournamentRules(
        name="t",
        start_date=date(2026, 9, 21),
        end_date=date(2026, 11, 13),
        initial_capital=1_000_000_000,
        category="autonomous",
        leverage_allowed=True,
        inverse_allowed=True,
        max_weight=1.0,
        cash_allowed=True,
        sponsor_etf_only=True,
        manifest_path=None,
        issuer_whitelist=None,
        commission_bps=3.0,
        slippage_bps=5.0,
        max_order_to_adv=0.01,
        stress_grid=(0.01, 0.02, 0.05),
    )
    ctx = DecisionContext(decision_date=date(2026, 1, 2), regime=None, capital=1.0e9, held={}, rules=rules)
    p28b = BASELINES["P28B"]()
    out = p28b.score(snap, ctx)
    assert isinstance(out, PortfolioIntent)
    assert out.kind == CASH_INTENT.kind
