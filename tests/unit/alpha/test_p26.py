def test_p26_registered_disables_crash_cash() -> None:
    from src.strategies.registry import STRATEGIES as BASELINES
    from src.alpha.sticky import StickyLeaderModel
    from src.strategies.ids import STICKY_MOM60_CONCENTRATED
    from src.strategies.sticky.overlays import overlay_param

    assert "sticky.mom60_concentrated" in BASELINES
    p26 = BASELINES["sticky.mom60_concentrated"]()
    p25 = BASELINES["sticky.house_money"]()
    assert isinstance(p26, StickyLeaderModel)
    assert p26.name == "sticky.mom60_concentrated"
    assert p25.name == "sticky.house_money"
    cfg = p26.config
    assert str(cfg.mom_col) == "mom_60"
    assert float(cfg.cash_drawdown) == 0.0
    assert float(cfg.min_gap) == 0.04
    assert int(cfg.min_hold) == 2
    assert float(cfg.impulse_gap) == 0.0
    assert cfg.only_plus_2 is True
    assert cfg.no_inverse is True
    assert cfg.collapse_family is False
    assert float(p25.config.cash_drawdown) == -0.12
    assert float(p25.config.min_gap) == 0.08
    assert overlay_param(STICKY_MOM60_CONCENTRATED, "arm", default=0.50) == 0.50
    assert overlay_param(STICKY_MOM60_CONCENTRATED, "lock_remaining", default=5) == 5
    assert not hasattr(p26, "allocate") or not callable(getattr(p26, "allocate", None))


def test_p26_score_ranks_mom60_and_keeps_crashed_leader() -> None:
    from datetime import date

    import polars as pl

    from src.alpha.base import DecisionContext
    from src.strategies.registry import STRATEGIES as BASELINES
    from src.portfolio.sizing import SizingScheme, weights_from_scores
    from src.universe.tournament import TournamentRules

    snap = pl.DataFrame(
        {
            "ticker": ["SLOW", "FAST"],
            "name": ["KODEX 레버리지", "KODEX 코스닥150레버리지"],
            "mom_20": [0.50, 0.10],
            "mom_60": [0.10, 0.40],
            "mom_5": [0.01, 0.02],
            "volume_expansion": [0.1, 0.1],
            "drawdown_20": [-0.20, 0.0],
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
    ctx = DecisionContext(
        decision_date=date(2026, 1, 2),
        regime=None,
        capital=1.0e9,
        held={"SLOW": 1.0},
        rules=rules,
    )
    p26 = BASELINES["sticky.mom60_concentrated"]()
    scores = p26.score(snap, ctx)
    assert scores, "crash-cash must stay disabled so scores remain non-empty"
    w = weights_from_scores(scores, SizingScheme.TOP1, k=1)
    assert set(w.keys()) <= {"SLOW", "FAST"}
    from src.portfolio.intent import CASH_INTENT

    p25 = BASELINES["sticky.house_money"]()
    empty = p25.score(snap, ctx)
    assert empty is CASH_INTENT
