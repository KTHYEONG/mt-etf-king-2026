def test_p23_registered_p21_sticky_with_allocate() -> None:
    from src.alpha.baselines import BASELINES
    from src.alpha.sticky import StickyLeaderModel
    from src.portfolio.split_fill import SplitFillStickyModel

    assert "P23" in BASELINES
    p23 = BASELINES["P23"]()
    assert isinstance(p23, SplitFillStickyModel)
    assert p23.name == "P23"
    inner = getattr(p23, "_inner", None) or getattr(p23, "inner", None)
    assert inner is None or isinstance(inner, StickyLeaderModel)
    cfg = getattr(p23, "config")
    assert float(cfg.min_gap) == 0.08
    assert int(cfg.min_hold) == 3
    assert float(cfg.impulse_gap) == 0.04
    assert float(cfg.cash_drawdown) == -0.12
    assert bool(cfg.collapse_family) is True
    assert abs(float(cfg.lock_level) - 0.40) < 1e-12
    assert callable(getattr(p23, "allocate", None))
    p22 = BASELINES["P22"]()
    assert float(getattr(p22, "config").min_gap) == 0.0
    assert abs(float(getattr(p22, "config").lock_level) - 0.50) < 1e-12
    p21 = BASELINES["P21"]()
    assert not hasattr(p21, "allocate") or not callable(getattr(p21, "allocate", None))


def test_p23_allocate_splits_theme_leader() -> None:
    from datetime import date

    import polars as pl

    from src.alpha.base import DecisionContext
    from src.alpha.baselines import BASELINES
    from src.universe.tournament import TournamentRules

    snap = pl.DataFrame(
        {
            "ticker": ["494310", "122630"],
            "name": ["KODEX 반도체레버리지", "KODEX 레버리지"],
            "underlying_index_name": ["KRX 반도체", "코스피 200"],
            "trading_value": [4.56e10, 6.6e12],
            "mom_20": [0.55, 0.21],
            "mom_5": [0.16, 0.05],
            "volume_expansion": [2.0, 1.2],
            "drawdown_20": [0.0, 0.0],
        }
    )
    rules = TournamentRules(
        name="t",
        start_date=date(2025, 9, 22),
        end_date=date(2025, 9, 22),
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
    ctx = DecisionContext(decision_date=date(2025, 9, 22), regime=None, capital=1_000_000_000.0, held={}, rules=rules)
    p23 = BASELINES["P23"]()
    scores = p23.score(snap, ctx)
    assert "494310" in scores
    alloc = p23.allocate(
        scores,
        adv={"494310": 4.56e10, "122630": 6.6e12},
        participation=0.01,
        capital=1.0e9,
        current_weights={},
    )
    weights = dict(getattr(alloc, "weights", alloc))
    assert "494310" in weights
    assert "122630" in weights
    assert weights["494310"] < 1.0
    assert abs(sum(weights.values()) - 1.0) < 1e-9
