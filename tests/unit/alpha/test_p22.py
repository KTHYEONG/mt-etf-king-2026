def test_p22_registered_min_gap_zero() -> None:
    from src.alpha.baselines import BASELINES
    from src.alpha.sticky import StickyLeaderModel

    assert "P22" in BASELINES
    p22 = BASELINES["P22"]()
    assert isinstance(p22, StickyLeaderModel)
    assert getattr(p22, "name", "") == "P22"
    cfg = getattr(p22, "config")
    assert float(cfg.min_gap) == 0.0
    assert int(cfg.min_hold) == 0
    assert bool(getattr(cfg, "collapse_family", False)) is True
    assert float(cfg.impulse_gap) == 0.0
    assert float(cfg.cash_drawdown) == 0.0
    assert float(cfg.lock_level) == 0.50
    assert cfg.only_plus_2 is True
    p20 = BASELINES["P20"]()
    p21 = BASELINES["P21"]()
    assert float(getattr(p20, "config").min_gap) == 0.08
    assert int(getattr(p20, "config").min_hold) == 3
    assert float(getattr(p21, "config").impulse_gap) == 0.04
    assert float(getattr(p21, "config").cash_drawdown) == -0.12
    assert not hasattr(p22, "allocate") or not callable(getattr(p22, "allocate", None))


def test_collapse_plus2_by_family_keeps_max_adv() -> None:
    import math
    import polars as pl
    from src.alpha.sticky import collapse_plus2_by_family

    snap = pl.DataFrame(
        {
            "ticker": ["LIQ", "THIN", "KQ", "ORPH"],
            "underlying_index_name": ["코스피 200", "코스피 200", "코스닥 150", None],
            "trading_value": [9.0e11, 1.0e9, 2.0e11, 3.0e10],
            "mom_20": [0.20, 0.22, 0.18, 0.30],
        }
    )
    scores = {"LIQ": 0.20, "THIN": 0.22, "KQ": 0.18, "ORPH": 0.30}
    out = collapse_plus2_by_family(scores, snap)
    assert set(out.keys()) == {"LIQ", "KQ", "ORPH"}
    assert "THIN" not in out
    assert out["LIQ"] == 0.20
    assert collapse_plus2_by_family({}, snap) == {}
    assert collapse_plus2_by_family(scores, None) == {}  # type: ignore[arg-type]
    nan_snap = snap.with_columns(pl.lit(float("nan")).alias("trading_value"))
    nan_out = collapse_plus2_by_family(scores, nan_snap)
    assert "THIN" in nan_out and math.isfinite(nan_out["THIN"])


def test_apply_sticky_leader_min_gap_zero_switches() -> None:
    from src.alpha.sticky import StickyLeaderConfig, apply_sticky_leader
    from src.portfolio.sizing import SizingScheme, weights_from_scores

    cfg = StickyLeaderConfig(min_gap=0.0, min_hold=0)
    scores = {"HOLD": 0.20, "CHAL": 0.21}
    out = apply_sticky_leader(scores, "HOLD", cfg, hold_len=1)
    w = weights_from_scores(out, SizingScheme.TOP1, k=1)
    assert set(w.keys()) == {"CHAL"}
    tied = apply_sticky_leader({"HOLD": 0.20, "CHAL": 0.20}, "HOLD", cfg, hold_len=1)
    w_tied = weights_from_scores(tied, SizingScheme.TOP1, k=1)
    assert set(w_tied.keys()) == {"HOLD"}
    sticky = StickyLeaderConfig(min_gap=0.08, min_hold=0)
    held = apply_sticky_leader(scores, "HOLD", sticky, hold_len=10)
    w_held = weights_from_scores(held, SizingScheme.TOP1, k=1)
    assert set(w_held.keys()) == {"HOLD"}


def test_p22_score_collapses_then_top1() -> None:
    import polars as pl
    from datetime import date
    from src.alpha.base import DecisionContext
    from src.alpha.baselines import BASELINES
    from src.portfolio.sizing import SizingScheme, weights_from_scores
    from src.universe.tournament import TournamentRules

    snap = pl.DataFrame(
        {
            "ticker": ["122630", "123320", "233740"],
            "name": ["KODEX 레버리지", "TIGER 레버리지", "KODEX 코스닥150레버리지"],
            "underlying_index_name": ["코스피 200", "코스피 200", "코스닥 150"],
            "trading_value": [5.0e12, 1.0e10, 2.0e12],
            "mom_20": [0.21, 0.22, 0.40],
            "mom_5": [0.01, 0.02, 0.03],
            "volume_expansion": [1.1, 1.2, 1.3],
            "drawdown_20": [0.0, 0.0, 0.0],
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
    ctx = DecisionContext(decision_date=date(2025, 9, 22), regime=None, capital=1_000_000_000.0, held={"122630": 1.0}, rules=rules)
    p22 = BASELINES["P22"]()
    scores = p22.score(snap, ctx)
    assert "123320" not in scores
    w = weights_from_scores(scores, SizingScheme.TOP1, k=1)
    assert set(w.keys()) == {"233740"}
