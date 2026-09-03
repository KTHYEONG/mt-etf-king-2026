from __future__ import annotations


def test_apply_capacity_filter_drops_unfillable() -> None:
    import polars as pl

    from src.strategies.sticky.capacity import apply_capacity_filter

    snap = pl.DataFrame(
        {
            "ticker": ["ILLIQ", "LIQ"],
            "name": ["TIGER 차이나전기차레버리지(합성)", "KODEX 반도체레버리지"],
            "trading_value": [4.56e8, 4.56e10],
            "mom_60": [0.70, 0.53],
        }
    )
    scores = {"ILLIQ": 0.70, "LIQ": 0.53}
    out = apply_capacity_filter(
        scores,
        snap,
        capital=1_000_000_000.0,
        max_order_to_adv=0.01,
        min_fill_ratio=0.25,
        sleeve_weight=0.95,
    )
    assert "ILLIQ" not in out
    assert out.get("LIQ") == 0.53
    disabled = apply_capacity_filter(
        scores, snap, capital=1_000_000_000.0, max_order_to_adv=0.01, min_fill_ratio=0.0
    )
    assert disabled == scores


def test_filter_plus2_scores_excludes_synthetic_when_enabled() -> None:
    import polars as pl

    from src.strategies.sticky.model import StickyLeaderConfig, filter_plus2_scores

    snap = pl.DataFrame(
        {
            "ticker": ["SYN", "CASH"],
            "name": ["TIGER 차이나전기차레버리지(합성)", "KODEX 반도체레버리지"],
            "mom_60": [0.90, 0.40],
            "trading_value": [1e11, 1e11],
        }
    )
    raw = filter_plus2_scores(snap, StickyLeaderConfig(mom_col="mom_60", only_plus_2=True, no_inverse=True))
    assert raw.get("SYN") == 0.90
    assert raw.get("CASH") == 0.40
    filtered = filter_plus2_scores(
        snap,
        StickyLeaderConfig(mom_col="mom_60", only_plus_2=True, no_inverse=True, exclude_synthetic=True),
    )
    assert "SYN" not in filtered
    assert filtered.get("CASH") == 0.40


def test_p30_score_prefers_fillable_equity_over_illiquid_synth() -> None:
    from datetime import date

    import polars as pl

    from src.alpha.base import DecisionContext
    from src.alpha.baselines import BASELINES
    from src.universe.tournament import TournamentRules

    snap = pl.DataFrame(
        {
            "ticker": ["456680", "494310"],
            "name": ["TIGER 차이나전기차레버리지(합성)", "KODEX 반도체레버리지"],
            "mom_60": [0.696, 0.526],
            "mom_5": [0.05, 0.08],
            "volume_expansion": [1.0, 1.0],
            "trading_value": [456_288_716.0, 45_598_866_119.0],
        }
    )
    rules = TournamentRules(
        name="t",
        start_date=date(2025, 9, 22),
        end_date=date(2025, 11, 14),
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
        decision_date=date(2025, 9, 22),
        regime=None,
        capital=1.0e9,
        held={},
        rules=rules,
    )
    p27 = BASELINES["P27"]()
    p30 = BASELINES["P30"]()
    s27 = p27.score(snap, ctx)
    s30 = p30.score(snap, ctx)
    assert isinstance(s27, dict) and isinstance(s30, dict)
    top27 = sorted(s27.items(), key=lambda kv: (-float(kv[1]), kv[0]))[0][0]
    top30 = sorted(s30.items(), key=lambda kv: (-float(kv[1]), kv[0]))[0][0]
    assert top27 == "456680"
    assert top30 == "494310"
    assert "456680" not in s30


def test_p30_factory_and_exposure_wiring() -> None:
    from src.alpha.baselines import BASELINES
    from src.portfolio.constraints import load_p27_exposure_limits, resolve_exposure_limits_for_model
    from src.strategies.ids import STICKY_FILLABLE_MOM60
    from src.strategies.sticky.model import DEFAULT_EXCLUDE_NAME_TOKENS
    from src.strategies.registry import resolve_strategy_id
    from src.strategies.sticky.factories import FACTORY_REGISTRY

    assert resolve_strategy_id("P30") == STICKY_FILLABLE_MOM60
    assert "P30" in BASELINES
    model = BASELINES["P30"]()
    assert model.name == "P30"
    assert bool(getattr(model.config, "exclude_synthetic", False)) is True
    assert abs(float(getattr(model.config, "min_fill_ratio", 0.0)) - 0.25) < 1e-12
    assert str(model.config.mom_col) == "mom_60"
    assert float(model.config.min_gap) == 0.04
    assert int(model.config.min_hold) == 2
    assert float(model.config.impulse_gap) == 0.0
    assert tuple(model.config.exclude_name_tokens) == DEFAULT_EXCLUDE_NAME_TOKENS
    assert STICKY_FILLABLE_MOM60 in FACTORY_REGISTRY
    assert resolve_exposure_limits_for_model("P30", comparison_mode="full_strategy_own") == load_p27_exposure_limits()
    p27 = BASELINES["P27"]()
    assert bool(getattr(p27.config, "exclude_synthetic", False)) is False
    assert float(getattr(p27.config, "min_fill_ratio", 0.0) or 0.0) == 0.0


def test_p27_unchanged_allows_synth_and_zero_min_fill() -> None:
    from datetime import date

    import polars as pl

    from src.alpha.base import DecisionContext
    from src.alpha.baselines import BASELINES
    from src.universe.tournament import TournamentRules

    snap = pl.DataFrame(
        {
            "ticker": ["SYN"],
            "name": ["TIGER 차이나전기차레버리지(합성)"],
            "mom_60": [0.50],
            "mom_5": [0.01],
            "volume_expansion": [1.0],
            "trading_value": [1.0e8],
        }
    )
    rules = TournamentRules(
        name="t",
        start_date=date(2025, 9, 22),
        end_date=date(2025, 11, 14),
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
    ctx = DecisionContext(decision_date=date(2025, 9, 22), regime=None, capital=1.0e9, held={}, rules=rules)
    p27 = BASELINES["P27"]()
    out = p27.score(snap, ctx)
    assert isinstance(out, dict)
    assert out.get("SYN") == 0.50


def test_capacity_filter_keeps_missing_adv_fail_open() -> None:
    import math

    import polars as pl

    from src.strategies.sticky.capacity import apply_capacity_filter

    snap = pl.DataFrame(
        {
            "ticker": ["UNK", "TINY"],
            "name": ["KODEX 레버리지", "ILLIQ 레버리지"],
            "trading_value": [None, 1.0e8],
            "mom_60": [0.10, 0.90],
        }
    )
    scores = {"UNK": 0.10, "TINY": 0.90}
    out = apply_capacity_filter(
        scores,
        snap,
        capital=1_000_000_000.0,
        max_order_to_adv=0.01,
        min_fill_ratio=0.25,
    )
    assert out.get("UNK") == 0.10
    assert "TINY" not in out
    bad_phi = apply_capacity_filter(scores, snap, capital=1e9, max_order_to_adv=0.0, min_fill_ratio=0.25)
    assert bad_phi == scores
    nan_cap = apply_capacity_filter(scores, snap, capital=float("nan"), max_order_to_adv=0.01, min_fill_ratio=0.25)
    assert nan_cap == scores
    _ = math.isfinite


def test_p30_sticky_module_line_budgets() -> None:
    from pathlib import Path

    caps = {
        "src/strategies/sticky/capacity.py": 200,
        "src/strategies/sticky/model.py": 700,
    }
    offenders: list[str] = []
    for rel, limit in caps.items():
        path = Path(rel)
        assert path.exists(), rel
        with path.open(encoding="utf-8") as fh:
            count = sum(1 for _ in fh)
        if count > limit:
            offenders.append(f"{rel}:{count}>{limit}")
    assert offenders == []


def test_fillable_sticky_session_cache_widens_score_snapshots() -> None:
    from datetime import date

    import polars as pl

    from src.alpha.baselines import BASELINES
    from src.alpha.base import DecisionContext
    from src.backtest.costs import CostConfig
    from src.backtest.engine import BacktestConfig
    from src.backtest.session_cache import build_session_cache
    from src.portfolio.sizing import SizingScheme
    from src.universe.tournament import TournamentRules
    from tests.unit.backtest.conftest import build_engine, panel_row

    d0 = date(2026, 3, 2)
    d1 = date(2026, 3, 3)
    rows = [
        {**panel_row(day=d0, ticker="SYN", close=100.0, name="TIGER 차이나전기차레버리지(합성)", trading_value=4.0e8), "mom_60": 0.90},
        {**panel_row(day=d0, ticker="EQ", close=200.0, name="KODEX 반도체레버리지", trading_value=4.0e10), "mom_60": 0.50},
        {**panel_row(day=d0, ticker="DEP", close=150.0, name="KODEX 2차전지산업레버리지", trading_value=2.0e10), "mom_60": 0.30},
        {**panel_row(day=d1, ticker="SYN", close=101.0, name="TIGER 차이나전기차레버리지(합성)", trading_value=4.0e8), "mom_60": 0.90},
        {**panel_row(day=d1, ticker="EQ", close=201.0, name="KODEX 반도체레버리지", trading_value=4.0e10), "mom_60": 0.50},
        {**panel_row(day=d1, ticker="DEP", close=151.0, name="KODEX 2차전지산업레버리지", trading_value=2.0e10), "mom_60": 0.30},
    ]
    panel = pl.DataFrame(rows)
    engine, cal, filt = build_engine(panel, warmup_sessions=1)
    config = BacktestConfig(
        start=d0,
        end=d1,
        capital=1_000_000_000.0,
        scheme=SizingScheme.TOP1,
        k=1,
        filters=filt,
        costs=CostConfig(0.0, 0.0, 0.0),
    )
    rules = TournamentRules(
        name="t",
        start_date=d0,
        end_date=d1,
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
    p27 = BASELINES["P27"]()
    p30 = BASELINES["P30"]()
    cache_p27 = build_session_cache(engine, p27, panel, config)
    cache_p30 = build_session_cache(engine, p30, panel, config)
    assert cache_p30.model_name == "P30"
    score_day = cache_p30.dates[0]
    snap_p30 = cache_p30.snapshots[score_day]
    assert snap_p30.height >= 3
    ctx = DecisionContext(decision_date=score_day, regime=None, capital=1e9, held={}, rules=rules)
    s27 = p27.score(snap_p30, ctx)
    s30 = p30.score(snap_p30, ctx)
    assert isinstance(s27, dict) and isinstance(s30, dict)
    assert s27.get("SYN", 0) > s27.get("EQ", 0)
    assert "SYN" not in s30
    assert max(s30, key=lambda k: s30[k]) == "EQ"
