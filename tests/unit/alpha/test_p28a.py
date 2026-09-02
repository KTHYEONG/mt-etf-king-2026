def test_apply_same_leader_hold_returns_hold_when_leader_equals_held() -> None:
    from src.alpha.sticky import apply_same_leader_hold
    from src.portfolio.intent import HOLD_INTENT, PortfolioIntent

    scores = {"AAA": 0.20, "BBB": 0.10}
    out = apply_same_leader_hold(scores, "AAA", True)
    assert isinstance(out, PortfolioIntent)
    assert out.kind == HOLD_INTENT.kind
    assert out.weights == {}
    assert scores == {"AAA": 0.20, "BBB": 0.10}


def test_apply_same_leader_hold_returns_scores_on_switch() -> None:
    from src.alpha.sticky import apply_same_leader_hold
    from src.portfolio.intent import PortfolioIntent

    scores = {"AAA": 0.10, "BBB": 0.25}
    out = apply_same_leader_hold(scores, "AAA", True)
    assert not isinstance(out, PortfolioIntent)
    assert out == {"AAA": 0.10, "BBB": 0.25}


def test_apply_same_leader_hold_initial_entry_is_target_scores() -> None:
    from src.alpha.sticky import apply_same_leader_hold
    from src.portfolio.intent import PortfolioIntent

    scores = {"AAA": 0.20}
    out = apply_same_leader_hold(scores, None, True)
    assert not isinstance(out, PortfolioIntent)
    assert out == {"AAA": 0.20}


def test_apply_same_leader_hold_preserves_cash_intent() -> None:
    from src.alpha.sticky import apply_same_leader_hold
    from src.portfolio.intent import CASH_INTENT, PortfolioIntent

    out = apply_same_leader_hold(CASH_INTENT, "AAA", True)
    assert out is CASH_INTENT or (isinstance(out, PortfolioIntent) and out.kind == "cash")


def test_apply_same_leader_hold_disabled_is_noop() -> None:
    from src.alpha.sticky import apply_same_leader_hold
    from src.portfolio.intent import PortfolioIntent

    scores = {"AAA": 0.20, "BBB": 0.10}
    out = apply_same_leader_hold(scores, "AAA", False)
    assert not isinstance(out, PortfolioIntent)
    assert out == {"AAA": 0.20, "BBB": 0.10}


def test_apply_same_leader_hold_empty_scores_stays_empty() -> None:
    from src.alpha.sticky import apply_same_leader_hold
    from src.portfolio.intent import PortfolioIntent

    out = apply_same_leader_hold({}, "AAA", True)
    assert not isinstance(out, PortfolioIntent)
    assert out == {}


def test_p28a_factory_matches_p27_except_hold_flag() -> None:
    from src.alpha.baselines import BASELINES
    from src.alpha.sticky import StickyLeaderModel
    from src.portfolio.constraints import load_p27_exposure_limits, resolve_exposure_limits_for_model

    assert "P28A" in BASELINES
    p27 = BASELINES["P27"]()
    p28 = BASELINES["P28A"]()
    assert isinstance(p28, StickyLeaderModel)
    assert p28.name == "P28A"
    assert p27.name == "P27"
    c27 = p27.config
    c28 = p28.config
    assert bool(c27.same_leader_hold) is False
    assert bool(c28.same_leader_hold) is True
    assert str(c28.mom_col) == str(c27.mom_col) == "mom_60"
    assert float(c28.min_gap) == float(c27.min_gap) == 0.04
    assert int(c28.min_hold) == int(c27.min_hold) == 2
    assert float(c28.impulse_gap) == float(c27.impulse_gap) == 0.0
    assert float(c28.cash_drawdown) == float(c27.cash_drawdown) == 0.0
    assert c28.only_plus_2 is True and c27.only_plus_2 is True
    assert c28.no_inverse is True and c27.no_inverse is True
    assert c28.collapse_family is False and c27.collapse_family is False
    assert resolve_exposure_limits_for_model("P28A", comparison_mode="full_strategy_own") == load_p27_exposure_limits()
    assert load_p27_exposure_limits() == (0.95, 1.90, 0.05)
    assert not hasattr(p28, "allocate") or not callable(getattr(p28, "allocate", None))


def test_p28a_score_emits_hold_intent_when_sticky_stays() -> None:
    from datetime import date

    import polars as pl

    from src.alpha.base import DecisionContext
    from src.alpha.baselines import BASELINES
    from src.portfolio.intent import HOLD_INTENT, PortfolioIntent
    from src.universe.tournament import TournamentRules

    snap = pl.DataFrame(
        {
            "ticker": ["SLOW", "FAST"],
            "name": ["KODEX 레버리지", "KODEX 코스닥150레버리지"],
            "mom_20": [0.50, 0.10],
            "mom_60": [0.10, 0.40],
            "mom_5": [0.01, 0.02],
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
    ctx = DecisionContext(
        decision_date=date(2026, 1, 2),
        regime=None,
        capital=1.0e9,
        held={"FAST": 0.95},
        rules=rules,
    )
    p28 = BASELINES["P28A"]()
    out = p28.score(snap, ctx)
    assert isinstance(out, PortfolioIntent)
    assert out.kind == HOLD_INTENT.kind


def test_p28a_score_emits_scores_when_sticky_switches() -> None:
    from datetime import date

    import polars as pl

    from src.alpha.base import DecisionContext
    from src.alpha.baselines import BASELINES
    from src.portfolio.intent import PortfolioIntent
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
    ctx = DecisionContext(
        decision_date=date(2026, 1, 2),
        regime=None,
        capital=1.0e9,
        held={"SLOW": 0.95},
        rules=rules,
    )
    p28 = BASELINES["P28A"]()
    p28.restore_state("SLOW", 10)
    out = p28.score(snap, ctx)
    assert not isinstance(out, PortfolioIntent)
    assert isinstance(out, dict)
    w = weights_from_scores(out, SizingScheme.TOP1, k=1)
    assert set(w.keys()) == {"FAST"}


def test_sticky_config_from_yaml_ignores_same_leader_hold_knob() -> None:
    from src.alpha.sticky import StickyLeaderConfig

    cfg = StickyLeaderConfig.from_yaml({"same_leader_hold": True, "min_gap": 0.04, "min_hold": 2})
    assert bool(getattr(cfg, "same_leader_hold", False)) is False
    assert float(cfg.min_gap) == 0.04
    assert int(cfg.min_hold) == 2


def test_p28a_exposure_limits_reuse_p27() -> None:
    from src.portfolio.constraints import (
        alpha_equal_exposure_limits,
        load_p27_exposure_limits,
        resolve_exposure_limits_for_model,
    )

    p27 = load_p27_exposure_limits()
    p28 = resolve_exposure_limits_for_model("P28A", comparison_mode="full_strategy_own")
    eq = resolve_exposure_limits_for_model("P28A", comparison_mode="alpha_equal")
    assert p28 == p27 == (0.95, 1.90, 0.05)
    assert eq == alpha_equal_exposure_limits()
