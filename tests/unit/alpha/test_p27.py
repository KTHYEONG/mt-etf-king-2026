def test_p27_registered_matches_p26_alpha() -> None:
    from src.alpha.baselines import BASELINES
    from src.alpha.sticky import StickyLeaderModel, load_p27_overlay_mode
    from src.portfolio.constraints import load_p26_exposure_limits, load_p27_exposure_limits

    assert "P27" in BASELINES
    p26 = BASELINES["P26"]()
    p27 = BASELINES["P27"]()
    assert isinstance(p27, StickyLeaderModel)
    assert p27.name == "P27"
    assert p26.name == "P26"
    c26 = p26.config
    c27 = p27.config
    assert str(c27.mom_col) == "mom_60"
    assert float(c27.cash_drawdown) == 0.0
    assert float(c27.min_gap) == 0.04
    assert int(c27.min_hold) == 2
    assert float(c27.impulse_gap) == 0.0
    assert c27.only_plus_2 is True
    assert c27.no_inverse is True
    assert c27.collapse_family is False
    assert str(c27.mom_col) == str(c26.mom_col)
    assert float(c27.cash_drawdown) == float(c26.cash_drawdown)
    assert float(c27.min_gap) == float(c26.min_gap)
    assert int(c27.min_hold) == int(c26.min_hold)
    assert float(c27.impulse_gap) == float(c26.impulse_gap)
    assert load_p27_overlay_mode() == "identity"
    assert load_p27_exposure_limits() == load_p26_exposure_limits()
    assert load_p27_exposure_limits() == (0.95, 1.90, 0.05)
    assert not hasattr(p27, "allocate") or not callable(getattr(p27, "allocate", None))


def test_sticky_leader_declares_path_dependent_and_reset_trackers() -> None:
    from src.alpha.baselines import BASELINES
    from src.alpha.sticky import StickyLeaderModel
    from src.tournament.simulator import model_requires_path_dependent

    p27 = BASELINES["P27"]()
    p21 = BASELINES["P21"]()
    p26 = BASELINES["P26"]()
    assert isinstance(p27, StickyLeaderModel)
    for model in (p27, p21, p26):
        assert model.path_dependent is True
        assert model.scores_path_independent is False
        assert model_requires_path_dependent(model) is True
        model._held = "X"
        model._hold_len = 7
        model.reset_trackers()
        assert model._held is None
        assert int(model._hold_len) == 0


def test_p27_factory_same_leader_hold_disabled() -> None:
    from src.alpha.baselines import BASELINES
    from src.alpha.sticky import StickyLeaderModel

    p27 = BASELINES["P27"]()
    assert isinstance(p27, StickyLeaderModel)
    assert p27.name == "P27"
    assert bool(getattr(p27.config, "same_leader_hold", False)) is False
    assert str(p27.config.mom_col) == "mom_60"
    assert float(p27.config.min_gap) == 0.04
    assert int(p27.config.min_hold) == 2


def test_p27_score_emits_mapping_when_sticky_stays() -> None:
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
        held={"FAST": 0.95},
        rules=rules,
    )
    p27 = BASELINES["P27"]()
    scores = p27.score(snap, ctx)
    assert not isinstance(scores, PortfolioIntent)
    assert isinstance(scores, dict)
    assert scores
    w = weights_from_scores(scores, SizingScheme.TOP1, k=1)
    assert set(w.keys()) == {"FAST"}


def test_p27_cashes_when_all_mom_nonpos() -> None:
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
            "trading_value": [1.0e11, 1.0e11],
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
    out = BASELINES["P27"]().score(snap, ctx)
    assert isinstance(out, PortfolioIntent)
    assert out.kind == CASH_INTENT.kind


def test_p21_keeps_scores_when_all_mom_nonpos() -> None:
    from datetime import date

    import polars as pl

    from src.alpha.base import DecisionContext
    from src.alpha.baselines import BASELINES
    from src.portfolio.intent import PortfolioIntent
    from src.universe.tournament import TournamentRules

    snap = pl.DataFrame(
        {
            "ticker": ["LEV1", "LEV2"],
            "name": ["KODEX 레버리지", "KODEX 코스닥150레버리지"],
            "mom_20": [-0.05, -0.02],
            "mom_60": [-0.05, -0.02],
            "mom_5": [0.0, 0.0],
            "volume_expansion": [0.1, 0.1],
            "drawdown_20": [0.0, 0.0],
            "trading_value": [1.0e11, 1.0e11],
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
    p21 = BASELINES["P21"]()
    assert bool(getattr(p21.config, "abs_mom_cash", False)) is False
    out = p21.score(snap, ctx)
    assert not isinstance(out, PortfolioIntent)
    assert isinstance(out, dict)
    assert "LEV1" in out and "LEV2" in out


def test_sticky_empty_plus2_returns_cash_intent() -> None:
    from datetime import date

    import polars as pl

    from src.alpha.base import DecisionContext
    from src.alpha.baselines import BASELINES
    from src.portfolio.intent import CASH_INTENT, PortfolioIntent
    from src.universe.tournament import TournamentRules

    snap = pl.DataFrame(
        {
            "ticker": ["CASH1"],
            "name": ["KODEX 200"],
            "mom_20": [0.50],
            "mom_60": [0.50],
            "mom_5": [0.10],
            "volume_expansion": [1.0],
            "drawdown_20": [0.0],
            "trading_value": [1.0e11],
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
    ctx = DecisionContext(decision_date=date(2026, 1, 2), regime=None, capital=1.0e9, held={"CASH1": 0.95}, rules=rules)
    for key in ("P21", "P27"):
        out = BASELINES[key]().score(snap, ctx)
        assert isinstance(out, PortfolioIntent), key
        assert out.kind == CASH_INTENT.kind, key


def test_from_yaml_parses_abs_mom_cash() -> None:
    from src.strategies.sticky.model import StickyLeaderConfig

    cfg = StickyLeaderConfig.from_yaml(
        {
            "abs_mom_cash": True,
            "exclude_synthetic": True,
            "min_fill_ratio": 0.25,
            "same_leader_hold": True,
        }
    )
    assert cfg.abs_mom_cash is True
    assert cfg.exclude_synthetic is True
    assert abs(float(cfg.min_fill_ratio) - 0.25) < 1e-12
    assert cfg.same_leader_hold is True
    blank = StickyLeaderConfig.from_yaml({})
    assert blank.abs_mom_cash is False
    assert blank.exclude_synthetic is False
    assert float(blank.min_fill_ratio) == 0.0
    assert blank.same_leader_hold is False


def test_p27_factory_fillability_and_abs_mom() -> None:
    from src.alpha.baselines import BASELINES
    from src.alpha.sticky import StickyLeaderModel

    p27 = BASELINES["P27"]()
    assert isinstance(p27, StickyLeaderModel)
    assert p27.name == "P27"
    assert str(p27.config.mom_col) == "mom_60"
    assert float(p27.config.min_gap) == 0.04
    assert int(p27.config.min_hold) == 2
    assert float(p27.config.impulse_gap) == 0.0
    assert float(p27.config.cash_drawdown) == 0.0
    assert p27.config.only_plus_2 is True
    assert p27.config.no_inverse is True
    assert p27.config.collapse_family is False
    assert bool(p27.config.same_leader_hold) is False
    assert bool(p27.config.abs_mom_cash) is True
    assert bool(p27.config.exclude_synthetic) is True
    assert abs(float(p27.config.min_fill_ratio) - 0.25) < 1e-12
    assert tuple(p27.config.exclude_name_tokens) == ()
    assert not hasattr(p27, "allocate") or not callable(getattr(p27, "allocate", None))
