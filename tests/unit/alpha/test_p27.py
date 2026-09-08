def test_p27_registered_matches_p26_alpha() -> None:
    from src.strategies.registry import STRATEGIES as BASELINES
    from src.alpha.sticky import StickyLeaderModel, load_p27_overlay_mode
    from src.portfolio.constraints import load_p26_exposure_limits, load_p27_exposure_limits

    assert "sticky.mom60_raw" in BASELINES
    p26 = BASELINES["sticky.mom60_concentrated"]()
    p27 = BASELINES["sticky.mom60_raw"]()
    assert isinstance(p27, StickyLeaderModel)
    assert p27.name == "sticky.mom60_raw"
    assert p26.name == "sticky.mom60_concentrated"
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
    from src.strategies.registry import STRATEGIES as BASELINES
    from src.alpha.sticky import StickyLeaderModel
    from src.tournament.simulator import model_requires_path_dependent

    p27 = BASELINES["sticky.mom60_raw"]()
    p21 = BASELINES["sticky.impulse_crash"]()
    p26 = BASELINES["sticky.mom60_concentrated"]()
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
    from src.strategies.registry import STRATEGIES as BASELINES
    from src.alpha.sticky import StickyLeaderModel

    p27 = BASELINES["sticky.mom60_raw"]()
    assert isinstance(p27, StickyLeaderModel)
    assert p27.name == "sticky.mom60_raw"
    assert bool(getattr(p27.config, "same_leader_hold", False)) is False
    assert str(p27.config.mom_col) == "mom_60"
    assert float(p27.config.min_gap) == 0.04
    assert int(p27.config.min_hold) == 2


def test_p27_score_emits_mapping_when_sticky_stays() -> None:
    from datetime import date

    import polars as pl

    from src.alpha.base import DecisionContext
    from src.strategies.registry import STRATEGIES as BASELINES
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
        championship_sleeve="LOTTERY_ON",
    )
    p27 = BASELINES["sticky.mom60_raw"]()
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
    from src.strategies.registry import STRATEGIES as BASELINES
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
    out = BASELINES["sticky.mom60_raw"]().score(snap, ctx)
    assert isinstance(out, PortfolioIntent)
    assert out.kind == CASH_INTENT.kind


def test_p21_keeps_scores_when_all_mom_nonpos() -> None:
    from datetime import date

    import polars as pl

    from src.alpha.base import DecisionContext
    from src.strategies.registry import STRATEGIES as BASELINES
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
    ctx = DecisionContext(decision_date=date(2026, 1, 2), regime=None, capital=1.0e9, held={}, rules=rules, championship_sleeve="LOTTERY_ON")
    p21 = BASELINES["sticky.impulse_crash"]()
    assert bool(getattr(p21.config, "abs_mom_cash", False)) is False
    out = p21.score(snap, ctx)
    assert not isinstance(out, PortfolioIntent)
    assert isinstance(out, dict)
    assert "LEV1" in out and "LEV2" in out


def test_sticky_empty_plus2_returns_cash_intent() -> None:
    from datetime import date

    import polars as pl

    from src.alpha.base import DecisionContext
    from src.strategies.registry import STRATEGIES as BASELINES
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
    for key in ("sticky.impulse_crash", "sticky.mom60_raw"):
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
    from src.strategies.registry import STRATEGIES as BASELINES
    from src.alpha.sticky import StickyLeaderModel

    p27 = BASELINES["sticky.mom60_raw"]()
    assert isinstance(p27, StickyLeaderModel)
    assert p27.name == "sticky.mom60_raw"
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


def test_p27_score_capacity_ignores_grown_equity() -> None:
    from datetime import date

    import polars as pl

    from src.alpha.base import DecisionContext
    from src.strategies.registry import STRATEGIES as BASELINES
    from src.universe.tournament import TournamentRules

    # required_adv(1e9, phi=0.01) = 2.375e10; required_adv(1.5e9) = 3.5625e10
    snap = pl.DataFrame(
        {
            "ticker": ["MID"],
            "name": ["KODEX 반도체레버리지"],
            "mom_60": [0.40],
            "mom_5": [0.02],
            "volume_expansion": [1.0],
            "drawdown_20": [0.0],
            "trading_value": [3.0e10],
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
    grown = DecisionContext(
        decision_date=date(2025, 9, 22),
        regime=None,
        capital=1.5e9,
        held={},
        rules=rules,
        championship_sleeve="LOTTERY_ON",
    )
    start = DecisionContext(
        decision_date=date(2025, 9, 22),
        regime=None,
        capital=1.0e9,
        held={},
        rules=rules,
        championship_sleeve="LOTTERY_ON",
    )
    model_a = BASELINES["sticky.mom60_raw"]()
    model_b = BASELINES["sticky.mom60_raw"]()
    out_grown = model_a.score(snap, grown)
    out_start = model_b.score(snap, start)
    assert isinstance(out_grown, dict)
    assert isinstance(out_start, dict)
    assert out_grown.get("MID") == 0.40
    assert out_start.get("MID") == 0.40


def test_sticky_leader_score_wires_resolve_capacity_capital() -> None:
    import inspect

    from src.strategies.sticky.model_runner import StickyLeaderModel

    src = inspect.getsource(StickyLeaderModel.score)
    cap_at = src.index("apply_capacity_filter")
    prefix = src[:cap_at]
    assert "resolve_capacity_capital(context)" in prefix


def test_p27_championship_invariants_unchanged() -> None:
    from src.alpha.sticky import load_p27_overlay_mode
    from src.portfolio.constraints import load_p27_exposure_limits
    from src.research.activation_state import ACTIVATION_STATE_IS_PRODUCTION_GATE
    from src.strategies.registry import STRATEGIES as BASELINES
    from src.tournament.policy import overlay_should_cash

    p27 = BASELINES["sticky.mom60_raw"]()
    cfg = p27.config
    assert str(cfg.mom_col) == "mom_60"
    assert float(cfg.min_gap) == 0.04
    assert int(cfg.min_hold) == 2
    assert cfg.only_plus_2 is True
    assert cfg.no_inverse is True
    assert bool(cfg.abs_mom_cash) is True
    assert bool(cfg.exclude_synthetic) is True
    assert float(cfg.min_fill_ratio) == 0.25
    assert bool(getattr(cfg, "same_leader_hold", False)) is False
    assert load_p27_overlay_mode() == "identity"
    assert overlay_should_cash("identity", 0.99, 0, 0.50, 5) is False
    assert load_p27_exposure_limits() == (0.95, 1.90, 0.05)
    assert ACTIVATION_STATE_IS_PRODUCTION_GATE is False


def test_p27_factory_crash_rebound_abs_mom_bypass_disabled() -> None:
    from src.alpha.sticky import load_p27_overlay_mode
    from src.strategies.registry import STRATEGIES as BASELINES
    from src.strategies.sticky.model_config import StickyLeaderConfig
    from src.tournament.championship_regime import CHAMPIONSHIP_SLEEVE_IS_PRODUCTION_GATE

    p27 = BASELINES["sticky.mom60_raw"]()
    assert p27.name == "sticky.mom60_raw"
    assert bool(getattr(p27.config, "crash_rebound_abs_mom_bypass", False)) is False
    assert bool(p27.config.abs_mom_cash) is True
    assert load_p27_overlay_mode() == "identity"
    assert CHAMPIONSHIP_SLEEVE_IS_PRODUCTION_GATE is False
    loaded = StickyLeaderConfig.from_yaml({"crash_rebound_abs_mom_bypass": True, "abs_mom_cash": True})
    assert bool(getattr(loaded, "crash_rebound_abs_mom_bypass", False)) is False


def test_p27_score_cashes_on_crash_rebound_sleeve_while_gate_false() -> None:
    from datetime import date

    import polars as pl

    from src.alpha.base import DecisionContext
    from src.portfolio.intent import CASH_INTENT, PortfolioIntent
    from src.strategies.registry import STRATEGIES as BASELINES
    from src.tournament.championship_regime import ChampionshipSleeve
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
    model = BASELINES["sticky.mom60_raw"]()
    model.config.crash_rebound_abs_mom_bypass = True
    ctx = DecisionContext(
        decision_date=date(2026, 1, 2),
        regime=None,
        capital=1.0e9,
        held={},
        rules=rules,
        championship_sleeve=ChampionshipSleeve.CRASH_REBOUND.value,
    )
    out = model.score(snap, ctx)
    assert isinstance(out, PortfolioIntent)
    assert out.kind == CASH_INTENT.kind


def test_sticky_score_routes_attack_sleeve_behind_production_gate() -> None:
    import inspect

    from src.strategies.sticky.model_runner import StickyLeaderModel
    from src.tournament.objective.cutoff_auc import CUTOFF_AUC_IS_PRODUCTION_GATE

    src = inspect.getsource(StickyLeaderModel.score)
    assert "rebound_leader_scores" in src
    assert "apply_attack_sleeve_route" in src
    assert "CUTOFF_AUC_IS_PRODUCTION_GATE" in src
    abs_at = src.index("apply_abs_mom_cash")
    route_at = src.index("apply_attack_sleeve_route")
    assert abs_at < route_at
    assert CUTOFF_AUC_IS_PRODUCTION_GATE is True


def test_sticky_score_routes_crash_rebound_to_mom20_when_gate_true() -> None:
    from datetime import date

    import polars as pl

    from src.alpha.base import DecisionContext
    from src.strategies.registry import STRATEGIES as BASELINES
    from src.tournament.championship_regime import ChampionshipSleeve
    from src.tournament.objective.cutoff_auc import CUTOFF_AUC_IS_PRODUCTION_GATE
    from src.universe.tournament import TournamentRules


    assert CUTOFF_AUC_IS_PRODUCTION_GATE is True
    snap = pl.DataFrame(
        {
            "ticker": ["122630", "233740"],
            "name": ["KODEX 레버리지", "KODEX 코스닥150레버리지"],
            "mom_60": [0.10, 0.05],
            "mom_20": [0.08, 0.25],
            "mom_5": [0.0, 0.0],
            "volume_expansion": [0.1, 0.2],
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
    model = BASELINES["sticky.mom60_raw"]()
    ctx = DecisionContext(
        decision_date=date(2026, 1, 2),
        regime=None,
        capital=1.0e9,
        held={},
        rules=rules,
        championship_sleeve=ChampionshipSleeve.CRASH_REBOUND.value,
    )
    out = model.score(snap, ctx)
    assert isinstance(out, dict)
    assert "233740" in out
    assert out["233740"] >= out.get("122630", float("-inf"))


def test_sticky_score_cashes_when_sleeve_missing_under_production_gate() -> None:
    from datetime import date

    import polars as pl

    from src.alpha.base import DecisionContext
    from src.portfolio.intent import CASH_INTENT, PortfolioIntent
    from src.strategies.registry import STRATEGIES as BASELINES
    from src.tournament.objective.cutoff_auc import CUTOFF_AUC_IS_PRODUCTION_GATE
    from src.universe.tournament import TournamentRules


    assert CUTOFF_AUC_IS_PRODUCTION_GATE is True
    snap = pl.DataFrame(
        {
            "ticker": ["122630"],
            "name": ["KODEX 레버리지"],
            "mom_60": [0.20],
            "mom_20": [0.10],
            "mom_5": [0.0],
            "volume_expansion": [0.1],
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
    model = BASELINES["sticky.mom60_raw"]()
    ctx = DecisionContext(
        decision_date=date(2026, 1, 2),
        regime=None,
        capital=1.0e9,
        held={},
        rules=rules,
        championship_sleeve=None,
    )
    out = model.score(snap, ctx)
    assert isinstance(out, PortfolioIntent)
    assert out.kind == CASH_INTENT.kind
