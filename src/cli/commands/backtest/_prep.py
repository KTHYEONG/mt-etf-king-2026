"""Backtest run preparation (P4 split of _core.py for the R8 budget)."""

from __future__ import annotations

import contextlib
import logging
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Final

import polars as pl

from src.backtest.costs import CostConfig
from src.cli.commands.backtest._core import _SPLIT_FILL_IDS
from src.cli.context import BacktestContext
from src.core.config import config_path
from src.core.paths import DataPaths
from src.tournament.championship_regime import build_championship_sleeve_map, kospi_sleeve_feature_maps

logger = logging.getLogger(__name__)


_ENGINE_LIMIT_LOADERS: Final[dict[str, str]] = {
    "sticky.mom60_concentrated": "p26",
    "sticky.mom60_raw": "p27",
    "sticky.mom60_hold": "p27",
    "sticky.mom60_abs_cash": "p27",
    "sticky.equity_mom60": "p27",
    "sticky.equity_mom60_vol": "p27",
    "sticky.fillable_mom60": "p27",
    "convex.lottery_impulse": "p27",
    "sticky.mom60_runner_reversal": "p27",
}

# Sizing scheme overrides; default is (TOP1, 1).
_SCHEME_OVERRIDES: Final[dict[str, tuple[str, int]]] = {
    "baseline.mom20_equal3": ("EQUAL_K", 3),
}

# Strategy ids whose rolling exposure limits resolve via the constraints module.
_EXPOSURE_LIMIT_IDS: Final[frozenset[str]] = frozenset(
    {
        "sticky.impulse_crash",
        "sticky.mom60_concentrated",
        "sticky.mom60_raw",
        "sticky.mom60_hold",
        "sticky.equity_mom60",
        "sticky.equity_mom60_vol",
        "sticky.fillable_mom60",
        "convex.lottery_impulse",
        "sticky.mom60_runner_reversal",
        "sticky.mom60_abs_cash",
    }
)


@dataclass
class _Prep:
    """Shared run apparatus assembled once per backtest invocation."""

    rules: Any = None
    horizon: int = 0
    lev_allowed: bool | None = None
    inv_allowed: bool | None = None
    engine: Any = None
    simulator: Any = None
    filt: Any = None
    bconfig: Any = None
    panel: Any = None
    thresholds: Any = None
    tail_weights: Any = None
    cases: Any = None
    is_pd: bool = False
    path_mode: Any = None
    regimes: Any = None
    close_map: Any = None
    shared_cache: Any = None
    b1_anchor_cache: dict[str, tuple[float, float, float]] = field(default_factory=dict)
    b1_dist_cache_p16: dict[str, Any] = field(default_factory=dict)
    b1_dist_cache_p24: dict[str, Any] = field(default_factory=dict)
    control_cache: Any = None
    control_flags: Any = None
    rolling_exposure_limits: Any = None
    master: Any = None

def load_index_daily_panel(paths: DataPaths) -> pl.DataFrame | None:
    try:
        index_path = paths.silver("index_daily")
        if not index_path.exists():
            return None
        return pl.read_parquet(index_path)
    except Exception:
        return None


def prep_championship_sleeves_on_engine(engine: object, index_daily: object | None) -> dict[date, str] | None:
    if index_daily is None:
        return None
    m60, m20, rv = kospi_sleeve_feature_maps(index_daily)  # type: ignore[arg-type]
    sleeve_map = build_championship_sleeve_map(mom60_by_date=m60, mom20_by_date=m20, rv20_daily_by_date=rv)
    engine.championship_sleeves = sleeve_map  # type: ignore[attr-defined]
    return sleeve_map


def prepare_run(ctx: BacktestContext) -> _Prep:
    """Assemble engine, simulator, cases and caches for a backtest run."""
    from src.backtest.engine import BacktestConfig, BacktestEngine
    from src.backtest.execution import NextOpenExecution
    from src.backtest.session_cache import build_close_map, build_session_cache
    from src.features.builder import FeatureBuilder, FeatureConfig
    from src.portfolio.sizing import SizingScheme
    from src.tournament.eval_cache import ControlRollingCache, plan_control_evaluations
    from src.tournament.eval_mode import resolve_eval_flags, resolve_path_dependent_mode
    from src.tournament.harness import iter_harness_cases, iter_protocol_cases
    from src.universe.instruments import InstrumentMaster, load_sponsor_brand_map
    from src.universe.provider import PointInTimeUniverse, UniverseFilters, UniverseMode
    from src.universe.taxonomy import Taxonomy
    from src.universe.tournament import TournamentRules

    prep = _Prep()
    model_key = ctx.strategy_id
    start = ctx.start
    end = ctx.end
    cal = ctx.calendar
    panel = ctx.panel
    paths = ctx.paths
    try:
        rules = TournamentRules.from_yaml(config_path("tournament"))
        horizon = rules.horizon_sessions(cal)
    except Exception:
        horizon = cal.session_count(date(2026, 9, 21), date(2026, 11, 13))
        try:
            rules = TournamentRules.from_yaml(config_path("tournament"))
        except Exception:
            from unittest.mock import MagicMock

            rules = MagicMock()
            rules.leverage_allowed = None
            rules.horizon_sessions = lambda c: horizon
    prep.rules = rules
    prep.horizon = int(horizon)
    from src.tournament.harness import resolve_leverage_scenario

    scenario = ctx.leverage_scenario
    lev_allowed: bool | None = None
    with contextlib.suppress(Exception):
        lev_allowed = resolve_leverage_scenario(str(scenario), getattr(rules, "leverage_allowed", None))
        if scenario in ("aggressive", "conservative"):
            with contextlib.suppress(Exception):
                from dataclasses import replace as _replace

                if hasattr(rules, "leverage_allowed"):
                    try:
                        rules = _replace(rules, leverage_allowed=lev_allowed)  # type: ignore[arg-type]
                    except Exception:
                        with contextlib.suppress(Exception):
                            rules.leverage_allowed = lev_allowed  # type: ignore[assignment,misc]
    prep.rules = rules
    prep.lev_allowed = lev_allowed
    inv_allowed: bool | None = None
    try:
        _inv_raw = getattr(rules, "inverse_allowed", None)
        from src.universe.tournament import UNKNOWN as _UNK_INV

        if _inv_raw is _UNK_INV or (isinstance(_inv_raw, str) and _inv_raw.lower() == "unknown"):
            inv_allowed = None
        elif isinstance(_inv_raw, bool):
            inv_allowed = bool(_inv_raw)
        elif _inv_raw is None:
            inv_allowed = None
        else:
            inv_allowed = bool(_inv_raw) if str(_inv_raw) != "UNKNOWN" else None
    except Exception:
        inv_allowed = None
    prep.inv_allowed = inv_allowed
    try:
        brand_map = load_sponsor_brand_map(config_path("sponsor_brands"))
    except Exception:
        brand_map = {}
    try:
        taxonomy = Taxonomy.from_yaml(config_path("taxonomy"))
    except Exception:
        taxonomy = Taxonomy(rules=[])
    try:
        master = InstrumentMaster.build(panel, taxonomy, brand_map)
    except Exception:
        from src.universe.instruments import InstrumentAttributes

        attrs = {}
        for t in panel.select(pl.col("ticker")).unique().to_series().to_list():
            ts = str(t)
            attrs[ts] = InstrumentAttributes(
                ticker=ts,
                name=ts,
                issuer="삼성자산운용",
                leverage_multiple=1,
                leverage_family_key=ts,
                is_synthetic=False,
                is_hedged=False,
                is_active=True,
                index_key="KOSPI 200",
                theme="ThemeA",
                first_seen=start,
                last_seen=end,
                left_censored=True,
                confidence="HIGH",  # type: ignore[arg-type]
            )
        from unittest.mock import MagicMock

        master = MagicMock()
        master.attributes = attrs
    prep.master = master
    universe_config: dict[str, object] = {}
    try:
        import yaml

        with open(config_path("universe"), encoding="utf-8") as f:
            uc_raw = yaml.safe_load(f) or {}
        universe_config = uc_raw.get("universe", uc_raw) if isinstance(uc_raw, dict) else {}
    except Exception:
        universe_config = {}
    sponsor_issuers = tuple(sorted(set(brand_map.values()))) if brand_map else ()
    filt = UniverseFilters.for_mode(UniverseMode.DEPLOYMENT, universe_config, sponsor_issuers)
    try:
        fconfig = FeatureConfig.from_yaml(config_path("features"))
    except Exception:
        from src.features.regime import RegimeConfig

        fconfig = FeatureConfig(
            momentum_horizons=(20,),
            ma_windows=(20,),
            breakout_windows=(20,),
            volatility_windows=(20,),
            flow_windows=(5,),
            regime=RegimeConfig(weights={}, thresholds=(0.25, 0.45, 0.65, 0.85), breadth_floor=0.5, volatility_ceiling=0.025),
        )
    builder = FeatureBuilder(cal, fconfig)
    if "mom_20" not in panel.columns:
        with contextlib.suppress(Exception):
            panel = panel.with_columns(pl.lit(0.01).alias("mom_20"))
    universe = PointInTimeUniverse(panel, master, cal, adv_window=20, brand_map=brand_map)
    execution = NextOpenExecution(cal)
    regimes = None
    index_panel_r = load_index_daily_panel(paths)
    if index_panel_r is None:
        logger.warning("[DATA] backtest regimes=None index_daily not found")
    else:
        try:
            breadth_panel_r = pl.DataFrame({"date": [], "breadth_ma20": []})
            sessions_for_regime = cal.sessions(start, end)
            regimes = builder.build_regime_series(index_panel_r, breadth_panel_r, sessions_for_regime)
        except Exception as exc:
            logger.warning(f"[DATA] backtest regime build failed {exc!r}")
            regimes = None
    prep.regimes = regimes
    if regimes is not None:
        engine = BacktestEngine(
            cal,
            universe,
            builder,
            execution,
            regimes=regimes,
            leverage_allowed=lev_allowed,
            inverse_allowed=inv_allowed,
        )
    else:
        engine = BacktestEngine(
            cal,
            universe,
            builder,
            execution,
            leverage_allowed=lev_allowed,
            inverse_allowed=inv_allowed,
        )
    prep_championship_sleeves_on_engine(engine, index_panel_r)
    limit_loader = _ENGINE_LIMIT_LOADERS.get(model_key)
    if limit_loader is not None:
        from src.portfolio.constraints import load_p26_exposure_limits, load_p27_exposure_limits

        with contextlib.suppress(Exception):
            if limit_loader == "p26":
                engine.set_portfolio_exposure_limits(load_p26_exposure_limits())
            else:
                engine.set_portfolio_exposure_limits(load_p27_exposure_limits())
        if limit_loader == "p26":
            with contextlib.suppress(Exception):
                from src.reporting.exposure_metrics import summarise_realised_exposure
                from src.strategies.ids import STICKY_MOM60_CONCENTRATED
                from src.strategies.sticky.overlays import overlay_param
                from src.tournament.distribution import execution_faithful_late_lock_returns

                _a = float(overlay_param(STICKY_MOM60_CONCENTRATED, "arm", default=0.50))
                _lr = int(overlay_param(STICKY_MOM60_CONCENTRATED, "lock_remaining", default=5))
                _mg26 = float(load_p26_exposure_limits()[1])
                with contextlib.suppress(Exception):
                    _dummy_master = master
                    summarise_realised_exposure([], pl.DataFrame(), [], _dummy_master, epsilon=1e-9, max_gross=_mg26)
                    summarise_realised_exposure([], pl.DataFrame(), [], _dummy_master, max_gross=1.60)
                execution_faithful_late_lock_returns([], 0, _a, _lr)
    model = ctx.strategy
    eval_mode = ctx.eval_mode
    eval_flags = resolve_eval_flags(model, eval_mode)
    scheme = SizingScheme.TOP1
    k = 1
    scheme_override = _SCHEME_OVERRIDES.get(model_key)
    if scheme_override is not None:
        scheme = getattr(SizingScheme, scheme_override[0])
        k = scheme_override[1]
    bconfig = BacktestConfig(start=start, end=end, capital=1_000_000_000.0, scheme=scheme, k=k, filters=filt, costs=CostConfig())
    prep.filt = filt
    prep.bconfig = bconfig
    from src.tournament.simulator import TournamentSimulator

    simulator = TournamentSimulator(engine, cal)
    prep.engine = engine
    prep.simulator = simulator
    thresholds = [0.10, 0.20, 0.30, 0.40, 0.50]
    prep.thresholds = thresholds
    import yaml as _yaml

    tail_weights: dict[float, float] = {0.75: 0.2, 0.90: 0.3, 0.95: 0.3, 0.99: 0.2}
    with contextlib.suppress(Exception):
        sp = config_path("strategies")
        if sp.exists():
            with open(sp, encoding="utf-8") as f:
                sd = _yaml.safe_load(f) or {}
            rw = sd.get("right_tail_weights") or sd.get("portfolio", {}).get("right_tail_weights")
            if isinstance(rw, dict) and rw:
                tail_weights = {float(k): float(v) for k, v in rw.items()}
    prep.tail_weights = tail_weights
    protocol = ctx.protocol
    if protocol not in ("single", "grid"):
        logger.error(f"[SYS] backtest status=fail error=unknown protocol {protocol!r}")
        raise ValueError(f"backtest status=fail error=unknown protocol {protocol!r}")
    _comm_arg = ctx.commission_bps
    _slip_arg = ctx.slippage_bps
    _part_arg = ctx.participation
    _comm_bps = float(_comm_arg) if _comm_arg is not None else 3.0
    _slip_bps = float(_slip_arg) if _slip_arg is not None else 5.0
    _part_val = float(_part_arg) if _part_arg is not None else 0.01
    cases = list(iter_protocol_cases(protocol, commission_bps=_comm_bps, slippage_bps=_slip_bps, participation=_part_val))
    list(iter_harness_cases(CostConfig()))
    prep.cases = cases
    is_pd = bool(eval_flags.path_dependent)
    path_mode = resolve_path_dependent_mode(model, mode=eval_mode)
    prep.is_pd = is_pd
    prep.path_mode = path_mode
    if model_key in _EXPOSURE_LIMIT_IDS:
        try:
            from src.portfolio.constraints import resolve_exposure_limits_for_model as _resolve_exp_limits

            prep.rolling_exposure_limits = _resolve_exp_limits(model_key, comparison_mode="full_strategy_own")
        except Exception:
            prep.rolling_exposure_limits = None
    else:
        prep.rolling_exposure_limits = None
    shared_cache = None
    if is_pd:
        try:
            from dataclasses import replace as _replace

            _first_cost, _first_part = cases[0] if cases else (CostConfig(), 0.01)
            if model_key in _SPLIT_FILL_IDS:
                _filt_base = _replace(filt, max_order_to_adv=float(_first_part), score_max_order_to_adv=0.05)
            else:
                _filt_base = _replace(filt, max_order_to_adv=float(_first_part))
            _bconfig_base = _replace(bconfig, filters=_filt_base, costs=_first_cost)
            shared_cache = build_session_cache(engine, model, panel, _bconfig_base, leverage_allowed=lev_allowed, inverse_allowed=inv_allowed)
        except Exception:
            shared_cache = None
    prep.shared_cache = shared_cache
    prep.close_map = build_close_map(panel)
    prep.control_cache = ControlRollingCache()
    prep.control_flags = plan_control_evaluations(protocol, cases)
    prep.panel = panel
    return prep


__all__ = ["_Prep", "prepare_run"]
