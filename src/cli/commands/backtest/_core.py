"""Shared backtest execution core (P4 decomposition of src/cli/_impl.py).

Holds the run preparation and the per-grid-cell execution loop. Per-model
differences live in the family modules as table-registered hooks
this core
contains no strategy-identity branches.
"""

from __future__ import annotations

import contextlib
import logging
from collections.abc import Callable
from dataclasses import dataclass
from dataclasses import replace as dataclass_replace
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Final

from src.cli.constants import (
    CONVEXITY_ADOPTION_MODELS,
    LOTTERY_ADOPTION_MODELS,
    STICKY_ADOPTION_MODELS,
)
from src.cli.context import BacktestCellBundle, BacktestContext
from src.core.config import config_path
from src.reporting.results import make_backtest_run_id
from src.strategies.registry import STRATEGIES

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from datetime import date

    import polars as pl

    from src.backtest.costs import CostConfig
    from src.backtest.engine import BacktestConfig, BacktestEngine
    from src.cli.commands.backtest._prep import _Prep
    from src.core.calendar import TradingCalendar
    from src.strategies.protocol import StrategyProtocol
    from src.tournament.distribution import ReturnDistribution
    from src.tournament.simulator import TournamentSimulator
    from src.universe.provider import UniverseFilters


def _fmt(v: float) -> str:
    return f"{float(v):.3f}"


# Preflight steps in original evaluation order: (member ids, missing label,
# exc label). Every step runs the same gold/silver span check; only the
# labels differ. "{model_key}" formats to the semantic id, as before.
# R4: labels are semantic ids, never legacy P/B/M codes.
_PREFLIGHT_STEPS: Final[tuple[tuple[frozenset[str], str, str], ...]] = (
    (frozenset({"portfolio.momentum_vehicle"}), "portfolio.momentum_vehicle", "portfolio.momentum_vehicle"),
    (frozenset({"portfolio.momentum_confidence"}), "portfolio.momentum_confidence", "portfolio.momentum_confidence"),
    (frozenset({"portfolio.lottery_exposure"}), "portfolio.lottery_exposure", "portfolio.lottery_exposure"),
    (frozenset(LOTTERY_ADOPTION_MODELS), "portfolio.lottery_rebalance", "portfolio.lottery_rebalance"),
    (frozenset(STICKY_ADOPTION_MODELS), "{model_key}", "sticky.leader_base"),
    (frozenset(CONVEXITY_ADOPTION_MODELS), "portfolio.convexity_hold", "portfolio.convexity_hold"),
)

# Engine exposure-limit loaders per strategy (fail-closed application).
# Strategies needing the split-fill universe-filter tweak.
_SPLIT_FILL_IDS: Final[frozenset[str]] = frozenset({"sticky.split_fill_lock"})

# Strategies needing the full-span artifact fallback run.
_FULLSPAN_FALLBACK_IDS: Final[frozenset[str]] = frozenset(
    {
        "sticky.mom60_raw",
        "sticky.mom60_hold",
        "sticky.mom60_abs_cash",
        "sticky.equity_mom60",
        "sticky.equity_mom60_vol",
        "sticky.fillable_mom60",
        "convex.lottery_impulse",
        "sticky.mom60_runner_reversal",
    }
)




@dataclass
class _Cell:
    """Mutable per-grid-cell namespace shared by the loop and family hooks."""

    model_key: str
    eval_mode: str
    start: date
    end: date
    cal: TradingCalendar
    panel: pl.DataFrame
    horizon: int
    engine: BacktestEngine
    model: StrategyProtocol
    simulator: TournamentSimulator
    thresholds: list[float]
    tail_weights: dict[float, float]
    lev_allowed: bool | None
    inv_allowed: bool | None
    regimes: Any
    close_map: Any
    shared_cache: Any
    rolling_exposure_limits: tuple[float, float, float] | None
    path_mode: Any
    master: Any
    cost_cfg: CostConfig
    participation: float
    filt_case: UniverseFilters
    case_config: BacktestConfig
    trace_sink: Any
    rolling: Any
    dist: ReturnDistribution
    run_id: str
    meta: dict[str, object]
    summary: dict[str, object]
    control_cache: Any
    control_flags: Any
    cell_idx: int
    b1_anchor_cache: dict[str, tuple[float, float, float]]
    b1_dist_cache_p16: dict[str, Any]
    b1_dist_cache_p24: dict[str, Any]
    ctx: BacktestContext | None = None
    prep: _Prep | None = None
    forensics: bool = False
    forensics_payload: dict[str, object] | None = None
    windows_df: Any = None
    daily: Any = None
    trades: Any = None


def run_preflight(ctx: BacktestContext) -> bool:
    """Gold/silver span preflight; False means the run must stop with code 1."""
    from src.tournament.distribution import preflight_features_span_ok

    for members, missing_label, exc_label in _PREFLIGHT_STEPS:
        if ctx.strategy_id not in members:
            continue
        gold_path = ctx.paths.gold("etf_features")
        silver_path = ctx.paths.silver("etf_daily")
        if not gold_path.exists() or not silver_path.exists():
            logger.error(
                "[SYS] backtest status=fail error="
                f"{missing_label.format(model_key=ctx.strategy_id)} requires gold features and silver panel (INV-10-5)"
            )
            return False
        try:
            import polars as preflight_pl

            gold_span = preflight_pl.scan_parquet(gold_path).select(
                preflight_pl.col("date").min().alias("min"),
                preflight_pl.col("date").max().alias("max"),
            ).collect()
            silver_span = preflight_pl.scan_parquet(silver_path).select(
                preflight_pl.col("date").min().alias("min"),
                preflight_pl.col("date").max().alias("max"),
            ).collect()
            if not preflight_features_span_ok(
                gold_span[0, "min"],
                gold_span[0, "max"],
                silver_span[0, "min"],
                silver_span[0, "max"],
            ):
                logger.error("[SYS] backtest status=fail error=gold features span does not cover silver (INV-10-5)")
                return False
        except Exception as exc:
            logger.error(f"[SYS] backtest status=fail error={exc_label} preflight failed {exc!r}")
            return False
    return True




# Strategies whose realised-exposure max-gross overrides the model default.
_MAX_GROSS_OVERRIDE_LOADERS: Final[dict[str, str]] = {
    "sticky.mom60_concentrated": "p26",
}


def _run_common_tail(
    ctx: BacktestContext,
    prep: _Prep,
    model_key: str,
    cell: _Cell,
    _cell_idx: int,
) -> None:
    """Shared per-cell tail: full-span artifacts, windows, exposure, B0 gates."""
    from src.reporting.exposure_metrics import (
        artifact_max_gross_for_model,
        prefer_execution_gross_count,
        summarise_realised_exposure,
    )
    from src.reporting.timeseries import build_window_timeseries
    from src.tournament.distribution import ReturnDistribution
    from src.tournament.eval_cache import protocol_cell_key
    from src.tournament.objective import ObjectiveGateConfig, evaluate_objective_gates

    summary = cell.summary
    rolling = cell.rolling
    engine = prep.engine
    model = ctx.strategy
    panel = prep.panel
    case_config = cell.case_config
    horizon = prep.horizon
    cal = ctx.calendar
    start = ctx.start
    end = ctx.end
    master = prep.master
    simulator = prep.simulator
    thresholds = prep.thresholds
    tail_weights = prep.tail_weights
    _bt_daily = None
    _bt_trades = None
    _bt = getattr(rolling, "backtest", None)
    if _bt is not None:
        _bt_daily = _bt.daily
        _bt_trades = _bt.trades
    elif model_key in _FULLSPAN_FALLBACK_IDS:
        try:
            _artifact_bt = engine.run(model, panel, case_config, trace=None, close_map=prep.close_map, open_map=prep.open_map)
            _bt_daily = _artifact_bt.daily
            _bt_trades = _artifact_bt.trades
        except Exception as _exc_artifact_bt:
            logger.warning(f"[EVAL] full-span artifact run failed model={model_key} error={_exc_artifact_bt!r}")
    cell.daily = _bt_daily
    cell.trades = _bt_trades
    _windows_df = None
    try:
        _windows_df = build_window_timeseries(
            rolling,
            cal.sessions(start, end),
            ruin_threshold=-0.25,
        )
        summary["windows_rows"] = int(_windows_df.height)
    except Exception:
        _windows_df = None
    cell.windows_df = _windows_df
    with contextlib.suppress(Exception):
        if _bt_trades is not None:
            # prev: _exposure_max_gross = 1.60
            _exposure_max_gross = artifact_max_gross_for_model(model_key)
            _override_loader = _MAX_GROSS_OVERRIDE_LOADERS.get(model_key)
            if _override_loader is not None:
                try:
                    from src.portfolio.constraints import load_p26_exposure_limits as _lpe26

                    _exposure_max_gross = float(_lpe26()[1])
                except Exception:
                    _exposure_max_gross = 1.60
            _exposure = summarise_realised_exposure(
                cal.sessions(start, end),
                _bt_trades,
                (),
                master,
                epsilon=1e-9,
                max_gross=float(_exposure_max_gross),
            )
            summary["realised_exposure"] = {
                "active_name_mean": float(_exposure.active_name_mean),
                "active_family_mean": float(_exposure.active_family_mean),
                "multi_family_rate": float(_exposure.multi_family_rate),
                "invested_weight_mean": float(_exposure.invested_weight_mean),
                "effective_gross_mean": float(_exposure.effective_gross_mean),
                "effective_gross_q90": float(_exposure.effective_gross_q90),
                "effective_gross_max": float(_exposure.effective_gross_max) if _exposure.effective_gross_max is not None else None,
                "gross_violation_count": int(_exposure.gross_violation_count) if _exposure.gross_violation_count is not None else None,
                "mult2_filled_notional_rate": float(_exposure.mult2_filled_notional_rate),
                "turnover": float(_exposure.turnover),
                "unfilled_session_rate": float(_exposure.unfilled_session_rate),
            }
            # prev: summary["gross_violation_count"] = int(_exposure.gross_violation_count)
            _gvc = summary.get("gross_violation_count")
            _resolved_gvc = prefer_execution_gross_count(_gvc if isinstance(_gvc, int) or _gvc is None else None, _exposure.gross_violation_count)
            summary["gross_violation_count"] = int(_resolved_gvc) if _resolved_gvc is not None else None
            summary["effective_gross_max"] = float(_exposure.effective_gross_max) if _exposure.effective_gross_max is not None else None
        _cfg_obj = ObjectiveGateConfig.from_yaml(config_path("gates"))
        _do_control = bool(prep.control_flags[_cell_idx]) if _cell_idx < len(prep.control_flags) else True
        _b0_dist = cell.dist
        _res_obj = None
        if _do_control:
            try:
                _b0_key = protocol_cell_key(cell.cost_cfg, cell.participation)
                def _b0_factory() -> object:
                    _bm = STRATEGIES["baseline.buy_hold"]()
                    _br = simulator.run_rolling(
                        _bm,
                        panel,
                        case_config,
                        horizon=horizon,
                        path_dependent=False,
                        close_map=prep.close_map,
                        open_map=prep.open_map,
                    )
                    return ReturnDistribution.summarise(
                        name="baseline.buy_hold",
                        returns=list(_br.returns),
                        horizon=horizon,
                        thresholds=thresholds,
                        tail_weights=tail_weights,
                    )
                _b0_dist = prep.control_cache.get_or_run(_b0_key, _b0_factory)
                if not isinstance(_b0_dist, ReturnDistribution):
                    raise TypeError("cache miss")
            except Exception:
                try:
                    _b0_model = STRATEGIES["baseline.buy_hold"]()
                    _b0_rolling = simulator.run_rolling(
                        _b0_model,
                        panel,
                        case_config,
                        horizon=horizon,
                        path_dependent=False,
                        close_map=prep.close_map,
                        open_map=prep.open_map,
                    )
                    _b0_dist = ReturnDistribution.summarise(
                        name="baseline.buy_hold",
                        returns=list(_b0_rolling.returns),
                        horizon=horizon,
                        thresholds=thresholds,
                        tail_weights=tail_weights,
                    )
                except Exception:
                    _b0_dist = cell.dist
            try:
                _res_obj = evaluate_objective_gates(cell.dist, _b0_dist, _cfg_obj)
            except Exception:
                _res_obj = None
        if _res_obj is not None:
            summary["objective_gate_status"] = str(_res_obj.status)
            summary["objective_gate_fails"] = list(_res_obj.failures)
            summary["objective_ruin_probability"] = float(_res_obj.ruin_probability)


ForensicsHook = Callable[[_Cell], None]


def run_cells(
    ctx: BacktestContext,
    prep: _Prep,
    forensics_hook: ForensicsHook | None,
) -> list[BacktestCellBundle]:
    """Execute the per-grid-cell loop; per-model forensics via hook (no I/O)."""
    from src.tournament.distribution import ReturnDistribution
    from src.universe.provider import UniverseFilters

    bundles: list[BacktestCellBundle] = []
    model_key = ctx.strategy_id
    model = ctx.strategy
    start = ctx.start
    end = ctx.end
    cal = ctx.calendar
    panel = prep.panel
    horizon = prep.horizon
    engine = prep.engine
    simulator = prep.simulator
    filt = prep.filt
    bconfig = prep.bconfig
    thresholds = prep.thresholds
    tail_weights = prep.tail_weights
    eval_mode = ctx.eval_mode
    lev_allowed = prep.lev_allowed
    inv_allowed = prep.inv_allowed
    regimes = prep.regimes
    close_map = prep.close_map
    open_map = prep.open_map
    rolling_exposure_limits = prep.rolling_exposure_limits
    is_pd = prep.is_pd
    path_mode = prep.path_mode
    cases = prep.cases
    for _cell_idx, (cost_cfg, participation) in enumerate(cases):
        if model_key in _SPLIT_FILL_IDS:
            filt_case = UniverseFilters(
                mode=filt.mode,
                warmup_sessions=filt.warmup_sessions,
                adv_window=filt.adv_window,
                capital=filt.capital,
                max_position_weight=filt.max_position_weight,
                max_order_to_adv=float(participation),
                allow_leverage=filt.allow_leverage,
                allow_inverse=filt.allow_inverse,
                issuer_whitelist=filt.issuer_whitelist,
                manifest=filt.manifest,
                score_max_order_to_adv=0.05,
            )
        else:
            filt_case = UniverseFilters(
                mode=filt.mode,
                warmup_sessions=filt.warmup_sessions,
                adv_window=filt.adv_window,
                capital=filt.capital,
                max_position_weight=filt.max_position_weight,
                max_order_to_adv=float(participation),
                allow_leverage=filt.allow_leverage,
                allow_inverse=filt.allow_inverse,
                issuer_whitelist=filt.issuer_whitelist,
                manifest=filt.manifest,
            )
        case_config = dataclass_replace(
            bconfig,
            start=start,
            end=end,
            capital=1_000_000_000.0,
            scheme=bconfig.scheme,
            k=bconfig.k,
            filters=filt_case,
            costs=cost_cfg,
        )
        trace_sink = None
        if ctx.trace:
            try:
                from src.core.trace import InMemoryTraceSink as _TraceSinkCls

                trace_sink = _TraceSinkCls()
            except Exception:
                trace_sink = None
        # 각 셀은 자신의 case_config에서 캐시를 해소한다 (비용 축은 키에서 제외되므로 동일 참여율은 적중)
        cell_cache = None
        if prep.cache_registry is not None:
            try:
                cell_cache = prep.cache_registry.get_or_build(
                    engine,
                    model,
                    panel,
                    case_config,
                    leverage_allowed=lev_allowed,
                    inverse_allowed=inv_allowed,
                )
            except Exception:
                cell_cache = None
        if is_pd:
            rolling = simulator.run_rolling(
                model,
                panel,
                case_config,
                horizon=horizon,
                path_dependent=True,
                path_dependent_mode=path_mode,
                session_cache=cell_cache,
                leverage_allowed=lev_allowed,
                inverse_allowed=inv_allowed,
                trace=None,
                close_map=close_map,
                open_map=open_map,
                exposure_limits=rolling_exposure_limits,
            )
        else:
            rolling = simulator.run_rolling(
                model,
                panel,
                case_config,
                horizon=horizon,
                path_dependent=False,
                leverage_allowed=lev_allowed,
                inverse_allowed=inv_allowed,
                trace=trace_sink,
                close_map=close_map,
                open_map=open_map,
            )
        dist = ReturnDistribution.summarise(
            name=model_key,
            returns=list(rolling.returns),
            horizon=horizon,
            thresholds=thresholds,
            tail_weights=tail_weights,
            givebacks=list(getattr(rolling, "givebacks", ())),
        )
        logger.info(
            f"[EVAL] backtest model={model_key} start={start} end={end} horizon={horizon} "
            f"commission_bps={_fmt(float(cost_cfg.commission_bps or 0.0))} "
            f"slippage_bps={_fmt(float(cost_cfg.slippage_bps or 0.0))} "
            f"participation={_fmt(participation)} "
            f"n_windows={dist.n_windows} n_effective={dist.n_effective} "
            + " ".join(f"q{int(k * 100):02d}={_fmt(v)}" for k, v in sorted(dist.quantiles.items()))
            + f" cvar_05={_fmt(dist.cvar_05)} giveback_median={_fmt(dist.giveback_median)} giveback_q90={_fmt(dist.giveback_q90)} rts={_fmt(dist.right_tail_score)}"
            + " ".join(f"p>{_fmt(t)}={_fmt(v)}" for t, v in sorted(dist.exceedance.items()))
        )
        base_id = make_backtest_run_id(model_key, start, end)
        suffix = f"{int(float(cost_cfg.commission_bps or 0)*100):04d}_{int(float(cost_cfg.slippage_bps or 0)*100):04d}_{int(float(participation)*1000):04d}"
        run_id = f"{base_id}_{suffix}"
        meta = {
            "model": model_key,
            "strategy_id": model_key,
            "legacy_model_id": model_key,
            "start": str(start),
            "end": str(end),
            "horizon": int(horizon),
            "commission_bps": float(cost_cfg.commission_bps or 0.0),
            "slippage_bps": float(cost_cfg.slippage_bps or 0.0),
            "participation": float(participation),
            "created_at": datetime.now(UTC).isoformat(),
        }
        summary = {
            "n_windows": int(dist.n_windows),
            "n_effective": int(dist.n_effective),
            "quantiles": {str(k): float(v) for k, v in sorted(dist.quantiles.items())},
            "exceedance": {str(k): float(v) for k, v in sorted(dist.exceedance.items())},
            "cvar_05": float(dist.cvar_05),
            "giveback_median": float(dist.giveback_median),
            "giveback_q90": float(dist.giveback_q90),
            "right_tail_score": float(dist.right_tail_score),
        }
        cell = _Cell(
            ctx=ctx,
            prep=prep,
            model_key=model_key,
            eval_mode=eval_mode,
            start=start,
            end=end,
            cal=cal,
            panel=panel,
            horizon=int(horizon),
            engine=engine,
            model=model,
            simulator=simulator,
            thresholds=thresholds,
            tail_weights=tail_weights,
            lev_allowed=lev_allowed,
            inv_allowed=inv_allowed,
            regimes=regimes,
            close_map=close_map,
            shared_cache=cell_cache,
            rolling_exposure_limits=prep.rolling_exposure_limits,
            path_mode=path_mode,
            master=prep.master,
            forensics=bool(ctx.forensics),
            b1_anchor_cache=prep.b1_anchor_cache,
            b1_dist_cache_p16=prep.b1_dist_cache_p16,
            b1_dist_cache_p24=prep.b1_dist_cache_p24,
            cost_cfg=cost_cfg,
            participation=float(participation),
            filt_case=filt_case,
            case_config=case_config,
            trace_sink=trace_sink,
            rolling=rolling,
            dist=dist,
            run_id=run_id,
            meta=meta,
            summary=summary,
            control_cache=prep.control_cache,
            control_flags=prep.control_flags,
            cell_idx=_cell_idx,
        )
        if forensics_hook is not None:
            forensics_hook(cell)
        _run_common_tail(ctx, prep, model_key, cell, _cell_idx)
        bundles.append(
            BacktestCellBundle(
                run_id=run_id,
                meta=dict(cell.meta),
                summary=dict(cell.summary),
                rolling=cell.rolling,
                dist=cell.dist,
                case_config=cell.case_config,
                engine=engine,
                model=model,
                panel=panel,
                shared_cache=cell_cache,
                horizon=int(horizon),
                leverage_allowed=lev_allowed,
                inverse_allowed=inv_allowed,
                trace_sink=cell.trace_sink,
                windows_df=cell.windows_df,
                daily=cell.daily,
                trades=cell.trades,
                forensics_payload=cell.forensics_payload,
                cost_label=suffix,
            )
        )
    return bundles


__all__ = [
    "ForensicsHook",
    "_Cell",
    "_fmt",
    "run_cells",
    "run_preflight",
]
